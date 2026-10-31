"""FastAPI service for DeepDetect — async video analysis, auth, batch uploads, Grad-CAM."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from src import database as db
from src.live_detector import LiveDetectionSession
from src.auth import (
    PasswordResetConfirm,
    PasswordResetRequest,
    RefreshRequest,
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    confirm_password_reset,
    create_password_reset_token,
    create_user,
    get_current_user,
    get_optional_user,
    get_user_by_email,
    init_auth_tables,
    issue_tokens,
    redeem_refresh_token,
    verify_password,
)
from src.forensics import compute_video_hashes
from src.inference import load_detector_model, predict_video, transform_pipeline
from src.ingestion import MAX_VIDEO_BYTES, VIDEO_EXTENSIONS, download_video_url
from src.models_zoo import SUPPORTED_MODEL_NAMES


ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path(tempfile.gettempdir()) / "deepdetect-api"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="DeepDetect AI API", version="2.0.0")
_cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "DEEPDETECT_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$|^https://[a-zA-Z0-9.-]+\.onrender\.com$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single-worker pool keeps GPU/model serialised
worker_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deepdetect")
model_state: dict = {"model": None, "device": None}
_full_winner_path = ROOT / "artifacts_full" / "mobilenet_xgboost.joblib"
_smoke_winner_path = ROOT / "artifacts_smoke" / "mobilenet_xgboost.joblib"
_trained_winner_path = (
    _full_winner_path if _full_winner_path.is_file() else _smoke_winner_path
)
MODEL_NAME = os.getenv(
    "DEEPDETECT_MODEL",
    "mobilenet_xgboost" if _trained_winner_path.is_file() else "resnet18_bilstm",
).lower()
MODEL_PATH = os.getenv(
    "DEEPDETECT_MODEL_PATH",
    str(
        _trained_winner_path
        if MODEL_NAME in {"mobilenet_xgb", "mobilenet_xgboost", "mobilenet"}
        and _trained_winner_path.is_file()
        else ROOT / "deepfake_model.pt"
    ),
)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    db.fail_stale_jobs()
    init_auth_tables()


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class UrlAnalysisRequest(BaseModel):
    url: str
    num_frames: int = 32


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _probe_video(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("The uploaded file could not be opened as a video.")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("The video has invalid or unreadable stream metadata.")
    hashes = compute_video_hashes(path)
    return {
        "frame_count": frame_count,
        "fps": float(fps),
        "width": width,
        "height": height,
        "duration_seconds": frame_count / fps,
        "size_bytes": os.path.getsize(path),
        "sha256": hashes["sha256"],
        "md5": hashes["md5"],
    }


def _start_job(
    path: str,
    filename: str,
    source: str,
    num_frames: int,
    user_id: Optional[str] = None,
    batch_id: Optional[str] = None,
) -> str:
    if not 8 <= num_frames <= 48:
        raise ValueError("Frame count must be between 8 and 48.")
    metadata = _probe_video(path)
    job_id = uuid.uuid4().hex
    db.insert_job(
        job_id=job_id,
        filename=filename,
        source=source,
        video_path=path,
        num_frames=num_frames,
        metadata=metadata,
        user_id=user_id,
        batch_id=batch_id,
    )
    return job_id


def _queue_analysis(job_id: str) -> None:
    """Submit inference after the HTTP response has been sent."""
    try:
        worker_pool.submit(_analyze_job, job_id)
    except RuntimeError as exc:
        db.set_job_failed(job_id, f"Analysis worker is unavailable: {exc}")


def _image_data_url(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _visual_explanation(image: Image.Image) -> tuple[str, str]:
    """Create a fast visual artifact map without blocking model inference."""
    frame = np.asarray(image.convert("RGB"))
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    importance = cv2.normalize(
        cv2.GaussianBlur(cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_32F)), (0, 0), 5),
        None, 0.0, 1.0, cv2.NORM_MINMAX,
    )
    heatmap = cv2.resize(importance, (width, height), interpolation=cv2.INTER_CUBIC)
    overlay = cv2.applyColorMap(np.uint8(heatmap * 255), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 0.55, overlay, 0.45, 0)
    image = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    peak = int(np.argmax(cv2.resize(importance, (3, 3), interpolation=cv2.INTER_AREA)))
    location = ("top" if peak // 3 == 0 else "middle" if peak // 3 == 1 else "bottom") + " " + (
        "left" if peak % 3 == 0 else "center" if peak % 3 == 1 else "right"
    )
    return _image_data_url(image), f"Visual artifact analysis highlights the {location} region as the strongest high-frequency area. This is an evidence cue, not pixel-level proof of manipulation."


def _backfill_occlusion_explanations(job: dict) -> None:
    """Add explanations to older MobileNet jobs created before occlusion support."""
    if MODEL_NAME not in {"mobilenet", "mobilenet_xgb", "mobilenet_xgboost"}:
        return
    result = job.get("result") or {}
    suspicious = result.get("suspicious") or {}
    frames = suspicious.get("suspicious_frames") or []
    evidence_frames = suspicious.get("evidence_frames") or []
    if not frames:
        scores = result.get("frame_scores") or []
        fps = float((job.get("metadata") or {}).get("fps") or 25.0)
        frames = [
            {
                "frame_index": index,
                "timestamp_sec": index / fps,
                "fake_score": float(score),
                "neighbour_mse": 0.0,
                "neighbour_delta": 0.0,
                "reason": "Sampled frame score is above the 50% screening threshold.",
            }
            for index, score in enumerate(scores)
            if float(score) >= 0.5
        ]
        suspicious["suspicious_frames"] = frames
        suspicious["suspicious_count"] = len(frames)
        result["suspicious"] = suspicious
    if not evidence_frames:
        evidence_frames = frames or [
            {
                "frame_index": index,
                "timestamp_sec": index / float((job.get("metadata") or {}).get("fps") or 25.0),
                "fake_score": float(score),
                "neighbour_mse": 0.0,
                "neighbour_delta": 0.0,
                "reason": "Highest-scoring sampled frame shown as visual evidence; it is below the suspicious threshold.",
            }
            for index, score in enumerate(result.get("frame_scores") or [])
        ]
        evidence_frames = sorted(
            evidence_frames, key=lambda frame: frame["fake_score"], reverse=True
        )[:8]
        suspicious["evidence_frames"] = evidence_frames
        result["suspicious"] = suspicious
    if not evidence_frames or any(frame.get("gradcam_data_url") for frame in evidence_frames):
        return
    images = []
    for data_url in result.get("face_frames") or []:
        try:
            images.append(Image.open(BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGB"))
        except (IndexError, ValueError):
            return
    if not images:
        return
    if model_state["model"] is None:
        model_state["model"], model_state["device"] = load_detector_model(
            model_path=MODEL_PATH, backbone_type=MODEL_NAME
        )
    for item in evidence_frames:
        overlay_url, explanation = _visual_explanation(
            images[int(item["frame_index"])]
        )
        item["gradcam_data_url"] = overlay_url
        item["reason"] = explanation
    db.set_job_completed(job["id"], result)


def _analyze_job(job_id: str) -> None:
    job = db.get_job(job_id)
    if not job:
        return
    db.set_job_running(job_id)

    try:
        if model_state["model"] is None:
            model_state["model"], model_state["device"] = load_detector_model(
                model_path=MODEL_PATH,
                backbone_type=MODEL_NAME,
            )

        result = predict_video(
            video_path=job["video_path"],
            model=model_state["model"],
            device=model_state["device"],
            num_frames=job["num_frames"],
            backbone_type=MODEL_NAME,
        )

        # Grad-CAM + suspicious frame detection
        fps = job.get("metadata", {}).get("fps", 25.0) or 25.0
        if hasattr(model_state["model"], "parameters"):
            from src.gradcam import compute_suspicious_frames
            suspicious_report = compute_suspicious_frames(
                face_pil_list=result.get("face_frames", []),
                frame_scores=result.get("frame_probabilities", []),
                fps=fps,
                model=model_state["model"],
                device=model_state["device"],
                transform=transform_pipeline,
            )
            suspicious = suspicious_report.to_dict(include_images=True)
        else:
            frame_scores = [
                float(value) for value in result.get("frame_probabilities", [])
            ]
            face_frames = result.get("face_frames", [])
            suspicious_frames = []
            for index, score in enumerate(frame_scores):
                if score < 0.5:
                    continue
                evidence = {
                    "frame_index": index,
                    "timestamp_sec": index * (1.0 / fps),
                    "fake_score": score,
                    "neighbour_mse": 0.0,
                    "neighbour_delta": 0.0,
                    "reason": (
                        "Sampled frame score is above the 50% "
                        "screening threshold."
                    ),
                    "face_data_url": (
                        _image_data_url(face_frames[index])
                        if index < len(face_frames)
                        else ""
                    ),
                }
                suspicious_frames.append(
                    evidence
                )
            evidence_frames = []
            for index in sorted(
                range(len(frame_scores)),
                key=lambda item: frame_scores[item],
                reverse=True,
            )[:8]:
                evidence = {
                    "frame_index": index,
                    "timestamp_sec": index * (1.0 / fps),
                    "fake_score": frame_scores[index],
                    "neighbour_mse": 0.0,
                    "neighbour_delta": 0.0,
                    "reason": (
                        "Highest-scoring sampled frame shown as visual evidence; "
                        "it is below the suspicious threshold."
                    ),
                    "face_data_url": (
                        _image_data_url(face_frames[index])
                        if index < len(face_frames)
                        else ""
                    ),
                }
                evidence_frames.append(evidence)
            if suspicious_frames and face_frames:
                explain_indices = sorted(
                    range(len(suspicious_frames)),
                    key=lambda item: suspicious_frames[item]["fake_score"],
                    reverse=True,
                )[:8]
                for item in explain_indices:
                    overlay_url, explanation = _visual_explanation(
                        face_frames[suspicious_frames[item]["frame_index"]]
                    )
                    suspicious_frames[item]["gradcam_data_url"] = overlay_url
                    suspicious_frames[item]["reason"] = explanation
            if evidence_frames and face_frames:
                for item in evidence_frames:
                    overlay_url, explanation = _visual_explanation(
                        face_frames[item["frame_index"]]
                    )
                    item["gradcam_data_url"] = overlay_url
                    item["reason"] = explanation
            suspicious = {
                "total_frames": result.get("num_frames", 0),
                "fps": fps,
                "suspicious_count": len(suspicious_frames),
                "detector_name": _get_detector_name(),
                "suspicious_frames": suspicious_frames,
                "evidence_frames": evidence_frames,
                "frame_scores": frame_scores,
                "note": (
                    "The active MobileNet-XGBoost model does not expose "
                    "gradients. Heatmaps show visual high-frequency artifact "
                    "cues and are not proof of manipulation."
                ),
            }

        db.set_job_completed(
            job_id,
            {
                "fake_score": result["fake_probability"],
                "real_score": result["real_probability"],
                "frames_analyzed": result["num_frames"],
                "frame_scores": result.get("frame_probabilities", []),
                "face_frames": [
                    _image_data_url(img) for img in result.get("face_frames", [])
                ],
                "suspicious": suspicious,
                "detector_name": _get_detector_name(),
                "assessment_note": (
                    "Model screening output only. Not proof of authenticity "
                    "or validated manipulation timestamps."
                ),
            },
        )
    except Exception as exc:
        db.set_job_failed(job_id, str(exc))


def _get_detector_name() -> str:
    try:
        from src.face_detector import get_detector_name
        return get_detector_name()
    except Exception:
        return "Unavailable"


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/api/auth/register", response_model=TokenResponse, status_code=201)
def register(req: UserRegisterRequest):
    if len(req.password) < 8:
        raise HTTPException(422, "Password must be at least 8 characters.")
    try:
        user = create_user(req.email, req.username, req.password)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return issue_tokens(user)


@app.post("/api/auth/login", response_model=TokenResponse)
def login(req: UserLoginRequest):
    user = get_user_by_email(req.email)
    if not user or not verify_password(req.password, user["hashed_pw"]):
        raise HTTPException(401, "Incorrect email or password.")
    if not user["is_active"]:
        raise HTTPException(403, "Account disabled.")
    return issue_tokens(user)


@app.post("/api/auth/refresh", response_model=TokenResponse)
def refresh_token(req: RefreshRequest):
    try:
        return redeem_refresh_token(req.refresh_token)
    except ValueError as exc:
        raise HTTPException(401, str(exc)) from exc


@app.post("/api/auth/password-reset/request")
def request_password_reset(req: PasswordResetRequest):
    token = create_password_reset_token(req.email)
    # In production: send email. For local dev, return the token directly.
    return {
        "detail": "If that email is registered you will receive a reset link.",
        "_dev_token": token or None,
    }


@app.post("/api/auth/password-reset/confirm")
def confirm_reset(req: PasswordResetConfirm):
    if len(req.new_password) < 8:
        raise HTTPException(422, "Password must be at least 8 characters.")
    ok = confirm_password_reset(req.token, req.new_password)
    if not ok:
        raise HTTPException(400, "Reset token invalid or expired.")
    return {"detail": "Password updated successfully."}


@app.get("/api/auth/me")
def me(current_user: dict = Depends(get_current_user)):
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "username": current_user["username"],
        "created_at": current_user["created_at"],
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_state["model"] is not None,
        "model": MODEL_NAME,
        "model_path": MODEL_PATH,
        "detector": _get_detector_name(),
    }


@app.get("/api/models")
def available_models():
    """Expose the trained model slots and the model currently used by the API."""
    return {
        "active": MODEL_NAME,
        "supported": list(SUPPORTED_MODEL_NAMES),
        "artifacts": {
            "resnet18_bilstm": str(ROOT / "artifacts_full" / "resnet18_bilstm.pt"),
            "efficientnet_v2_bilstm": str(
                ROOT / "artifacts_full" / "efficientnet_v2_bilstm.pt"
            ),
            "mobilenet_xgboost": str(
                ROOT / "artifacts_full" / "mobilenet_xgboost.joblib"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Live Deepfake Detection (WebCam WebSocket)
# ---------------------------------------------------------------------------

@app.get("/api/live/status")
def live_status():
    """Configuration and model status for live webcam deepfake detection."""
    return {
        "status": "ready",
        "model": "MobileNetV3 + XGBoost",
        "detector": _get_detector_name(),
        "sequence_frames": 32,
        "input_size": "224x224",
        "ws_endpoint": "/api/ws/live",
    }


@app.websocket("/api/ws/live")
@app.websocket("/ws/live")
async def websocket_live_detection(websocket: WebSocket):
    """
    WebSocket endpoint for live webcam deepfake detection.
    Receives frame images in base64 format, performs real-time face detection with SCRFD,
    maintains a 32-frame sliding temporal buffer, and runs continuous MobileNetV3 + XGBoost inference.
    """
    await websocket.accept()
    session = LiveDetectionSession(target_frames=32)
    try:
        while True:
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
            except Exception:
                await websocket.send_json({"type": "error", "message": "Invalid JSON format"})
                continue

            msg_type = data.get("type", "frame")
            if msg_type == "frame":
                payload = data.get("image")
                if not payload:
                    await websocket.send_json({"type": "error", "message": "Missing image payload"})
                    continue

                frame_rgb = await asyncio.to_thread(session.decode_image, payload)
                if frame_rgb is None:
                    await websocket.send_json({"type": "error", "message": "Failed to decode frame image"})
                    continue

                res = await asyncio.to_thread(session.process_frame, frame_rgb)
                await websocket.send_json(res)

            elif msg_type == "reset":
                session.reset()
                await websocket.send_json({
                    "type": "reset_ack",
                    "message": "Live 32-frame sequence buffer reset successfully",
                })

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logging.getLogger(__name__).warning("WebSocket live session error: %s", exc)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Single upload
# ---------------------------------------------------------------------------

@app.post("/api/analyses/upload", status_code=202)
async def create_upload_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    num_frames: int = Query(32, ge=8, le=48),
    current_user: Optional[dict] = Depends(get_optional_user),
):
    filename = Path(file.filename or "video.mp4").name
    suffix = Path(filename).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(415, "Unsupported video extension.")

    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    try:
        with path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_VIDEO_BYTES:
                    raise HTTPException(413, "Videos must be 500 MB or smaller.")
                out.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    user_id = current_user["id"] if current_user else None
    try:
        job_id = await asyncio.to_thread(
            _start_job, str(path), filename, "Local upload", num_frames, user_id
        )
    except ValueError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc

    background_tasks.add_task(_queue_analysis, job_id)
    return db._public_job(db.get_job(job_id))


# ---------------------------------------------------------------------------
# Batch upload
# ---------------------------------------------------------------------------

@app.post("/api/analyses/batch", status_code=202)
async def create_batch_analysis(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    num_frames: int = Query(32, ge=8, le=48),
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """Upload up to 10 videos in one request; returns a batch_id + list of job stubs."""
    if len(files) > 10:
        raise HTTPException(422, "Maximum 10 videos per batch.")
    if len(files) == 0:
        raise HTTPException(422, "No files provided.")

    batch_id = db.create_batch_id()
    user_id = current_user["id"] if current_user else None
    job_stubs = []

    for file in files:
        filename = Path(file.filename or "video.mp4").name
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            await file.close()
            continue  # skip unsupported files silently

        path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        size = 0
        try:
            with path.open("wb") as out:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_VIDEO_BYTES:
                        path.unlink(missing_ok=True)
                        break
                    out.write(chunk)
            if path.exists():
                job_id = await asyncio.to_thread(
                    _start_job, str(path), filename, "Batch upload",
                    num_frames, user_id, batch_id,
                )
                background_tasks.add_task(_queue_analysis, job_id)
                job_stubs.append(db._public_job(db.get_job(job_id)))
        except Exception:
            path.unlink(missing_ok=True)
        finally:
            await file.close()

    if not job_stubs:
        raise HTTPException(422, "No valid video files were processed.")

    return {"batch_id": batch_id, "jobs": job_stubs}


@app.get("/api/analyses/batch/{batch_id}")
def get_batch(batch_id: str):
    summary = db.get_batch_summary(batch_id)
    if summary["total"] == 0:
        raise HTTPException(404, "Batch not found.")
    return summary


# ---------------------------------------------------------------------------
# URL analysis
# ---------------------------------------------------------------------------

@app.post("/api/analyses/url", status_code=202)
async def create_url_analysis(
    request: UrlAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    if not 8 <= request.num_frames <= 48:
        raise HTTPException(422, "Frame count must be between 8 and 48.")
    try:
        from urllib.parse import urlparse
        host = urlparse(request.url).hostname or "unknown"
        path, filename, _ = await asyncio.to_thread(
            download_video_url, request.url
        )
        user_id = current_user["id"] if current_user else None
        job_id = await asyncio.to_thread(
            _start_job, path, filename,
            f"Direct video link from {host}",
            request.num_frames, user_id,
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc

    background_tasks.add_task(_queue_analysis, job_id)
    return db._public_job(db.get_job(job_id))


# ---------------------------------------------------------------------------
# Job CRUD
# ---------------------------------------------------------------------------

@app.get("/api/analyses")
def list_analyses(current_user: dict = Depends(get_current_user)):
    jobs = db.get_jobs_for_user(current_user["id"])
    return [db._public_job(j) for j in jobs]


@app.get("/api/analyses/{job_id}")
def get_analysis(
    job_id: str,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Analysis not found or expired.")
    if job.get("status") == "completed":
        _backfill_occlusion_explanations(job)
        job = db.get_job(job_id) or job
    return db._public_job(job)


@app.get("/api/analyses/{job_id}/video")
def get_analysis_video(
    job_id: str,
    request: Request,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    job = db.get_job(job_id)
    if not job or not Path(job["video_path"]).is_file():
        raise HTTPException(404, "Video not found.")
    
    file_path = Path(job["video_path"])
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")

    if range_header and range_header.startswith("bytes="):
        try:
            h = range_header.replace("bytes=", "").split("-")
            start = int(h[0]) if h[0] else 0
            end = int(h[1]) if h[1] else file_size - 1
        except ValueError:
            start, end = 0, file_size - 1

        start = max(0, min(start, file_size - 1))
        end = max(start, min(end, file_size - 1))
        content_length = (end - start) + 1

        def iterfile():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(64 * 1024, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Type": "video/mp4",
        }
        return StreamingResponse(iterfile(), status_code=206, headers=headers)

    return FileResponse(
        file_path,
        filename=job["filename"],
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


@app.get("/api/analyses/{job_id}/preview")
def get_analysis_preview(
    job_id: str,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """Stream an MJPEG preview when the original video codec is browser-incompatible."""
    del current_user
    job = db.get_job(job_id)
    if not job or not Path(job["video_path"]).is_file():
        raise HTTPException(404, "Video not found.")

    file_path = Path(job["video_path"])
    fps = float((job.get("metadata") or {}).get("fps") or 25.0)
    delay = max(0.01, min(0.2, 1.0 / fps))

    def iter_preview():
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            return
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                encoded, buffer = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
                )
                if not encoded:
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
                time.sleep(delay)
        finally:
            cap.release()

    return StreamingResponse(
        iter_preview(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/api/analyses/{job_id}/report")
def download_pdf_report(
    job_id: str,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """Generate and stream a forensic PDF report for a completed analysis."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Analysis not found.")
    if job["status"] != "completed" or not job.get("result"):
        raise HTTPException(409, "Analysis not yet complete.")

    from src.report import generate_forensic_pdf

    meta = job["metadata"] or {}
    result_data = job["result"] or {}

    case_info = {
        "case_id": f"DD-{job_id[:10].upper()}",
        "examiner": current_user["username"] if current_user else "DeepDetect AI",
        "agency": "DeepDetect Forensic Lab",
    }
    video_meta = {
        "filename": job["filename"],
        "resolution": f"{meta.get('width', '?')}×{meta.get('height', '?')}",
        "fps": meta.get("fps", 0.0),
        "duration": meta.get("duration_seconds", 0.0),
        "size_mb": meta.get("size_bytes", 0) / (1024 * 1024),
        "sha256": meta.get("sha256", "N/A"),
        "md5": meta.get("md5", "N/A"),
        "source": job["source"],
    }
    report_result = {
        "is_fake": result_data.get("fake_score", 0) >= 0.5,
        "fake_probability": result_data.get("fake_score", 0),
        "confidence": result_data.get("fake_score", 0) * 100,
        "num_frames": result_data.get("frames_analyzed", 0),
        "frame_probabilities": result_data.get("frame_scores", []),
        "suspicious_count": result_data.get("suspicious", {}).get("suspicious_count", 0),
    }

    # Decode face frame data-URLs back to PIL for the PDF
    face_images = []
    for data_url in (result_data.get("face_frames") or [])[:4]:
        try:
            b64 = data_url.split(",", 1)[1]
            face_images.append(Image.open(BytesIO(base64.b64decode(b64))))
        except Exception:
            pass

    # Decode Grad-CAM overlays from suspicious frames
    gradcam_images = []
    suspicious = result_data.get("suspicious", {}).get("suspicious_frames", [])
    for sf in suspicious[:4]:
        du = sf.get("gradcam_data_url", "")
        if du:
            try:
                b64 = du.split(",", 1)[1]
                gradcam_images.append(Image.open(BytesIO(base64.b64decode(b64))))
            except Exception:
                pass

    pdf_bytes = generate_forensic_pdf(
        case_info=case_info,
        video_meta=video_meta,
        result=report_result,
        face_images=face_images,
        fft_images=gradcam_images,
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="deepdetect-{job_id[:10]}.pdf"'
            )
        },
    )


@app.delete("/api/analyses/{job_id}", status_code=204)
def delete_analysis(
    job_id: str,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    video_path = db.delete_job(job_id)
    if video_path is None:
        raise HTTPException(404, "Analysis not found.")
    Path(video_path).unlink(missing_ok=True)
    return None


# ---------------------------------------------------------------------------
# Frontend SPA catch-all
# ---------------------------------------------------------------------------

@app.get("/")
def frontend_root():
    if FRONTEND_DIST.is_dir() and (FRONTEND_DIST / "index.html").is_file():
        return FileResponse(FRONTEND_DIST / "index.html")
    return {
        "service": "DeepDetect API",
        "status": "ok",
        "message": "Frontend build is not present. Run npm run build in frontend.",
        "health": "/api/health",
    }


if FRONTEND_DIST.is_dir():
    assets = FRONTEND_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}")
    def frontend(full_path: str):
        del full_path
        return FileResponse(FRONTEND_DIST / "index.html")
