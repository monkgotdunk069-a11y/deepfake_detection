"""
Live Deepfake Detection Engine.

Provides real-time webcam frame processing using:
- InsightFace SCRFD for accurate real-time face detection and cropping.
- MobileNet-V3 + XGBoost ensemble model for fast, accurate 32-frame temporal sequence analysis.
- Non-blocking inference pipeline for smooth, low-latency live streaming.
- Session management for streaming WebSocket clients with continuous deepfake risk scoring.
"""

from __future__ import annotations

import base64
import collections
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import time
from typing import Any, Optional

import cv2
import numpy as np
import torch

from src.face_detector import detect_faces, get_detector, get_detector_name
from src.models_zoo import MobileXGBDetector

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# Cache for loaded MobileNet+XGBoost live model
_live_model: Optional[MobileXGBDetector] = None


def get_live_model() -> MobileXGBDetector:
    """Load or return the singleton MobileNet-V3 + XGBoost model for live detection."""
    global _live_model
    if _live_model is not None:
        return _live_model

    # Look for MobileNet-XGBoost model checkpoint
    candidates = [
        ROOT / "artifacts_full" / "mobilenet_xgboost.joblib",
        ROOT / "artifacts_smoke" / "mobilenet_xgboost.joblib",
    ]
    model_path = None
    for cand in candidates:
        if cand.is_file():
            model_path = str(cand)
            break

    if not model_path:
        raise FileNotFoundError(
            f"MobileNet-XGBoost model checkpoint not found in: {[str(c) for c in candidates]}"
        )

    logger.info("Loading Live MobileNet-V3 + XGBoost model from %s", model_path)
    model = MobileXGBDetector.load(model_path, pretrained=True)
    _live_model = model
    return _live_model


def _crop_face_patch(
    frame_rgb: np.ndarray,
    box: Any,
    margin_ratio: float = 0.2,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Crop face box with surrounding margin and resize to target size."""
    h_img, w_img = frame_rgb.shape[:2]
    x, y, w, h = int(box.x), int(box.y), int(box.w), int(box.h)

    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w_img, x + w + mx)
    y2 = min(h_img, y + h + my)

    cropped = frame_rgb[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = frame_rgb

    return cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)


def _to_thumbnail_data_url(image_rgb: np.ndarray, size: tuple[int, int] = (96, 96)) -> str:
    """Encode an RGB image to a small base64 JPEG data URL for frontend live preview."""
    resized = cv2.resize(image_rgb, size, interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# Public aliases
crop_face_patch = _crop_face_patch
to_thumbnail_data_url = _to_thumbnail_data_url


class LiveDetectionSession:
    """
    Stateful session for a single live webcam client.

    Maintains a rolling buffer of 32 temporal face frames and runs
    continuous real-time MobileNet-V3 + XGBoost inference.
    """

    def __init__(self, target_frames: int = 32) -> None:
        self.target_frames = target_frames
        self.sequence_buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=target_frames
        )
        self.recent_crops: collections.deque[str] = collections.deque(maxlen=6)

        # Telemetry & performance tracking
        self.frame_count = 0
        self.face_count = 0
        self.start_time = time.perf_counter()
        self.last_frame_time = self.start_time
        self.rolling_fps = 0.0

        self.last_detect_ms = 0.0
        self.last_inference_ms = 0.0
        self.last_box: Optional[tuple[int, int, int, int]] = None
        self._smoothed_fake_score: Optional[float] = None

        # Background inference executor to avoid stalling the video feed
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live_mobilenet_xgb")
        self._infer_future: Optional[Any] = None

        # Last confirmed prediction state
        self.last_prediction: dict[str, Any] = {
            "fake_score": 0.0,
            "real_score": 1.0,
            "confidence": 0.0,
            "verdict": "ANALYZING",
            "risk_level": "WAITING",
            "risk_label": "Analyzing Biometric Stream...",
            "frame_scores": [],
            "has_evaluated": False,
        }

    def reset(self) -> None:
        """Clear session buffer and statistics."""
        self.sequence_buffer.clear()
        self.recent_crops.clear()
        self.frame_count = 0
        self.face_count = 0
        self.start_time = time.perf_counter()
        self.last_frame_time = self.start_time
        self.rolling_fps = 0.0
        self.last_detect_ms = 0.0
        self.last_inference_ms = 0.0
        self.last_box = None
        self._smoothed_fake_score = None
        self.last_prediction = {
            "fake_score": 0.0,
            "real_score": 1.0,
            "confidence": 0.0,
            "verdict": "ANALYZING",
            "risk_level": "WAITING",
            "risk_label": "Analyzing Biometric Stream...",
            "frame_scores": [],
            "has_evaluated": False,
        }

    def close(self) -> None:
        """Shut down background executor."""
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def decode_image(self, payload: str | bytes) -> Optional[np.ndarray]:
        """Decode incoming base64 or raw bytes into RGB numpy array."""
        try:
            if isinstance(payload, str):
                if "," in payload:
                    payload = payload.split(",", 1)[1]
                raw_bytes = base64.b64decode(payload)
            else:
                raw_bytes = payload

            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                return None
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            logger.warning("Failed to decode webcam frame: %s", exc)
            return None

    def _run_model_inference_sync(self, frames_list: list[np.ndarray]) -> dict[str, Any]:
        """Execute MobileNet-V3 + XGBoost inference on exactly 32 frames."""
        model = get_live_model()

        t0 = time.perf_counter()
        seq_np = np.stack(frames_list, axis=0)  # (32, 224, 224, 3) uint8

        fake_prob, raw_frame_scores = model.analyze_sequence(seq_np, is_live=True)

        # Exponential Moving Average for smooth, flicker-free live telemetry
        if self._smoothed_fake_score is None:
            self._smoothed_fake_score = fake_prob
        else:
            self._smoothed_fake_score = 0.75 * self._smoothed_fake_score + 0.25 * fake_prob

        fake_prob = float(np.clip(self._smoothed_fake_score, 0.0, 1.0))
        real_prob = float(1.0 - fake_prob)
        frame_scores = [float(np.clip(p, 0.0, 1.0)) for p in raw_frame_scores]

        t1 = time.perf_counter()
        inf_ms = float((t1 - t0) * 1000.0)

        # Calibration & thresholds
        if fake_prob >= 0.65:
            verdict = "FAKE"
            risk_level = "HIGH"
            risk_label = "High Deepfake Risk"
            confidence = float(fake_prob * 100.0)
        elif fake_prob >= 0.40:
            verdict = "SUSPICIOUS"
            risk_level = "SUSPICIOUS"
            risk_label = "Suspicious / Inconclusive"
            confidence = float(fake_prob * 100.0)
        else:
            verdict = "REAL"
            risk_level = "LOW"
            risk_label = "Authentic / Low Risk"
            confidence = float(real_prob * 100.0)

        return {
            "fake_score": fake_prob,
            "real_score": real_prob,
            "confidence": confidence,
            "verdict": verdict,
            "risk_level": risk_level,
            "risk_label": risk_label,
            "frame_scores": frame_scores,
            "inference_ms": inf_ms,
            "has_evaluated": True,
        }

    def process_frame(self, frame_rgb: np.ndarray) -> dict[str, Any]:
        """
        Process a single RGB webcam frame:
        1. Run InsightFace SCRFD face detector.
        2. If face found: crop, push to 32-frame buffer, and trigger MobileNet+XGBoost sequence model.
        3. Returns real-time metrics, face bounding boxes, and continuously updated deepfake risk scores.
        """
        now = time.perf_counter()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        if dt > 0:
            inst_fps = 1.0 / dt
            self.rolling_fps = (
                inst_fps if self.rolling_fps == 0 else 0.85 * self.rolling_fps + 0.15 * inst_fps
            )
        self.frame_count += 1

        h_img, w_img = frame_rgb.shape[:2]

        # 1. SCRFD face detection
        t_det0 = time.perf_counter()
        boxes = detect_faces(frame_rgb)
        self.last_detect_ms = float((time.perf_counter() - t_det0) * 1000.0)

        all_boxes_info = [
            {
                "x": int(b.x),
                "y": int(b.y),
                "w": int(b.w),
                "h": int(b.h),
                "confidence": round(float(b.confidence), 3),
            }
            for b in boxes
        ]

        # Check completed background inference if ready
        if self._infer_future is not None and self._infer_future.done():
            try:
                pred = self._infer_future.result()
                self.last_inference_ms = float(pred["inference_ms"])
                self.last_prediction = pred
            except Exception as exc:
                logger.error("Background sequence inference error: %s", exc)
            self._infer_future = None

        if not boxes:
            self.last_box = None
            return {
                "type": "result",
                "face_detected": False,
                "box": None,
                "all_faces": [],
                "fake_score": float(self.last_prediction["fake_score"]),
                "real_score": float(self.last_prediction["real_score"]),
                "confidence": float(self.last_prediction["confidence"]),
                "verdict": "NO_FACE",
                "risk_level": str(self.last_prediction["risk_level"]),
                "risk_label": "No Face Detected",
                "buffer_count": int(len(self.sequence_buffer)),
                "buffer_target": int(self.target_frames),
                "is_buffer_full": bool(len(self.sequence_buffer) >= self.target_frames),
                "frame_scores": [float(s) for s in self.last_prediction.get("frame_scores", [])],
                "face_crop_data_url": None,
                "recent_crops": list(self.recent_crops),
                "detect_ms": round(float(self.last_detect_ms), 1),
                "inference_ms": round(float(self.last_inference_ms), 1),
                "fps": round(float(self.rolling_fps), 1),
                "model_name": "MobileNetV3 + XGBoost",
                "detector_name": get_detector_name(),
                "has_evaluated": bool(self.last_prediction.get("has_evaluated", False)),
                "frame_width": int(w_img),
                "frame_height": int(h_img),
                "timestamp": float(now),
            }

        # Face detected: track primary face box using nearest-neighbor continuity
        self.face_count += 1
        if self.last_box is not None and len(boxes) > 1:
            lx, ly, lw, lh = self.last_box
            lcx, lcy = lx + lw / 2, ly + lh / 2
            best_box = boxes[0]
            best_dist = float("inf")
            for b in boxes:
                bcx, bcy = b.x + b.w / 2, b.y + b.h / 2
                dist = (bcx - lcx) ** 2 + (bcy - lcy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_box = b
            primary_box = best_box
        else:
            primary_box = boxes[0]

        self.last_box = (primary_box.x, primary_box.y, primary_box.w, primary_box.h)
        face_crop = _crop_face_patch(frame_rgb, primary_box, margin_ratio=0.25)
        self.sequence_buffer.append(face_crop)

        crop_data_url = _to_thumbnail_data_url(face_crop)
        if crop_data_url:
            self.recent_crops.append(crop_data_url)

        curr_buf_len = len(self.sequence_buffer)

        # Trigger background MobileNet-V3 + XGBoost 32-frame sequence inference if not already running
        if self._infer_future is None:
            if curr_buf_len < self.target_frames:
                # Early warmup: uniformly sample available frames to 32
                indices = np.linspace(0, curr_buf_len - 1, self.target_frames, dtype=int)
                seq_to_run = [self.sequence_buffer[i].copy() for i in indices]
            else:
                seq_to_run = [f.copy() for f in self.sequence_buffer]

            self._infer_future = self._executor.submit(
                self._run_model_inference_sync, seq_to_run
            )

        return {
            "type": "result",
            "face_detected": True,
            "box": {
                "x": int(primary_box.x),
                "y": int(primary_box.y),
                "w": int(primary_box.w),
                "h": int(primary_box.h),
                "confidence": round(float(primary_box.confidence), 3),
            },
            "all_faces": all_boxes_info,
            "fake_score": float(self.last_prediction["fake_score"]),
            "real_score": float(self.last_prediction["real_score"]),
            "confidence": float(self.last_prediction["confidence"]),
            "verdict": str(self.last_prediction["verdict"]),
            "risk_level": str(self.last_prediction["risk_level"]),
            "risk_label": str(self.last_prediction["risk_label"]),
            "buffer_count": int(curr_buf_len),
            "buffer_target": int(self.target_frames),
            "is_buffer_full": bool(curr_buf_len >= self.target_frames),
            "frame_scores": [float(s) for s in self.last_prediction.get("frame_scores", [])],
            "face_crop_data_url": crop_data_url,
            "recent_crops": list(self.recent_crops),
            "detect_ms": round(float(self.last_detect_ms), 1),
            "inference_ms": round(float(self.last_inference_ms), 1),
            "fps": round(float(self.rolling_fps), 1),
            "model_name": "MobileNetV3 + XGBoost",
            "detector_name": get_detector_name(),
            "has_evaluated": bool(self.last_prediction.get("has_evaluated", False)),
            "frame_width": int(w_img),
            "frame_height": int(h_img),
            "timestamp": float(now),
        }
