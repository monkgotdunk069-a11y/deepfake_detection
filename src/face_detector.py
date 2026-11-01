"""
Face detector abstraction for DeepDetect.

Priority order:
  1. InsightFace SCRFD (fast, accurate, production detector)
  2. OpenCV Haar Cascade (opt-in development fallback only)

All detectors return bounding boxes in (x, y, w, h) format (pixel coords).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Optional, Protocol, runtime_checkable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0

    def as_xywh(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def area(self) -> int:
        return self.w * self.h


@runtime_checkable
class FaceDetectorBackend(Protocol):
    name: str

    def detect(self, frame_rgb: np.ndarray) -> list[FaceBox]:
        ...


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

class _HaarBackend:
    """OpenCV Haar Cascade face detector — always available."""

    def __init__(self) -> None:
        cascade_path = os.path.join(
            os.path.dirname(cv2.__file__ or ""),
            "data",
            "haarcascade_frontalface_default.xml",
        )
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError("Haar cascade XML not found.")
        self.name = "Haar Cascade (OpenCV)"

    def detect(self, frame_rgb: np.ndarray) -> list[FaceBox]:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        raw = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
        boxes: list[FaceBox] = []
        if len(raw) > 0:
            for x, y, w, h in raw:
                boxes.append(FaceBox(int(x), int(y), int(w), int(h), confidence=0.9))
        return sorted(boxes, key=lambda b: b.area(), reverse=True)


class _SCRFDBackend:
    """InsightFace SCRFD detector — requires `insightface` package."""

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_insight = os.path.join(project_root, ".insightface")
        root_dir = local_insight if os.path.isdir(local_insight) else os.path.expanduser("~/.insightface")

        self._app = FaceAnalysis(
            name="buffalo_sc",
            root=root_dir,
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=(640, 640))
        self.name = "SCRFD (InsightFace)"

    def detect(self, frame_rgb: np.ndarray) -> list[FaceBox]:
        # InsightFace expects BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        faces = self._app.get(frame_bgr)
        boxes: list[FaceBox] = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            conf = float(face.det_score) if hasattr(face, "det_score") else 1.0
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                boxes.append(FaceBox(x1, y1, w, h, confidence=conf))
        return sorted(boxes, key=lambda b: b.confidence, reverse=True)


class _RetinaFaceBackend:
    """InsightFace RetinaFace model — high-accuracy fallback to SCRFD."""

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=(640, 640))
        self.name = "RetinaFace (InsightFace)"

    def detect(self, frame_rgb: np.ndarray) -> list[FaceBox]:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        faces = self._app.get(frame_bgr)
        boxes: list[FaceBox] = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            conf = float(face.det_score) if hasattr(face, "det_score") else 1.0
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                boxes.append(FaceBox(x1, y1, w, h, confidence=conf))
        return sorted(boxes, key=lambda b: b.confidence, reverse=True)


# ---------------------------------------------------------------------------
# Singleton detector factory
# ---------------------------------------------------------------------------

_detector_instance: Optional[FaceDetectorBackend] = None
_detector_name: str = "unknown"


def _build_detector() -> FaceDetectorBackend:
    """Build the configured production detector.

    Haar is intentionally opt-in. Silently using it in production would make
    benchmark results incomparable with the upgraded pipeline.
    """
    try:
        backend = _SCRFDBackend()
        logger.info("Face detector: %s", backend.name)
        return backend
    except Exception as exc:
        if os.getenv("DEEPDETECT_ALLOW_HAAR", "").lower() not in {"1", "true", "yes"}:
            raise RuntimeError(
                "SCRFD is unavailable. Install insightface and onnxruntime, "
                "or set DEEPDETECT_ALLOW_HAAR=1 only for local development."
            ) from exc
        logger.warning("SCRFD unavailable (%s); using opt-in Haar fallback.", exc)
        return _HaarBackend()


def get_detector() -> FaceDetectorBackend:
    """Return the module-level singleton detector (lazy init)."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = _build_detector()
    return _detector_instance


def get_detector_name() -> str:
    det = get_detector()
    return getattr(det, "name", "Unknown")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_faces(frame_rgb: np.ndarray) -> list[FaceBox]:
    """Detect all faces in *frame_rgb* (H×W×3 uint8 RGB). Returns sorted by confidence."""
    det = get_detector()
    try:
        return det.detect(frame_rgb)
    except Exception as exc:
        logger.warning("Face detection error (%s): %s", get_detector_name(), exc)
        return []


def detect_and_crop_face(
    frame_rgb: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
    margin_ratio: float = 0.2,
    last_box: Optional[tuple[int, int, int, int]] = None,
) -> tuple[np.ndarray, Optional[tuple[int, int, int, int]]]:
    """
    Detect the primary face, add margin, and return a cropped + resized patch.

    Returns:
        (cropped_rgb np.ndarray, (x, y, w, h) or None)
    """
    h_img, w_img = frame_rgb.shape[:2]
    boxes = detect_faces(frame_rgb)

    if boxes:
        if last_box is not None:
            lx, ly, lw, lh = last_box
            lcx, lcy = lx + lw / 2, ly + lh / 2
            best_idx = 0
            best_dist = float("inf")
            for b_i, b in enumerate(boxes):
                bcx, bcy = b.x + b.w / 2, b.y + b.h / 2
                dist = (bcx - lcx) ** 2 + (bcy - lcy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_idx = b_i
            fb = boxes[best_idx]
        else:
            fb = boxes[0]
        x, y, w, h = fb.x, fb.y, fb.w, fb.h
    elif last_box is not None:
        x, y, w, h = last_box
    else:
        # Centre crop
        dim = int(min(h_img, w_img) * 0.7)
        x = (w_img - dim) // 2
        y = (h_img - dim) // 2
        w = h = dim

    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w_img, x + w + mx)
    y2 = min(h_img, y + h + my)

    cropped = frame_rgb[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = frame_rgb

    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
    return resized, (x, y, w, h)
