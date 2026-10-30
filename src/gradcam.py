"""
Grad-CAM implementation for the ResNeXtLSTMDetector and suspicious-frame detection.

Produces:
  - Per-frame Grad-CAM saliency overlays (PIL Images)
  - Suspicious frame timestamps with neighbour-comparison scores
  - A consolidated SuspiciousFrameReport per video analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

logger = logging.getLogger(__name__)

HAS_TORCH = True
try:
    import torch  # noqa: F401 (already imported above, guard for type-check)
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Grad-CAM core
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Hooks onto the last spatial conv layer of a ResNet backbone and computes
    class-discriminative activation maps.

    Usage::

        gcam = GradCAM(model)
        overlay = gcam.generate(frame_tensor, class_idx=1)   # 1 = fake
        gcam.remove_hooks()
    """

    def __init__(self, model) -> None:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for Grad-CAM.")
        self._model = model
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._hooks: list = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _target_layer(self):
        """Return the last conv layer of the ResNet backbone."""
        bb = self._model.feature_extractor
        # All standard ResNet variants end with layer4
        return bb.layer4[-1].conv2  # type: ignore[attr-defined]

    def _register_hooks(self) -> None:
        layer = self._target_layer()

        def _save_activation(_, __, output):
            self._activations = output.detach()

        def _save_gradient(_, grad_input, grad_output):  # noqa: ARG001
            self._gradients = grad_output[0].detach()

        self._hooks.append(layer.register_forward_hook(_save_activation))
        self._hooks.append(layer.register_backward_hook(_save_gradient))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        frame_tensor: torch.Tensor,
        class_idx: int = 1,
        original_size: tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for one frame.

        Args:
            frame_tensor: Shape (1, C, H, W) — already normalised.
            class_idx: 1 = fake class, 0 = real.
            original_size: (width, height) of the target overlay.

        Returns:
            float32 heatmap in [0, 1], shape (H, W).
        """
        self._model.zero_grad()
        frame_tensor = frame_tensor.requires_grad_(True)

        # Forward through backbone only
        features = self._model.feature_extractor(frame_tensor)  # (1, 512) after global pool
        # We need spatial features — re-hook before global pool
        # Actually we need activations from layer4 which we hooked.
        # Run a minimal forward to get activations:
        activations = self._activations
        if activations is None:
            return np.zeros((original_size[1], original_size[0]), dtype=np.float32)

        # Use the projection + classifier head as the scoring function
        projected = self._model.feature_proj(features)
        # Fake a single-frame sequence for the LSTM
        seq = projected.unsqueeze(1)  # (1, 1, 512)
        lstm_out, _ = self._model.lstm(seq)
        score = self._model.classifier(lstm_out[:, 0, :])  # (1, 1)

        target = score[0, 0] if class_idx == 1 else -score[0, 0]
        target.backward()

        if self._gradients is None or self._activations is None:
            return np.zeros((original_size[1], original_size[0]), dtype=np.float32)

        # Pool gradients over spatial dimensions
        pooled_grads = self._gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        weighted = (pooled_grads * self._activations).sum(dim=1, keepdim=True)  # (1,1,H,W)
        cam = F.relu(weighted).squeeze().cpu().numpy()

        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize to original frame size
        cam_resized = cv2.resize(cam, original_size, interpolation=cv2.INTER_LINEAR)
        return cam_resized.astype(np.float32)


def overlay_gradcam(
    face_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """
    Blends a Grad-CAM heatmap over the face image.

    Args:
        face_rgb: (H, W, 3) uint8 RGB face crop.
        heatmap: (H, W) float32 in [0, 1].
        alpha: Overlay opacity.
        colormap: OpenCV colormap constant.

    Returns:
        PIL Image (RGB) with the heatmap overlay.
    """
    h, w = face_rgb.shape[:2]
    cam_uint8 = np.uint8(255 * heatmap)
    cam_uint8 = cv2.resize(cam_uint8, (w, h))
    cam_colored = cv2.applyColorMap(cam_uint8, colormap)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(face_rgb, 1 - alpha, cam_colored, alpha, 0)
    return Image.fromarray(blended)


# ---------------------------------------------------------------------------
# Suspicious frame detection
# ---------------------------------------------------------------------------

SUSPICIOUS_THRESHOLD = 0.60   # frame score above this is suspicious
NEIGHBOUR_MSE_SPIKE = 400.0   # inter-frame MSE spike to flag discontinuity


@dataclass
class SuspiciousFrame:
    frame_index: int
    timestamp_sec: float
    fake_score: float
    neighbour_mse: float          # MSE vs. previous frame
    neighbour_delta: float        # score jump vs. previous frame
    gradcam_overlay: Optional[Image.Image] = None
    reason: str = ""


@dataclass
class SuspiciousFrameReport:
    total_frames: int
    fps: float
    suspicious_frames: list[SuspiciousFrame] = field(default_factory=list)
    detector_name: str = "InsightFace SCRFD"

    # ------------------------------------------------------------------ #
    # Serialisable summary for API / frontend
    # ------------------------------------------------------------------ #
    def to_dict(self, include_images: bool = False) -> dict:
        frames_out = []
        for sf in self.suspicious_frames:
            entry: dict = {
                "frame_index": sf.frame_index,
                "timestamp_sec": round(sf.timestamp_sec, 3),
                "fake_score": round(sf.fake_score, 4),
                "neighbour_mse": round(sf.neighbour_mse, 2),
                "neighbour_delta": round(sf.neighbour_delta, 4),
                "reason": sf.reason,
            }
            if include_images and sf.gradcam_overlay is not None:
                import base64
                import io as _io
                buf = _io.BytesIO()
                sf.gradcam_overlay.save(buf, format="JPEG", quality=80)
                entry["gradcam_data_url"] = (
                    "data:image/jpeg;base64,"
                    + base64.b64encode(buf.getvalue()).decode()
                )
            frames_out.append(entry)
        return {
            "total_frames": self.total_frames,
            "fps": self.fps,
            "suspicious_count": len(self.suspicious_frames),
            "detector_name": self.detector_name,
            "suspicious_frames": frames_out,
        }


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

def compute_suspicious_frames(
    face_pil_list: list[Image.Image],
    frame_scores: list[float],
    fps: float,
    model=None,
    device=None,
    transform=None,
    suspicious_threshold: float = SUSPICIOUS_THRESHOLD,
) -> SuspiciousFrameReport:
    """
    Identify suspicious frames from a completed analysis and optionally compute
    Grad-CAM overlays for each flagged frame.

    Args:
        face_pil_list:  List of extracted face PIL images.
        frame_scores:   Per-frame fake probability list (same length as face_pil_list).
        fps:            Video frame-rate (for timestamp calculation).
        model:          (Optional) ResNeXtLSTMDetector instance for Grad-CAM.
        device:         Torch device.
        transform:      Inference transform pipeline (torchvision Compose).
        suspicious_threshold: Score threshold to classify a frame as suspicious.

    Returns:
        SuspiciousFrameReport
    """
    n = len(face_pil_list)
    report = SuspiciousFrameReport(total_frames=n, fps=fps)

    if n == 0:
        return report

    # --- pre-compute inter-frame MSE ---
    grays: list[np.ndarray] = [
        np.array(img.convert("L"), dtype=np.float32) for img in face_pil_list
    ]
    mse_values: list[float] = [0.0]  # frame 0 has no predecessor
    for i in range(1, n):
        diff = grays[i] - grays[i - 1]
        mse_values.append(float(np.mean(diff ** 2)))

    # --- Grad-CAM setup ---
    gcam: Optional[GradCAM] = None
    if HAS_TORCH and model is not None and isinstance(model, torch.nn.Module):
        try:
            gcam = GradCAM(model)
        except Exception as exc:
            logger.warning("Grad-CAM init failed: %s", exc)

    # --- Flag frames ---
    for i, score in enumerate(frame_scores):
        prev_score = frame_scores[i - 1] if i > 0 else score
        delta = abs(score - prev_score)
        mse = mse_values[i]

        reasons: list[str] = []
        if score >= suspicious_threshold:
            reasons.append(f"score {score:.2%} ≥ threshold {suspicious_threshold:.2%}")
        if mse >= NEIGHBOUR_MSE_SPIKE and i > 0:
            reasons.append(f"inter-frame MSE spike ({mse:.0f})")
        if delta >= 0.25 and i > 0:
            reasons.append(f"score jump Δ{delta:.2%} vs. previous frame")

        if not reasons:
            continue

        # Optional Grad-CAM overlay
        overlay: Optional[Image.Image] = None
        if gcam is not None and transform is not None:
            try:
                face_np = np.array(face_pil_list[i])
                tensor = transform(face_np).unsqueeze(0).to(device)
                heatmap = gcam.generate(tensor, class_idx=1)
                overlay = overlay_gradcam(face_np, heatmap)
            except Exception as exc:
                logger.debug("Grad-CAM frame %d failed: %s", i, exc)

        sf = SuspiciousFrame(
            frame_index=i,
            timestamp_sec=i / fps if fps > 0 else 0.0,
            fake_score=score,
            neighbour_mse=mse,
            neighbour_delta=delta,
            gradcam_overlay=overlay,
            reason="; ".join(reasons),
        )
        report.suspicious_frames.append(sf)

    if gcam is not None:
        gcam.remove_hooks()

    return report


def batch_gradcam_overlays(
    face_pil_list: list[Image.Image],
    frame_scores: list[float],
    model,
    device,
    transform,
    max_frames: int = 8,
) -> list[Image.Image]:
    """
    Generate Grad-CAM overlays for up to *max_frames* highest-scoring frames.
    Returns a list of PIL Images (RGB).
    """
    if not HAS_TORCH or model is None:
        return []

    # Pick top-scoring frame indices
    indexed = sorted(enumerate(frame_scores), key=lambda t: t[1], reverse=True)
    top_indices = [idx for idx, _ in indexed[:max_frames]]

    gcam = GradCAM(model)
    overlays: list[Image.Image] = []
    for idx in sorted(top_indices):
        face_np = np.array(face_pil_list[idx])
        try:
            tensor = transform(face_np).unsqueeze(0).to(device)
            heatmap = gcam.generate(tensor, class_idx=1)
            overlays.append(overlay_gradcam(face_np, heatmap))
        except Exception as exc:
            logger.debug("Grad-CAM overlay failed at frame %d: %s", idx, exc)
    gcam.remove_hooks()
    return overlays
