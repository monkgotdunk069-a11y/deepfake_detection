"""
Alternative model zoo for DeepDetect.

Models:
  - EfficientNet-V2-S + BiLSTM  (EfficientDetector)
  - MobileNet-V3-Large + XGBoost ensemble  (MobileXGBDetector)

Both expose the same interface as ResNeXtLSTMDetector so they can be loaded
via load_detector_model(backbone_type=...).
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from xgboost import XGBClassifier

from src.forensics import extract_forensic_descriptors

logger = logging.getLogger(__name__)

SUPPORTED_MODEL_NAMES = ("resnet18_bilstm", "efficientnet_v2_bilstm", "mobilenet_xgboost")


# ---------------------------------------------------------------------------
# EfficientNet-V2 + BiLSTM
# ---------------------------------------------------------------------------

class EfficientV2LSTMDetector(nn.Module):
    """
    EfficientNet-V2-Small backbone → 512-dim projection → BiLSTM → classifier.
    Drop-in replacement for ResNeXtLSTMDetector in inference.py.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        weights = tvm.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        backbone = tvm.efficientnet_v2_s(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.feature_extractor = backbone

        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out = hidden_dim * self.num_directions
        self.frame_classifier = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        self.temporal_attention = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        feats = self.feature_extractor(x.view(B * T, C, H, W))
        proj = self.feature_proj(feats).view(B, T, 512)
        lstm_out, _ = self.lstm(proj)
        frame_logits = self.frame_classifier(lstm_out)

        attn = torch.softmax(
            self.temporal_attention(lstm_out).squeeze(-1), dim=1
        ).unsqueeze(-1)
        pooled = (lstm_out * attn).sum(dim=1)
        logits = self.classifier(pooled)
        return logits, frame_logits

    def freeze_backbone(self) -> None:
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.feature_extractor.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# MobileNet-V3 feature extractor + XGBoost head
# ---------------------------------------------------------------------------

class MobileXGBDetector:
    """
    MobileNetV3-Large extracts frame-level 960-dim features.
    XGBoost classifies the aggregated feature vector (mean + std pooling).

    This is a *non-PyTorch* model that fits the same predict_video interface
    when used via the SpatialTemporalForensicDetector path.
    """

    backbone: nn.Module
    _feat_dim: int
    xgb: XGBClassifier
    _xgb_fitted: bool

    def __init__(
        self,
        xgb_params: Optional[dict] = None,
        pretrained: bool = True,
    ) -> None:
        weights = tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = tvm.mobilenet_v3_large(weights=weights)
        # Remove the classification head — keep features
        backbone.classifier = nn.Identity()
        self.backbone = backbone.eval()
        self._feat_dim = 960

        default_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "tree_method": "hist",
        }
        if xgb_params:
            default_params.update(xgb_params)
        self.xgb = XGBClassifier(**default_params)
        self._xgb_fitted = False

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(
        self, face_sequence_np: np.ndarray, include_forensic_descriptors: bool = False
    ) -> np.ndarray:
        """
        Extract MobileNet features per frame and aggregate (mean, std).
        Optionally appends multi-domain forensic descriptors (FFT, color boundary, Laplacian, ELA, jitter).

        Args:
            face_sequence_np: (N, H, W, 3) or (H, W, 3) uint8
            include_forensic_descriptors: Whether to append the 33-dim forensic descriptor vector.

        Returns:
            (2 * feat_dim,) or (2 * feat_dim + 33,) float32 vector.
        """
        if face_sequence_np.ndim == 3:
            face_sequence_np = np.expand_dims(face_sequence_np, axis=0)

        if face_sequence_np.shape[1:3] == (224, 224):
            # Fast vectorized normalization for (N, 224, 224, 3) uint8
            tensors = torch.from_numpy(face_sequence_np).permute(0, 3, 1, 2).float().div_(255.0)
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
            tensors = (tensors - mean) / std
        else:
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensors = torch.stack([transform(f) for f in face_sequence_np])  # (N, 3, 224, 224)
        with torch.no_grad():
            feats = self.backbone(tensors).cpu().numpy()  # (N, 960)
        spatial_vec = np.concatenate([feats.mean(axis=0), feats.std(axis=0)], axis=0).astype(np.float32)
        if include_forensic_descriptors:
            forensic_vec = extract_forensic_descriptors(face_sequence_np)
            return np.concatenate([spatial_vec, forensic_vec], axis=0).astype(np.float32)
        return spatial_vec

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        video_feature_matrix: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        Train XGBoost on pre-extracted feature matrix.

        Args:
            video_feature_matrix: (num_videos, 2*feat_dim) float32
            labels: (num_videos,) int — 0=real, 1=fake
        """
        self.xgb.fit(video_feature_matrix, labels)
        self._xgb_fitted = True

    # ------------------------------------------------------------------
    # Inference (same interface as SpatialTemporalForensicDetector)
    # ------------------------------------------------------------------

    def analyze_sequence(
        self, face_sequence_np: np.ndarray, is_live: bool = False
    ) -> tuple[float, list[float]]:
        """Return (video_fake_prob, [frame_fake_prob, ...])."""
        if face_sequence_np.ndim == 3:
            face_sequence_np = np.expand_dims(face_sequence_np, axis=0)

        if not self._xgb_fitted:
            logger.warning(
                "MobileXGBDetector XGBoost not fitted; returning fallback 0.5."
            )
            n = len(face_sequence_np)
            return 0.5, [0.5] * n

        if face_sequence_np.shape[1:3] == (224, 224):
            # Fast vectorized normalization for (N, 224, 224, 3) uint8
            tensors = torch.from_numpy(face_sequence_np).permute(0, 3, 1, 2).float().div_(255.0)
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
            tensors = (tensors - mean) / std
        else:
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensors = torch.stack([transform(f) for f in face_sequence_np])  # (N, 3, 224, 224)
        with torch.no_grad():
            feats = self.backbone(tensors).cpu().numpy()  # (N, 960)

        agg_feat = np.concatenate([feats.mean(axis=0), feats.std(axis=0)], axis=0).astype(np.float32)
        expected_features = getattr(self.xgb, "n_features_in_", 1920)
        forensic_vec = None
        if expected_features == len(agg_feat) + 33:
            forensic_vec = extract_forensic_descriptors(face_sequence_np)
            agg_feat = np.concatenate([agg_feat, forensic_vec], axis=0).astype(np.float32)

        proba = float(self.xgb.predict_proba(agg_feat.reshape(1, -1))[0, 1])

        # Multi-Threat Physical Forensic & Generative AI Fusion Shield
        # Evaluates physical sensor cues: ELA compression inconsistencies, boundary softening,
        # Fourier frequency spikes, and latent pixel morphing to detect modern generative models (Kling, Luma, Runway, SadTalker, etc.)
        # In live webcam mode (is_live=True), calibrated to avoid false alarms from natural 3D head movement, angles, and camera auto-exposure.
        if forensic_vec is not None and len(face_sequence_np) > 1:
            fft_0 = float(forensic_vec[0])
            fft_1 = float(forensic_vec[1])
            delta_cb = float(forensic_vec[12])
            delta_cr = float(forensic_vec[13])
            lap_ratio = float(forensic_vec[20])
            ela_max = float(forensic_vec[26])
            temp_mean = float(forensic_vec[30])
            temp_max = float(forensic_vec[32])

            synthetic_signals: list[tuple[str, float]] = []

            if not is_live:
                # 1. Fourier High-Frequency Energy Spike (GAN/Diffusion upsampling grid artifacts)
                # Authentic optical camera recordings have fft_0 < 8.55, fft_1 < 6.65
                if fft_0 > 8.60 and fft_1 > 6.65:
                    fft_score = float(min(0.95, 0.65 + (fft_0 - 8.60) / 0.35 * 0.30))
                    synthetic_signals.append(("FFT_Spike", fft_score))

                # 2. Extreme Latent pixel morphing & non-rigid warping (Image-to-Video diffusion)
                if temp_max > 5000.0 and temp_mean > 2500.0:
                    j_score = float(min(0.95, 0.65 + (temp_max - 5000.0) / 4000.0 * 0.30))
                    synthetic_signals.append(("Latent_Morphing", j_score))

                # 3. Boundary Chrominance / Edge Disparity (Face-swap paste seam)
                if (delta_cb > 16.0 or delta_cr > 16.0) and lap_ratio > 3.5:
                    bdry_score = float(min(0.94, 0.60 + (max(delta_cb, delta_cr) - 16.0) / 10.0 * 0.30))
                    synthetic_signals.append(("Boundary_Softening", bdry_score))

                # 4. Error Level Analysis (ELA) Compression Disparity
                if ela_max > 60.0:
                    ela_score = float(min(0.95, 0.60 + (ela_max - 60.0) / 40.0 * 0.35))
                    synthetic_signals.append(("ELA_Disparity", ela_score))

                if len(synthetic_signals) >= 2:
                    anomaly_prob = float(np.mean([s[1] for s in synthetic_signals]))
                    proba = max(proba, anomaly_prob)
                elif len(synthetic_signals) == 1 and synthetic_signals[0][0] == "FFT_Spike" and synthetic_signals[0][1] >= 0.70:
                    proba = max(proba, synthetic_signals[0][1])
            else:
                # In live webcam mode: only trigger if extreme virtual camera / screen replay spoofing is detected
                if fft_0 > 9.15 and fft_1 > 7.40 and ela_max > 50.0:
                    proba = max(proba, 0.85)

        # Dynamic per-frame scores modulated by feature variance
        n = len(face_sequence_np)
        if n > 1:
            mean_feat = feats.mean(axis=0, keepdims=True)
            dists = np.linalg.norm(feats - mean_feat, axis=1)
            dist_std = dists.std()
            if dist_std > 1e-6:
                norm_dists = (dists - dists.mean()) / dist_std
                frame_scores = [float(np.clip(proba + 0.08 * d, 0.0, 1.0)) for d in norm_dists]
            else:
                frame_scores = [float(proba)] * n
        else:
            frame_scores = [float(proba)] * n

        return float(proba), frame_scores

    def save(self, path: str) -> None:
        joblib.dump({"xgb": self.xgb, "fitted": self._xgb_fitted}, path)

    @classmethod
    def load(cls, path: str, pretrained: bool = True) -> MobileXGBDetector:
        data = joblib.load(path)
        obj = cls(pretrained=pretrained)
        obj.xgb = data["xgb"]
        obj._xgb_fitted = data["fitted"]
        return obj


def build_sequence_model(name: str, pretrained: bool = False) -> Union[nn.Module, MobileXGBDetector]:
    """Construct one of the models used by the notebook and API."""
    normalized = name.lower().replace("-", "_")
    if normalized in {"resnet18", "resnet18_bilstm"}:
        from src.model import ResNeXtLSTMDetector
        return ResNeXtLSTMDetector(backbone_type="resnet18", pretrained=pretrained)
    if normalized in {"efficientnet_v2", "efficientnet_v2_bilstm", "efficientnet"}:
        return EfficientV2LSTMDetector(pretrained=pretrained)
    if normalized in {"mobilenet_xgboost", "mobilenet_xgb", "mobilenet"}:
        return MobileXGBDetector(pretrained=pretrained)
    raise ValueError(
        f"Unsupported model '{name}'. Choose one of: {', '.join(SUPPORTED_MODEL_NAMES)}."
    )


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_model_comparison(
    model_results: dict[str, dict],
) -> dict:
    """
    Compare multiple model results on the same video.

    Args:
        model_results: {model_name: {"fake_probability": float, "frame_scores": [...]}}

    Returns:
        Comparison dict with ensemble vote, agreement score, individual outputs.
    """
    probs = {k: v["fake_probability"] for k, v in model_results.items()}
    ensemble_prob = float(np.mean(list(probs.values())))
    agreement = 1.0 - float(np.std(list(probs.values())))

    votes = sum(1 for p in probs.values() if p >= 0.5)
    majority_fake = votes > len(probs) / 2

    return {
        "ensemble_fake_probability": ensemble_prob,
        "ensemble_is_fake": majority_fake,
        "model_agreement": max(0.0, min(1.0, agreement)),
        "vote_count_fake": votes,
        "vote_count_real": len(probs) - votes,
        "individual": probs,
    }
