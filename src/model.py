import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models

HAS_TORCH = True


if HAS_TORCH:

    class ResNeXtLSTMDetector(nn.Module):

        def __init__(
            self,
            backbone_type="resnet18",
            hidden_dim=128,
            num_layers=2,
            bidirectional=True,
            dropout=0.4,
            pretrained=False,
            use_temporal_attention=True
        ):
            super(ResNeXtLSTMDetector, self).__init__()

            self.backbone_type = backbone_type
            self.hidden_dim = hidden_dim
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1
            self.use_temporal_attention = use_temporal_attention

            if backbone_type == "resnet18":
                weights = (
                    models.ResNet18_Weights.DEFAULT
                    if pretrained else None
                )
                backbone = models.resnet18(weights=weights)
                in_features = backbone.fc.in_features
                setattr(backbone, "fc", nn.Identity())
                self.feature_extractor = backbone

            elif backbone_type == "resnext50":
                weights = (
                    models.ResNeXt50_32X4D_Weights.DEFAULT
                    if pretrained else None
                )
                backbone = models.resnext50_32x4d(weights=weights)
                in_features = backbone.fc.in_features
                setattr(backbone, "fc", nn.Identity())
                self.feature_extractor = backbone

            elif backbone_type == "resnet50":
                weights = (
                    models.ResNet50_Weights.DEFAULT
                    if pretrained else None
                )
                backbone = models.resnet50(weights=weights)
                in_features = backbone.fc.in_features
                setattr(backbone, "fc", nn.Identity())
                self.feature_extractor = backbone

            else:
                raise ValueError(
                    f"Unsupported backbone: {backbone_type}"
                )

            self.feature_proj = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5)
            )

            self.lstm = nn.LSTM(
                input_size=512,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0
            )

            lstm_out_dim = hidden_dim * self.num_directions

            self.frame_classifier = nn.Sequential(
                nn.Linear(lstm_out_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )

            self.classifier = nn.Sequential(
                nn.Linear(lstm_out_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )
            if use_temporal_attention:
                self.temporal_attention = nn.Sequential(
                    nn.Linear(lstm_out_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1)
                )

        def forward(self, x):
            batch_size, seq_len, c, h, w = x.shape

            x_reshaped = x.reshape(
                batch_size * seq_len,
                c,
                h,
                w
            )

            if any(
                parameter.requires_grad
                for parameter in self.feature_extractor.parameters()
            ):
                features = self.feature_extractor(x_reshaped)
            else:
                with torch.no_grad():
                    features = self.feature_extractor(x_reshaped)

            projected = self.feature_proj(features)

            seq_features = projected.reshape(
                batch_size,
                seq_len,
                512
            )

            lstm_out, _ = self.lstm(seq_features)

            frame_logits = self.frame_classifier(lstm_out)

            if self.use_temporal_attention:
                attention_scores = self.temporal_attention(lstm_out).squeeze(-1)
                attention_weights = torch.softmax(
                    attention_scores,
                    dim=1
                ).unsqueeze(-1)
                temporal_pooled = torch.sum(
                    lstm_out * attention_weights,
                    dim=1
                )
            else:
                temporal_pooled = torch.mean(lstm_out, dim=1)

            logits = self.classifier(temporal_pooled)

            return logits, frame_logits

        def freeze_backbone(self):
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        def unfreeze_backbone(self):
            for param in self.feature_extractor.parameters():
                param.requires_grad = True


class SpatialTemporalForensicDetector:

    def __init__(self):
        self.weights = {
            "boundary_weight": 0.35,
            "temporal_jitter_weight": 0.35,
            "color_inconsistency_weight": 0.30
        }

    def analyze_sequence(self, face_sequence_np):

        num_frames = len(face_sequence_np)
        frame_anomaly_scores = []

        for i in range(num_frames):

            frame = face_sequence_np[i]

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2GRAY
            )

            laplacian_var = cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()

            gx = cv2.Sobel(
                gray,
                cv2.CV_64F,
                1,
                0,
                ksize=3
            )

            gy = cv2.Sobel(
                gray,
                cv2.CV_64F,
                0,
                1,
                ksize=3
            )

            grad_mag = np.sqrt(
                gx ** 2 + gy ** 2
            )

            h, w = gray.shape

            border_mask = np.ones(
                (h, w),
                dtype=bool
            )

            border_mask[
                int(h * 0.15):int(h * 0.85),
                int(w * 0.15):int(w * 0.85)
            ] = False

            border_grad = np.mean(
                grad_mag[border_mask]
            )

            center_grad = (
                np.mean(
                    grad_mag[~border_mask]
                ) + 1e-5
            )

            boundary_discrepancy = (
                abs(border_grad - center_grad)
                / center_grad
            )

            frame_score = min(
                1.0,
                max(
                    0.0,
                    float(
                        boundary_discrepancy * 0.8
                        +
                        (
                            1.0 /
                            (
                                1.0 +
                                np.exp(
                                    -0.01 *
                                    (laplacian_var - 250)
                                )
                            )
                        ) * 0.2
                    )
                )
            )

            frame_anomaly_scores.append(
                frame_score
            )

        temporal_diffs = []

        for i in range(num_frames - 1):

            f1 = face_sequence_np[i].astype(
                np.float32
            )

            f2 = face_sequence_np[i + 1].astype(
                np.float32
            )

            diff = np.mean(
                np.abs(f1 - f2)
            )

            temporal_diffs.append(diff)

        temporal_jitter = (
            float(np.std(temporal_diffs))
            if temporal_diffs
            else 0.0
        )

        norm_temporal_score = min(
            1.0,
            max(
                0.0,
                temporal_jitter / 15.0
            )
        )

        final_frame_scores = []

        for score in frame_anomaly_scores:

            blended = (
                0.65 * score
                +
                0.35 * norm_temporal_score
            )

            calibrated = (
                1.0 /
                (
                    1.0 +
                    np.exp(
                        -5.0 *
                        (blended - 0.45)
                    )
                )
            )

            final_frame_scores.append(
                float(calibrated)
            )

        video_fake_prob = float(
            np.mean(final_frame_scores)
        )

        return (
            video_fake_prob,
            final_frame_scores
        )