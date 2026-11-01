import os
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import cast

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import HAS_TORCH, ResNeXtLSTMDetector, SpatialTemporalForensicDetector
from src.models_zoo import EfficientV2LSTMDetector, MobileXGBDetector
from src.preprocessing import process_video, process_video_multi_subject


if HAS_TORCH:
    transform_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "deepfake_model.pt"
)


def load_detector_model(
    model_path=MODEL_PATH,
    backbone_type="resnet18",
    device=None
):

    if not HAS_TORCH:
        forensic_engine = SpatialTemporalForensicDetector()
        return forensic_engine, "CPU"

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    print(f"Using device: {device}")
    print(f"Loading model: {model_path}")
    print(f"Backbone: {backbone_type}")

    if backbone_type in {"mobilenet_xgb", "mobilenet_xgboost", "mobilenet"}:
        return MobileXGBDetector.load(model_path, pretrained=True), device

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:
        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    state_dict = cleaned_state_dict

    lstm_weight = state_dict.get("lstm.weight_hh_l0")
    if lstm_weight is None or len(lstm_weight.shape) != 2:
        raise RuntimeError(
            "Checkpoint does not contain a valid LSTM state. "
            "Expected 'lstm.weight_hh_l0'."
        )

    hidden_dim = int(lstm_weight.shape[1])
    uses_attention = any(
        key.startswith("temporal_attention.")
        for key in state_dict
    )

    if backbone_type in {"efficientnet_v2", "efficientnet_v2_bilstm", "efficientnet"}:
        model = EfficientV2LSTMDetector(
            hidden_dim=hidden_dim,
            pretrained=False,
        )
    else:
        model = ResNeXtLSTMDetector(
            backbone_type="resnet18",
            hidden_dim=hidden_dim,
            pretrained=False,
            use_temporal_attention=uses_attention
        )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(device)

    model.eval()

    print("Model loaded successfully.")

    return model, device


def predict_video(
    video_path,
    model=None,
    device=None,
    num_frames=32,
    backbone_type="resnet18"
):

    if model is None:
        model, device = load_detector_model(
            backbone_type=backbone_type
        )

    multi_subjects = process_video_multi_subject(
        video_path,
        num_frames=num_frames
    )

    if not multi_subjects:
        raise RuntimeError(
            "No usable face frames were extracted from the video."
        )

    # Multi-subject evaluation: assess all people in frame and pool by max threat
    subject_predictions = []
    for face_sequence_np, face_pil_list in multi_subjects:
        if isinstance(model, MobileXGBDetector):
            s_fake_prob, s_frame_probs = model.analyze_sequence(face_sequence_np)
        elif HAS_TORCH and isinstance(model, torch.nn.Module):
            frame_tensors = [transform_pipeline(face) for face in face_sequence_np]
            video_tensor = torch.stack(frame_tensors, dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(video_tensor)

            if isinstance(output, tuple):
                logits, frame_logits = output
            else:
                logits = output
                frame_logits = None

            s_fake_prob = torch.sigmoid(logits).detach().cpu().item()

            if frame_logits is not None:
                s_frame_probs = torch.sigmoid(
                    frame_logits
                ).detach().cpu().reshape(-1).numpy().tolist()
            else:
                s_frame_probs = [s_fake_prob for _ in range(len(face_pil_list))]
        else:
            if not isinstance(model, SpatialTemporalForensicDetector):
                model = SpatialTemporalForensicDetector()
            s_fake_prob, s_frame_probs = model.analyze_sequence(face_sequence_np)

        subject_predictions.append((s_fake_prob, s_frame_probs, face_pil_list))

    # Highest risk subject determines the overall video authenticity
    max_subject = max(subject_predictions, key=lambda s: s[0])
    fake_prob = max_subject[0]
    frame_probs = max_subject[1]
    face_pil_list = max_subject[2]

    fake_prob = float(
        np.clip(
            fake_prob,
            0.0,
            1.0
        )
    )

    frame_probs = [
        float(
            np.clip(
                p,
                0.0,
                1.0
            )
        )
        for p in frame_probs
    ]

    real_prob = 1.0 - fake_prob

    is_fake = fake_prob >= 0.5

    verdict = "FAKE" if is_fake else "REAL"

    confidence = (
        fake_prob
        if is_fake
        else real_prob
    ) * 100.0

    return {
        "verdict": verdict,
        "is_fake": is_fake,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "confidence": confidence,
        "frame_probabilities": frame_probs,
        "face_frames": face_pil_list,
        "num_frames": len(face_pil_list)
    }