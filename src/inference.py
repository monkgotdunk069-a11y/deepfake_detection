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
from src.preprocessing import process_video


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

    face_sequence_np, face_pil_list = process_video(
        video_path,
        num_frames=num_frames
    )

    if len(face_sequence_np) == 0:
        raise RuntimeError(
            "No usable face frames were extracted from the video."
        )

    if isinstance(model, MobileXGBDetector):
        fake_prob, frame_probs = model.analyze_sequence(face_sequence_np)
    elif HAS_TORCH and isinstance(model, torch.nn.Module):

        frame_tensors = []

        for face in face_sequence_np:

            tensor = transform_pipeline(face)

            frame_tensors.append(tensor)

        video_tensor = torch.stack(
            frame_tensors,
            dim=0
        ).unsqueeze(0)

        video_tensor = video_tensor.to(device)

        with torch.no_grad():

            output = model(video_tensor)

        if isinstance(output, tuple):

            logits, frame_logits = output

        else:

            logits = output
            frame_logits = None

        fake_prob = torch.sigmoid(
            logits
        ).detach().cpu().item()

        if frame_logits is not None:

            frame_probs = torch.sigmoid(
                frame_logits
            ).detach().cpu().reshape(-1).numpy().tolist()

        else:

            frame_probs = [
                fake_prob
                for _ in range(len(face_pil_list))
            ]

    else:

        if not isinstance(
            model,
            SpatialTemporalForensicDetector
        ):
            model = SpatialTemporalForensicDetector()

        fake_prob, frame_probs = model.analyze_sequence(
            face_sequence_np
        )

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