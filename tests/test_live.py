"""
Unit and integration tests for Live Deepfake Detection.
Validates SCRFD face detection, MobileNet-V3 + XGBoost 32-frame sequence inference,
WebSocket session management, and live status endpoints.
"""

import base64
import cv2
import numpy as np

from src.live_detector import (
    LiveDetectionSession,
    get_live_model,
    crop_face_patch,
    to_thumbnail_data_url,
)
from src.face_detector import detect_faces, get_detector, get_detector_name
from src.models_zoo import MobileXGBDetector
from src.api import app


def test_live_model_loading():
    """Verify that the MobileNet-V3 + XGBoost detector loads properly with 32-frame sequence support."""
    model = get_live_model()
    assert model is not None
    assert isinstance(model, MobileXGBDetector)
    assert hasattr(model, "analyze_sequence")
    assert hasattr(model, "xgb")

    # Test forward pass with 32 synthetic face frames (32, 224, 224, 3) uint8
    dummy_seq = np.random.randint(0, 255, (32, 224, 224, 3), dtype=np.uint8)
    prob, frame_probs = model.analyze_sequence(dummy_seq)
    assert 0.0 <= prob <= 1.0
    assert len(frame_probs) == 32
    for p in frame_probs:
        assert 0.0 <= p <= 1.0


def test_scrfd_detector_integration():
    """Verify that SCRFD detector is active and functional."""
    det = get_detector()
    assert det is not None
    name = get_detector_name()
    assert "SCRFD" in name

    # Test detection on synthetic image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = detect_faces(img)
    assert isinstance(boxes, list)


def test_face_patch_cropping():
    """Verify that face cropping and resizing to (224, 224) works accurately."""
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    class MockBox:
        x, y, w, h = 100, 100, 120, 120
        confidence = 0.95

    patch = crop_face_patch(img, MockBox(), margin_ratio=0.2, target_size=(224, 224))
    assert patch.shape == (224, 224, 3)
    assert patch.dtype == np.uint8

    thumb_url = to_thumbnail_data_url(patch)
    assert thumb_url.startswith("data:image/jpeg;base64,")


def test_live_session_lifecycle():
    """Verify state transitions, sequence buffer accumulation, and non-mock predictions."""
    session = LiveDetectionSession(target_frames=32)
    assert session.target_frames == 32
    assert len(session.sequence_buffer) == 0

    # 1. Non-face frame
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res_no_face = session.process_frame(black_frame)
    assert res_no_face["type"] == "result"
    assert res_no_face["face_detected"] is False
    assert res_no_face["box"] is None
    assert res_no_face["buffer_count"] == 0
    assert res_no_face["buffer_target"] == 32

    # 2. Simulate frames by adding face-like patterns or synthetic face patch
    mock_face = np.full((224, 224, 3), 128, dtype=np.uint8)
    for _ in range(10):
        session.sequence_buffer.append(mock_face)

    assert len(session.sequence_buffer) == 10

    # 3. Test direct sequence inference on 32-frame buffer
    frames_32 = [mock_face] * 32
    pred = session._run_model_inference_sync(frames_32)
    assert "fake_score" in pred
    assert "real_score" in pred
    assert "verdict" in pred
    assert "risk_level" in pred
    assert len(pred["frame_scores"]) == 32
    assert 0.0 <= pred["fake_score"] <= 1.0
    assert abs((pred["fake_score"] + pred["real_score"]) - 1.0) < 1e-4

    # 4. Reset
    session.reset()
    assert len(session.sequence_buffer) == 0
    assert session.frame_count == 0

    session.close()


def test_frame_image_decoding():
    """Verify that session decodes both data-URLs and raw bytes."""
    session = LiveDetectionSession(target_frames=32)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    b64_url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

    decoded_from_url = session.decode_image(b64_url)
    assert decoded_from_url is not None
    assert decoded_from_url.shape == (100, 100, 3)

    decoded_from_bytes = session.decode_image(buf.tobytes())
    assert decoded_from_bytes is not None
    assert decoded_from_bytes.shape == (100, 100, 3)
    session.close()


def test_live_status_endpoint():
    """Verify FastAPI /api/live/status route response directly."""
    from src.api import live_status

    res = live_status()
    assert res["status"] == "ready"
    assert "MobileNet" in res["model"]
    assert "XGBoost" in res["model"]
    assert res["sequence_frames"] == 32
    assert res["ws_endpoint"] == "/api/ws/live"

    routes = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/api/live/status" in routes
    assert "/api/ws/live" in routes
    assert "/ws/live" in routes
