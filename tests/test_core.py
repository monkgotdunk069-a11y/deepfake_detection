import os
import tempfile
import numpy as np
from PIL import Image

from src.forensics import compute_fft_spectrum, compute_ela
from src.auth import hash_password, verify_password, jwt_sign, jwt_verify
import src.database as db


def test_password_hashing():
    pw = "SuperSecretPassword123!"
    hashed = hash_password(pw)
    assert hashed != pw
    assert verify_password(pw, hashed) is True
    assert verify_password("WrongPassword", hashed) is False


def test_jwt_sign_and_verify():
    payload = {"sub": "user_test_123", "role": "admin", "exp": 9999999999}
    token = jwt_sign(payload)
    assert isinstance(token, str)
    assert len(token.split(".")) == 3

    decoded = jwt_verify(token)
    assert decoded["sub"] == "user_test_123"
    assert decoded["role"] == "admin"


def test_forensic_fft_and_ela():
    img_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    img_arr[20:80, 20:80] = [255, 128, 64]
    test_img = Image.fromarray(img_arr)

    fft_img = compute_fft_spectrum(test_img)
    assert isinstance(fft_img, Image.Image)
    assert fft_img.size == (100, 100)

    ela_img = compute_ela(test_img)
    assert isinstance(ela_img, Image.Image)
    assert ela_img.size == (100, 100)


def test_database_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test.db")
        original_db = db.DB_PATH
        try:
            db.DB_PATH = test_db
            db.init_db()

            db.insert_job(
                job_id="test-job-001",
                filename="sample.mp4",
                source="upload",
                video_path="/path/to/sample.mp4",
                num_frames=16,
                metadata={"width": 1920, "height": 1080},
                batch_id="batch-001",
            )

            job = db.get_job("test-job-001")
            assert job is not None
            assert job["status"] == "queued"
            assert job["filename"] == "sample.mp4"
            assert job["batch_id"] == "batch-001"
            assert job["metadata"]["width"] == 1920

            db.set_job_running("test-job-001")
            running = db.get_job("test-job-001")
            assert running["status"] == "running"

            db.set_job_completed(
                "test-job-001",
                result={"fake_score": 0.85, "real_score": 0.15},
            )

            updated = db.get_job("test-job-001")
            assert updated["status"] == "completed"
            assert updated["result"]["fake_score"] == 0.85

            batch_jobs = db.get_jobs_in_batch("batch-001")
            assert len(batch_jobs) == 1
            assert batch_jobs[0]["id"] == "test-job-001"
        finally:
            db.DB_PATH = original_db


def test_preprocessing_utilities():
    from src.preprocessing import (
        normalize_face_sequence,
        train_transform,
        test_transform,
        default_transform,
        detect_and_crop_face,
        DeepfakeVideoDataset,
    )
    import torch

    # Verify normalization shape and type
    dummy_seq = np.zeros((4, 224, 224, 3), dtype=np.uint8)
    norm = normalize_face_sequence(dummy_seq)
    assert norm.shape == (4, 3, 224, 224)
    assert norm.dtype == np.float32

    # Verify transforms exist and are callable
    assert callable(train_transform)
    assert callable(test_transform)
    assert callable(default_transform)
    assert callable(detect_and_crop_face)

    # Verify DeepfakeVideoDataset initialization and transform pipeline
    ds = DeepfakeVideoDataset(
        video_items=[("dummy.mp4", 1)],
        num_frames=4,
        is_training=True,
    )
    assert len(ds) == 1
    # Test transform_sequence directly with dummy frames
    tensors = ds.transform_sequence(list(dummy_seq))
    assert len(tensors) == 4
    assert isinstance(tensors[0], torch.Tensor)
    assert tensors[0].shape == (3, 224, 224)

