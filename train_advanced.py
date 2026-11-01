"""
Advanced Multi-Domain Forensic Training Pipeline for DeepDetect AI.

Fuses:
1. MobileNetV3-Large deep spatial semantic features (1,920 dims: mean + std)
2. 2D Fast Fourier Transform (FFT) radial spectrum power distribution (10 dims)
3. YCbCr Chrominance boundary discrepancy (Delta-Cb, Delta-Cr inner vs outer face) (8 dims)
4. Laplacian texture sharpness & edge variance ratios (6 dims)
5. Error Level Analysis (ELA) compression residual statistics (6 dims)
6. Inter-frame temporal motion jitter & MSE continuity (3 dims)

Total: 1,953 Multi-Domain Forensic Descriptors.
Trains XGBoost on balanced Real (Celeb-real + YouTube-real) and Fake (Celeb-synthesis) videos.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# Ensure project root in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models_zoo import MobileXGBDetector
from src.preprocessing import process_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_advanced")


def collect_dataset(
    data_dir: str = "data",
    limit_per_class: int | None = 300,
    seed: int = 42,
) -> list[tuple[str, int]]:
    """
    Collect and balance Real (Celeb-real + YouTube-real) and Fake (Celeb-synthesis) videos.
    0 = REAL, 1 = FAKE
    """
    root = Path(data_dir)
    real_paths = sorted(list((root / "Celeb-real").glob("*.mp4")) + list((root / "YouTube-real").glob("*.mp4")))
    fake_paths = sorted(list((root / "Celeb-synthesis").glob("*.mp4")))

    logger.info("Found %d Real videos (Celeb-real + YouTube-real)", len(real_paths))
    logger.info("Found %d Fake videos (Celeb-synthesis)", len(fake_paths))

    rng = random.Random(seed)
    if limit_per_class and limit_per_class > 0:
        n_real = min(limit_per_class, len(real_paths))
        n_fake = min(limit_per_class, len(fake_paths))
        sampled_real = rng.sample(real_paths, n_real)
        sampled_fake = rng.sample(fake_paths, n_fake)
    else:
        # Match real count if full mode
        n_real = len(real_paths)
        n_fake = min(len(real_paths), len(fake_paths))
        sampled_real = real_paths
        sampled_fake = rng.sample(fake_paths, n_fake)

    items = [(str(p), 0) for p in sampled_real] + [(str(p), 1) for p in sampled_fake]
    rng.shuffle(items)
    logger.info("Curated balanced training set: %d Real, %d Fake (Total: %d)", len(sampled_real), len(sampled_fake), len(items))
    return items


def extract_features_with_cache(
    detector: MobileXGBDetector,
    video_items: list[tuple[str, int]],
    num_frames: int = 32,
    cache_dir: str = ".features_cache",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 1953 multi-domain forensic features for each video with persistent disk cache."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    x_list: list[np.ndarray] = []
    y_list: list[int] = []

    logger.info("Extracting multi-domain forensic features (num_frames=%d)...", num_frames)
    t0 = time.perf_counter()

    for idx, (vpath, label) in enumerate(video_items):
        vname = Path(vpath).stem
        feat_file = cache_path / f"{vname}_{num_frames}f_v2.npy"

        if feat_file.is_file():
            vec = np.load(feat_file)
        else:
            try:
                face_crops, _ = process_video(vpath, num_frames=num_frames)
                if len(face_crops) == 0:
                    continue
                # Extract MobileNet (1920) + Multi-Domain Forensic Descriptors (33) = 1953
                vec = detector.extract_features(face_crops, include_forensic_descriptors=True)
                np.save(feat_file, vec)
            except Exception as exc:
                logger.warning("Failed processing %s: %s", vpath, exc)
                continue

        x_list.append(vec)
        y_list.append(label)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(video_items):
            elapsed = time.perf_counter() - t0
            logger.info("Processed %d / %d videos (%.1f s)", idx + 1, len(video_items), elapsed)

    return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def train_advanced_model(
    data_dir: str = "data",
    output_dir: str = "artifacts_full",
    limit_per_class: int | None = 300,
    num_frames: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Train the advanced multi-domain forensic model and evaluate on held-out test split."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    detector = MobileXGBDetector(pretrained=True)

    items = collect_dataset(data_dir=data_dir, limit_per_class=limit_per_class, seed=seed)
    test_size = 2 if len(items) < 10 else max(2, int(0.20 * len(items)))
    train_items, test_items = train_test_split(
        items, test_size=test_size, random_state=seed, stratify=[y for _, y in items]
    )

    logger.info("Train split: %d videos | Held-out test split: %d videos", len(train_items), len(test_items))

    x_train, y_train = extract_features_with_cache(detector, train_items, num_frames=num_frames)
    x_test, y_test = extract_features_with_cache(detector, test_items, num_frames=num_frames)

    logger.info("Feature matrix shapes: X_train=%s, X_test=%s", x_train.shape, x_test.shape)
    logger.info("Fitting XGBoost Classifier with Multi-Domain Forensic Representations...")

    detector.fit(x_train, y_train)

    # Predictions & probabilities
    test_probs = detector.xgb.predict_proba(x_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    # Find optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(y_test, test_probs)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    optimal_threshold = float(thresholds[best_idx])
    optimal_preds = (test_probs >= optimal_threshold).astype(int)

    acc = float(accuracy_score(y_test, test_preds))
    bal_acc = float(balanced_accuracy_score(y_test, test_preds))
    prec = float(precision_score(y_test, test_preds, zero_division=0))
    rec = float(recall_score(y_test, test_preds, zero_division=0))
    f1 = float(f1_score(y_test, test_preds, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, test_probs))
    cm = confusion_matrix(y_test, test_preds).tolist()

    logger.info("========================================================")
    logger.info(" HELD-OUT TEST EVALUATION METRICS (Default 0.5 Threshold)")
    logger.info("========================================================")
    logger.info(" Accuracy:          %.2f%%", acc * 100)
    logger.info(" Balanced Accuracy: %.2f%%", bal_acc * 100)
    logger.info(" Precision (Fake):  %.2f%%", prec * 100)
    logger.info(" Recall (Fake):     %.2f%%", rec * 100)
    logger.info(" F1-Score:          %.2f%%", f1 * 100)
    logger.info(" ROC-AUC:           %.4f", roc_auc)
    logger.info(" Confusion Matrix (TN, FP / FN, TP): %s", cm)
    logger.info(" Optimal Threshold (Youden J): %.3f", optimal_threshold)

    # Save trained checkpoint
    save_file = output_path / "mobilenet_xgboost.joblib"
    detector.save(str(save_file))
    logger.info("Saved trained model to: %s", save_file)

    report = {
        "timestamp": time.time(),
        "seed": seed,
        "winner": "mobilenet_xgboost",
        "num_features": int(x_train.shape[1]),
        "num_frames": num_frames,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "optimal_threshold": round(optimal_threshold, 3),
        "metrics": {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
        },
        "production_training": "MobileNetV3-Large + Multi-Domain Forensic Descriptors (FFT, Color, Texture, ELA, Jitter) + XGBoost",
    }

    report_file = output_path / "benchmark_results.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved benchmark report to: %s", report_file)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepDetect AI Advanced Forensic Training")
    parser.add_argument("--data-dir", default="data", help="Root data directory containing Celeb-real, YouTube-real, Celeb-synthesis")
    parser.add_argument("--output-dir", default="artifacts_full", help="Output directory for model checkpoint and report")
    parser.add_argument("--limit", type=int, default=300, help="Number of videos per class (default: 300). Pass 0 or --full for all.")
    parser.add_argument("--full", action="store_true", help="Train on all available dataset videos")
    parser.add_argument("--num-frames", type=int, default=32, help="Temporal sequence length (default: 32)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    limit = None if args.full or args.limit <= 0 else args.limit
    train_advanced_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        limit_per_class=limit,
        num_frames=args.num_frames,
        seed=args.seed,
    )
