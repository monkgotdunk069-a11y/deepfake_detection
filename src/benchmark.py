"""Reproducible comparison of the baseline and upgraded detector models.

The command is intentionally data-driven: the production model is selected
from held-out test metrics, not from a hard-coded architecture preference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models_zoo import MobileXGBDetector, build_sequence_model
from src.preprocessing import DeepfakeVideoDataset, process_video


def collect_items(data_dir: str, limit_per_class: int | None = None, seed: int = 42) -> list[tuple[str, int]]:
    root = Path(data_dir)
    real = sorted((root / "Celeb-real").glob("*.mp4"))
    real += sorted((root / "YouTube-real").glob("*.mp4"))
    fake = sorted((root / "Celeb-synthesis").glob("*.mp4"))
    rng = random.Random(seed)
    if limit_per_class:
        real = rng.sample(real, min(limit_per_class, len(real)))
        fake = rng.sample(fake, min(limit_per_class, len(fake)))
    items = [(str(path), 0) for path in real] + [(str(path), 1) for path in fake]
    rng.shuffle(items)
    if len({label for _, label in items}) != 2:
        raise ValueError("The dataset must contain both real and fake videos.")
    return items


def _metrics(labels: list[int], probabilities: list[float]) -> dict[str, float]:
    predictions = [int(p >= 0.5) for p in probabilities]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for videos, batch_labels in loader:
            output = model(videos.to(device))
            logits = output[0] if isinstance(output, tuple) else output
            probabilities.extend(torch.sigmoid(logits.reshape(-1)).cpu().tolist())
            labels.extend(batch_labels.int().tolist())
    return _metrics(labels, probabilities)


def _train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> tuple[nn.Module, dict[str, float]]:
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    best: tuple[float, dict[str, Any]] | None = None
    for _ in range(epochs):
        model.train()
        for videos, labels in train_loader:
            optimizer.zero_grad()
            output = model(videos.to(device))
            logits = output[0] if isinstance(output, tuple) else output
            loss = criterion(logits.reshape(-1), labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation = _evaluate(model, val_loader, device)
        score = validation["balanced_accuracy"]
        if best is None or score > best[0]:
            best = (score, validation)
    return model, best[1] if best else {}


def run_benchmark(
    data_dir: str = "data",
    output_dir: str = "artifacts",
    limit_per_class: int | None = None,
    num_frames: int = 16,
    epochs: int = 3,
    batch_size: int = 2,
    pretrained: bool = True,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Train and evaluate all three candidates on identical video splits."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    items = collect_items(data_dir, limit_per_class, seed)
    train_items, test_items = train_test_split(
        items, test_size=0.25, random_state=seed, stratify=[label for _, label in items]
    )
    train_items, val_items = train_test_split(
        train_items, test_size=0.2, random_state=seed, stratify=[label for _, label in train_items]
    )
    datasets = [
        DeepfakeVideoDataset(split, num_frames=num_frames, is_training=name == "train", cache_dir="data_cache")
        for name, split in (("train", train_items), ("val", val_items), ("test", test_items))
    ]
    train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(datasets[1], batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(datasets[2], batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}

    for name in ("resnet18_bilstm", "efficientnet_v2_bilstm"):
        seq_model = build_sequence_model(name, pretrained=pretrained)
        if isinstance(seq_model, nn.Module):
            trained_model, _ = _train_torch_model(seq_model, train_loader, val_loader, device, epochs)
            results[name] = _evaluate(trained_model, test_loader, device)
            torch.save(trained_model.state_dict(), output_path / f"{name}.pt")

    mobile = build_sequence_model("mobilenet_xgboost", pretrained=pretrained)
    assert isinstance(mobile, MobileXGBDetector)

    def features(split: list[tuple[str, int]]) -> np.ndarray:
        return np.stack([mobile.extract_features(process_video(path, num_frames)[0]) for path, _ in split])

    x_train, y_train = features(train_items), np.array([label for _, label in train_items])
    x_test, y_test = features(test_items), np.array([label for _, label in test_items])
    mobile.fit(x_train, y_train)
    mobile.save(str(output_path / "mobilenet_xgboost.joblib"))
    mobile_probabilities = mobile.xgb.predict_proba(x_test)[:, 1].tolist()
    results["mobilenet_xgboost"] = _metrics(y_test.tolist(), mobile_probabilities)

    winner = max(results, key=lambda key: (results[key]["balanced_accuracy"], results[key]["roc_auc"]))
    report = {"seed": seed, "device": str(device), "winner": winner, "results": results}
    (output_path / "benchmark_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--limit-per-class", type=int)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_benchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        limit_per_class=args.limit_per_class,
        num_frames=args.num_frames,
        epochs=args.epochs,
        batch_size=args.batch_size,
        pretrained=not args.no_pretrained,
    ), indent=2))
