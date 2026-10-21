import os
import sys
import glob
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import ResNeXtLSTMDetector
from src.preprocessing import DeepfakeVideoDataset


def get_balanced_dataset_paths(data_dir="data", max_real=150, max_fake=150, seed=42):
    """
    Collects real and synthetic video paths from data folders and creates a balanced dataset list.
    0 = REAL, 1 = FAKE
    """
    random.seed(seed)
    
    # 1. Collect Real Videos (Celeb-real + YouTube-real)
    celeb_real = glob.glob(os.path.join(data_dir, "Celeb-real", "*.mp4"))
    yt_real = glob.glob(os.path.join(data_dir, "YouTube-real", "*.mp4"))
    all_real = celeb_real + yt_real
    
    # 2. Collect Fake Videos (Celeb-synthesis)
    all_fake = glob.glob(os.path.join(data_dir, "Celeb-synthesis", "*.mp4"))
    
    print(f"Total available Real videos: {len(all_real)}")
    print(f"Total available Fake videos: {len(all_fake)}")
    
    # Sample balanced subsets
    sampled_real = random.sample(all_real, min(max_real, len(all_real)))
    sampled_fake = random.sample(all_fake, min(max_fake, len(all_fake)))
    
    items = [(p, 0) for p in sampled_real] + [(p, 1) for p in sampled_fake]
    random.shuffle(items)
    print(f"Curated balanced dataset: {len(sampled_real)} Real (0) + {len(sampled_fake)} Fake (1) = {len(items)} videos.")
    return items

def compute_class_balance(items):
    """Return class counts and imbalance diagnostics for binary labels in 0/1 format."""
    counts = {0: 0, 1: 0}
    for _, label in items:
        label = int(label)
        if label not in counts:
            counts[label] = 0
        counts[label] += 1

    total = sum(counts.values())
    if total == 0:
        return {
            "counts": counts,
            "is_imbalanced": False,
            "imbalance_ratio": 1.0,
            "pos_weight": 1.0,
            "neg_weight": 1.0,
        }

    majority = max(counts.values())
    minority = min(v for v in counts.values() if v > 0) if any(v > 0 for v in counts.values()) else 0
    imbalance_ratio = (majority / minority) if minority > 0 else 1.0
    is_imbalanced = minority > 0 and imbalance_ratio > 1.2

    negative_count = counts.get(0, 0)
    positive_count = counts.get(1, 0)
    pos_weight = (negative_count / positive_count) if positive_count > 0 else 1.0
    neg_weight = (positive_count / negative_count) if negative_count > 0 else 1.0

    return {
        "counts": counts,
        "is_imbalanced": is_imbalanced,
        "imbalance_ratio": float(imbalance_ratio),
        "pos_weight": float(pos_weight),
        "neg_weight": float(neg_weight),
    }


def train_model(
    data_dir="data",
    output_model_path="deepfake_model.pt",
    backbone_type="resnet18",
    num_frames=32,
    batch_size=4,
    epochs=25,
    lr=1e-4,
    weight_decay=1e-3,
    max_real=150,
    max_fake=150,
    freeze_epochs=1,
    patience=6,
    device=None
):
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1")
    if freeze_epochs < 0 or freeze_epochs >= epochs:
        raise ValueError("freeze_epochs must be non-negative and less than epochs")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Dataset Splits
    dataset_items = get_balanced_dataset_paths(data_dir=data_dir, max_real=max_real, max_fake=max_fake)
    labels = [item[1] for item in dataset_items]
    class_count = len(set(labels))
    if len(dataset_items) < class_count * 2:
        raise ValueError(
            "Need at least two videos per class to create a stratified train/validation split."
        )
    validation_fraction = max(0.2, class_count / len(dataset_items))
    train_items, test_items = train_test_split(
        dataset_items,
        test_size=validation_fraction,
        random_state=42,
        stratify=labels
    )
    train_balance = compute_class_balance(train_items)
    print(f"Training label distribution: {train_balance['counts']} | imbalance_ratio={train_balance['imbalance_ratio']:.3f}")
    
    # Use augmented transforms for training, clean transforms for validation
    train_dataset = DeepfakeVideoDataset(train_items, num_frames=num_frames, is_training=True)
    test_dataset = DeepfakeVideoDataset(test_items, num_frames=num_frames, is_training=False)

    if train_balance["is_imbalanced"]:
        sample_weights = [1.0 / train_balance["counts"][int(label)] for _, label in train_items]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
        print("Detected class imbalance in training set -> using weighted sampler.")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        print("Training set is approximately balanced -> using standard shuffling.")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 2. Instantiate Model
    print(f"Instantiating model with backbone: {backbone_type}...")
    model = ResNeXtLSTMDetector(backbone_type=backbone_type, pretrained=True)
    model.to(device)
    
    # 3. Freeze backbone initially — let LSTM head learn first
    model.freeze_backbone()
    print(f"Backbone FROZEN for first {freeze_epochs} epoch(s) — training LSTM head only.")
    
    pos_weight = torch.tensor([train_balance["pos_weight"]], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight if train_balance["is_imbalanced"] else None)
    
    # Only optimize non-frozen parameters initially
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_without_improvement = 0
    best_model_state = None
    
    print(
        f"\n--- Starting Training ({len(train_loader)} train batches/epoch, "
        f"{len(test_loader)} validation batches/epoch) ---",
        flush=True
    )
    for epoch in range(1, epochs + 1):
        # Unfreeze backbone after freeze_epochs and rebuild optimizer with lower backbone LR
        if epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
            print(f"\n>> Epoch {epoch}: Backbone UNFROZEN — fine-tuning all layers with reduced backbone LR.")
            optimizer = torch.optim.AdamW([
                {"params": model.feature_extractor.parameters(), "lr": lr * 0.1},  # 10x lower for backbone
                {"params": model.feature_proj.parameters()},
                {"params": model.lstm.parameters()},
                {"params": model.frame_classifier.parameters()},
                {"params": model.classifier.parameters()},
            ], lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - freeze_epochs)
            )
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_started = time.perf_counter()
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(videos)
            logits = logits.squeeze(1)
            
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * videos.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            elapsed = time.perf_counter() - epoch_started
            print(
                f"\rEpoch [{epoch}/{epochs}] "
                f"batch [{batch_idx + 1}/{len(train_loader)}] "
                f"elapsed {elapsed:.0f}s",
                end="",
                flush=True
            )

        print(flush=True)
            
        scheduler.step()
        
        epoch_train_loss = running_loss / max(total, 1)
        epoch_train_acc = correct / max(total, 1)
        
        # Validation Step
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (videos, labels) in enumerate(test_loader):
                videos = videos.to(device)
                labels = labels.to(device)
                logits, _ = model(videos)
                logits = logits.squeeze(1)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * videos.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                print(
                    f"\rEpoch [{epoch}/{epochs}] validation "
                    f"batch [{batch_idx + 1}/{len(test_loader)}]",
                    end="",
                    flush=True
                )

        print(flush=True)
                
        epoch_val_loss = val_loss / max(val_total, 1)
        epoch_val_acc = val_correct / max(val_total, 1)
        
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        
        print(f"Epoch [{epoch}/{epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")
        
        # Save best model based on val_loss (more reliable than val_acc for overfitting)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), output_model_path)
            print(f"  -> Best model saved to {output_model_path} (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc*100:.2f}%)")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement for {epochs_without_improvement} epoch(s) (patience: {patience})")
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\n>> Early stopping triggered at epoch {epoch} (no val_loss improvement for {patience} epochs)")
            break
    
    # Restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model weights (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc*100:.2f}%)")
            
    # 4. Model Evaluation on Test Set
    evaluate_and_plot(model, test_loader, device)
    return model, history

def evaluate_and_plot(model, test_loader, device, cm_save_path="confusion_matrix.png"):
    """
    Computes confusion matrix, classification report, and plots confusion matrix heatmap.
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in test_loader:
            videos = videos.to(device)
            logits, _ = model(videos)
            probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().astype(int))
            
    cm = confusion_matrix(all_labels, all_preds)
    print("\n--- Confusion Matrix ---")
    print(cm)
    
    target_names = ["Real", "Fake"]
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    print("\n--- Classification Report ---")
    print(report)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth Label')
    plt.title('Deepfake Detection Confusion Matrix')
    plt.tight_layout()
    plt.savefig(cm_save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_save_path}")

if __name__ == "__main__":
    train_model()
