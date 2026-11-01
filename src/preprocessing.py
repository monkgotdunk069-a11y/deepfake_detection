"""
Preprocessing utilities for DeepDetect.

Provides frame extraction, face alignment/cropping, normalization,
and PyTorch dataset loaders for deepfake detection training and inference.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# Pluggable face detector — re-exported for backward compatibility
from src.face_detector import (
    detect_and_crop_face as detect_and_crop_face,
    detect_faces,
)

__all__ = [
    "extract_frames",
    "process_video",
    "process_video_multi_subject",
    "normalize_face_sequence",
    "train_transform",
    "test_transform",
    "default_transform",
    "DeepfakeVideoDataset",
    "detect_and_crop_face",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "HAS_TORCH",
]


def extract_frames(video_path: str, num_frames: int = 32) -> List[np.ndarray]:
    """
    Extracts evenly spaced frames from a video file.
    
    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to extract uniformly.
        
    Returns:
        List of RGB frames as uint8 arrays (H, W, 3).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1
        
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
        while len(frame_indices) < num_frames:
            frame_indices.append(frame_indices[-1])
    else:
        frame_indices = [int(x) for x in np.linspace(0, total_frames - 1, num_frames)]
        
    frames: List[np.ndarray] = []
    current_idx = 0
    selected_set = set(frame_indices)
    
    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if current_idx in selected_set:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        current_idx += 1
        
    cap.release()
    
    while len(frames) < num_frames:
        if len(frames) > 0:
            frames.append(frames[-1].copy())
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
    return frames[:num_frames]


def process_video(
    video_path: str,
    num_frames: int = 32,
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[np.ndarray, List[Image.Image]]:
    """
    Extracts frames and crops faces to produce a normalized face sequence.
    
    Returns:
        tuple: (face_sequence_np (N, H, W, 3), list_of_raw_face_pil_images)
    """
    raw_frames = extract_frames(video_path, num_frames=num_frames)
    cropped_faces: List[np.ndarray] = []
    last_box = None
    
    for frame in raw_frames:
        face, box = detect_and_crop_face(frame, target_size=target_size, last_box=last_box)
        if box is not None:
            last_box = box
        cropped_faces.append(face)
        
    pil_images = [Image.fromarray(face) for face in cropped_faces]
    np_sequence = np.stack(cropped_faces, axis=0)
    return np_sequence, pil_images


def process_video_multi_subject(
    video_path: str,
    num_frames: int = 32,
    target_size: Tuple[int, int] = (224, 224)
) -> List[Tuple[np.ndarray, List[Image.Image]]]:
    """
    Extracts and tracks face sequences for every distinct subject present in the video.
    Ensures that multi-person videos evaluate each person independently.
    
    Returns:
        List of (face_sequence_np (N, H, W, 3), list_of_raw_face_pil_images) per subject.
    """
    raw_frames = extract_frames(video_path, num_frames=num_frames)
    if not raw_frames:
        return []

    all_faces_per_frame = [detect_faces(f) for f in raw_frames]

    # Find the frame with the maximum number of detected faces as initial reference tracks
    max_frame_idx = max(range(len(all_faces_per_frame)), key=lambda i: len(all_faces_per_frame[i]))
    ref_faces = all_faces_per_frame[max_frame_idx]

    if not ref_faces:
        # Fallback to single sequence process_video
        seq_np, pil_imgs = process_video(video_path, num_frames=num_frames, target_size=target_size)
        return [(seq_np, pil_imgs)]

    # Sort reference faces left-to-right (by x coordinate) for stable subject ordering
    ref_faces = sorted(ref_faces, key=lambda f: f.x)

    subjects: List[Tuple[np.ndarray, List[Image.Image]]] = []
    for ref_f in ref_faces:
        subject_crops: List[np.ndarray] = []
        last_box = ref_f
        for frame, frame_faces in zip(raw_frames, all_faces_per_frame):
            if frame_faces:
                ref_cx = last_box.x + last_box.w / 2.0
                ref_cy = last_box.y + last_box.h / 2.0
                best_f = min(
                    frame_faces,
                    key=lambda f: (f.x + f.w / 2.0 - ref_cx) ** 2 + (f.y + f.h / 2.0 - ref_cy) ** 2
                )
                last_box = best_f
                h, w = frame.shape[:2]
                x1 = max(0, best_f.x)
                y1 = max(0, best_f.y)
                x2 = min(w, best_f.x + best_f.w)
                y2 = min(h, best_f.y + best_f.h)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop = cv2.resize(crop, target_size)
                else:
                    crop = np.zeros((*target_size, 3), dtype=np.uint8)
            else:
                crop = subject_crops[-1].copy() if subject_crops else np.zeros((*target_size, 3), dtype=np.uint8)
            subject_crops.append(crop)

        pil_images = [Image.fromarray(c) for c in subject_crops]
        seq_np = np.stack(subject_crops, axis=0)
        subjects.append((seq_np, pil_images))

    return subjects


# Standard normalization constants (ImageNet standard)
IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_face_sequence(face_sequence_np: np.ndarray) -> np.ndarray:
    """
    Normalizes a (N, 224, 224, 3) uint8 image sequence into (N, 3, 224, 224) float32 array.
    """
    normalized = (face_sequence_np.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(normalized, (0, 3, 1, 2))  # (N, 3, H, W)


# ---------------------------------------------------------------------------
# Torch transforms for training & evaluation
# ---------------------------------------------------------------------------

train_transform: transforms.Compose = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
        transforms.RandomAffine(
            degrees=8,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=4
        )
    ], p=0.6),
    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.2,
        hue=0.08
    ),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
    ], p=0.25),
    transforms.RandomGrayscale(p=0.03),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform: transforms.Compose = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

default_transform: transforms.Compose = test_transform
HAS_TORCH: bool = True


# ---------------------------------------------------------------------------
# PyTorch Dataset for video sequences
# ---------------------------------------------------------------------------

class DeepfakeVideoDataset(Dataset):
    """
    PyTorch Dataset for loading and batching deepfake video sequences.
    Uses augmented transforms during training and clean transforms during evaluation.
    """
    def __init__(
        self,
        video_items: Sequence[Tuple[str, Union[int, float]]],
        num_frames: int = 32,
        transform: Optional[Any] = None,
        is_training: bool = False,
        cache_dir: Optional[str] = None
    ) -> None:
        self.video_items = list(video_items)
        self.num_frames = num_frames
        self.is_training = is_training
        self.cache_dir = cache_dir
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            
        if transform is not None:
            self.transform: Optional[Any] = transform
        else:
            self.transform = train_transform if is_training else test_transform

    def __len__(self) -> int:
        return len(self.video_items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path, label = self.video_items[idx]
        try:
            face_seq = self._load_face_sequence(video_path)
            if self.is_training:
                tensors = self._transform_sequence(face_seq)
                video_tensor = torch.stack(tensors, dim=0)
            elif self.transform is not None:
                t = self.transform
                tensor_list: List[torch.Tensor] = [t(face) for face in face_seq]
                video_tensor = torch.stack(tensor_list, dim=0)
            else:
                norm_seq = normalize_face_sequence(face_seq)
                video_tensor = torch.from_numpy(norm_seq).float()
        except Exception:
            video_tensor = torch.zeros((self.num_frames, 3, 224, 224), dtype=torch.float32)

        return video_tensor, torch.tensor(label, dtype=torch.float32)

    def _cache_path(self, video_path: str) -> str:
        if not self.cache_dir:
            return ""
        key = hashlib.sha1(
            f"{os.path.abspath(video_path)}:{self.num_frames}".encode()
        ).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.npy")

    def _load_face_sequence(self, video_path: str) -> np.ndarray:
        if not self.cache_dir:
            face_seq, _ = process_video(video_path, num_frames=self.num_frames)
            return face_seq

        cache_path = self._cache_path(video_path)
        if cache_path and os.path.exists(cache_path):
            return np.load(cache_path)

        face_seq, _ = process_video(video_path, num_frames=self.num_frames)
        if cache_path:
            np.save(cache_path, face_seq)
        return face_seq

    def _transform_sequence(self, face_seq: Any) -> List[torch.Tensor]:
        params = {
            "flip": bool(torch.rand(()) < 0.5),
            "angle": float(torch.empty(1).uniform_(-8, 8).item()),
            "translate": [
                int(torch.empty(1).uniform_(-11, 12).item()),
                int(torch.empty(1).uniform_(-11, 12).item())
            ],
            "scale": float(torch.empty(1).uniform_(0.95, 1.05).item()),
            "shear": [
                float(torch.empty(1).uniform_(-4, 4).item()),
                0.0
            ]
        }

        if isinstance(face_seq, np.ndarray):
            raw_tensor = torch.from_numpy(face_seq).permute(0, 3, 1, 2).float().div_(255.0)
        else:
            frame_list: List[torch.Tensor] = []
            for f in face_seq:
                if isinstance(f, Image.Image):
                    arr = np.array(f)
                else:
                    arr = np.asarray(f)
                frame_list.append(torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0))
            raw_tensor = torch.stack(frame_list, dim=0)

        if raw_tensor.shape[-2:] != (224, 224):
            raw_tensor = TF.resize(raw_tensor, [224, 224])

        if params["flip"]:
            raw_tensor = TF.hflip(raw_tensor)

        raw_tensor = TF.affine(
            raw_tensor,
            angle=params["angle"],
            translate=params["translate"],
            scale=params["scale"],
            shear=params["shear"]
        )

        norm_tensor = TF.normalize(
            raw_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        return list(torch.unbind(norm_tensor, dim=0))

    # Public alias for external test callers
    transform_sequence = _transform_sequence
