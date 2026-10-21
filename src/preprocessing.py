import os
import hashlib
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# OpenCV Haar Cascade for face detection
HAAR_CASCADE_PATH = os.path.join(
    os.path.dirname(cv2.__file__ or ""),
    "data",
    "haarcascade_frontalface_default.xml"
)
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

def extract_frames(video_path, num_frames=32):
    """
    Extracts evenly spaced frames from a video file.
    
    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to extract uniformly.
        
    Returns:
        list[np.ndarray]: List of RGB frames (H, W, 3).
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
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
    frames = []
    current_idx = 0
    selected_set = set(frame_indices)
    
    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
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


def detect_and_crop_face(frame_rgb, target_size=(224, 224), margin_ratio=0.2, last_box=None):
    """
    Detects the primary face in a frame, adds margin, and crops/resizes it.
    
    Args:
        frame_rgb (np.ndarray): Input RGB frame (H, W, 3).
        target_size (tuple): Output (width, height).
        margin_ratio (float): Ratio to expand the bounding box for boundary context.
        last_box (tuple): Optional previous bounding box (x, y, w, h) for temporal continuity.
        
    Returns:
        tuple: (cropped_face_rgb np.ndarray, bounding_box tuple)
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40)
    )
    
    h_img, w_img, _ = frame_rgb.shape
    
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
    elif last_box is not None:
        x, y, w, h = last_box
    else:
        w = int(min(h_img, w_img) * 0.7)
        h = w
        x = (w_img - w) // 2
        y = (h_img - h) // 2
        
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w_img, x + w + margin_x)
    y2 = min(h_img, y + h + margin_y)
    
    cropped = frame_rgb[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = frame_rgb
        
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
    return resized, (x, y, w, h)


def process_video(video_path, num_frames=32, target_size=(224, 224)):
    """
    Extracts frames and crops faces to produce a normalized face sequence.
    
    Returns:
        tuple: (face_sequence_np (N, H, W, 3), list_of_raw_face_pil_images)
    """
    raw_frames = extract_frames(video_path, num_frames=num_frames)
    cropped_faces = []
    last_box = None
    
    for frame in raw_frames:
        face, box = detect_and_crop_face(frame, target_size=target_size, last_box=last_box)
        if box is not None:
            last_box = box
        cropped_faces.append(face)
        
    pil_images = [Image.fromarray(face) for face in cropped_faces]
    np_sequence = np.stack(cropped_faces, axis=0)
    return np_sequence, pil_images


# Standard normalization constants (ImageNet standard)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def normalize_face_sequence(face_sequence_np):
    """
    Normalizes a (N, 224, 224, 3) uint8 image sequence into (N, 3, 224, 224) float32 array.
    """
    normalized = (face_sequence_np.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(normalized, (0, 3, 1, 2)) # (N, 3, H, W)


try:
    # Training-only augmentation improves robustness to compression, lighting,
    # small pose changes, and frame crops while preserving face structure.
    train_transform = transforms.Compose([
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

    # Test/inference transform — no augmentation, clean normalization only
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Keep backward compatibility alias
    default_transform = test_transform
    HAS_TORCH = True
except Exception:
    train_transform = None
    test_transform = None
    default_transform = None
    HAS_TORCH = False


class DeepfakeVideoDataset(Dataset):
    """
    PyTorch Dataset for loading and batching deepfake video sequences.
    Uses augmented transforms during training and clean transforms during evaluation.
    """
    def __init__(
        self,
        video_items,
        num_frames=32,
        transform=None,
        is_training=False,
        cache_dir=None
    ):
        self.video_items = video_items
        self.num_frames = num_frames
        self.is_training = is_training
        self.cache_dir = cache_dir
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        # Auto-select transform based on training/eval mode if not explicitly provided
        if transform is not None:
            self.transform = transform
        elif HAS_TORCH:
            self.transform = train_transform if is_training else test_transform
        else:
            self.transform = None

    def __len__(self):
        return len(self.video_items)

    def __getitem__(self, idx):
        video_path, label = self.video_items[idx]
        try:
            face_seq = self._load_face_sequence(video_path)
            if self.is_training and self.transform is not None:
                tensors = self._transform_sequence(face_seq)
                video_tensor = torch.stack(tensors, dim=0)
            elif self.transform is not None:
                tensors = [self.transform(face) for face in face_seq]
                video_tensor = torch.stack(tensors, dim=0)  # (num_frames, 3, 224, 224)
            else:
                norm_seq = normalize_face_sequence(face_seq)
                video_tensor = torch.from_numpy(norm_seq).float()
        except Exception:
            if HAS_TORCH:
                video_tensor = torch.zeros((self.num_frames, 3, 224, 224), dtype=torch.float32)
            else:
                video_tensor = np.zeros((self.num_frames, 3, 224, 224), dtype=np.float32)

        if HAS_TORCH:
            return video_tensor, torch.tensor(label, dtype=torch.float32)
        return video_tensor, label

    def _cache_path(self, video_path):
        key = hashlib.sha1(
            f"{os.path.abspath(video_path)}:{self.num_frames}".encode()
        ).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.npy")

    def _load_face_sequence(self, video_path):
        if self.cache_dir is None:
            face_seq, _ = process_video(video_path, num_frames=self.num_frames)
            return face_seq

        cache_path = self._cache_path(video_path)
        if os.path.exists(cache_path):
            return np.load(cache_path)

        face_seq, _ = process_video(video_path, num_frames=self.num_frames)
        np.save(cache_path, face_seq)
        return face_seq

    def _transform_sequence(self, face_seq):
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
        tensors = []
        for face in face_seq:
            image = self.transform.transforms[0](face)
            image = TF.resize(image, [224, 224])
            if params["flip"]:
                image = TF.hflip(image)
            image = TF.affine(
                image,
                angle=params["angle"],
                translate=params["translate"],
                scale=params["scale"],
                shear=params["shear"]
            )
            image = transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.2,
                hue=0.08
            )(image)
            if torch.rand(()) < 0.25:
                image = TF.gaussian_blur(image, [3, 3], [0.1, 1.5])
            if torch.rand(()) < 0.03:
                image = TF.rgb_to_grayscale(image, num_output_channels=3)
            tensors.append(self.transform.transforms[-2](image))
            tensors[-1] = self.transform.transforms[-1](tensors[-1])
        return tensors
