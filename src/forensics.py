from __future__ import annotations

import hashlib
import io
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


def compute_video_hashes(video_path: str) -> dict[str, str]:
    """Computes SHA-256 and MD5 hashes of the video file for forensic integrity."""
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    try:
        with open(video_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
                md5.update(chunk)
        return {
            "sha256": sha256.hexdigest(),
            "md5": md5.hexdigest(),
        }
    except Exception:
        return {
            "sha256": "N/A",
            "md5": "N/A",
        }


def compute_fft_spectrum(image: Union[Image.Image, np.ndarray]) -> Image.Image:
    """
    Computes the 2D Fast Fourier Transform magnitude spectrum of an image.
    Reveals periodic frequency artifacts typical of GAN/diffusion upsampling.
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("L"))
    else:
        img_arr = np.asarray(image)
        if img_arr.ndim == 3:
            img_np = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        else:
            img_np = img_arr.copy()

    f = np.fft.fft2(img_np.astype(float))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)

    # Normalize to 0-255 for display
    norm_spec = np.empty_like(magnitude_spectrum, dtype=np.float64)
    cv2.normalize(
        magnitude_spectrum,
        norm_spec,
        0,
        255,
        cv2.NORM_MINMAX,
    )
    norm_spec = np.ascontiguousarray(norm_spec, dtype=np.uint8)

    # Apply a high-contrast colormap for forensic visual inspection
    colored_spec = cv2.applyColorMap(
        norm_spec,
        int(cv2.COLORMAP_MAGMA),
    )
    colored_spec = cv2.cvtColor(colored_spec, cv2.COLOR_BGR2RGB)

    return Image.fromarray(colored_spec)


def compute_ela(image: Union[Image.Image, np.ndarray], quality: int = 90, scale: int = 15) -> Image.Image:
    """
    Computes Error Level Analysis (ELA) of an image.
    Resaves as JPEG at a given quality and measures the difference.
    Different compression histories reveal manipulated and spliced areas.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Convert to RGB if needed
    image = image.convert("RGB")

    # Save to memory buffer at specified quality
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)

    # Reload compressed image
    compressed = Image.open(buffer)

    # Calculate difference
    diff = ImageChops.difference(image, compressed)

    # Enhance difference brightness for visibility
    diff = ImageEnhance.Brightness(diff).enhance(scale)

    # Apply colormap to highlight artifacts
    diff_np = np.array(diff)
    gray_diff = cv2.cvtColor(diff_np, cv2.COLOR_RGB2GRAY)
    gray_diff = np.ascontiguousarray(gray_diff, dtype=np.uint8)
    ela_colored = cv2.applyColorMap(gray_diff, int(cv2.COLORMAP_INFERNO))
    ela_colored = cv2.cvtColor(ela_colored, cv2.COLOR_BGR2RGB)

    return Image.fromarray(ela_colored)


def compute_laplacian_sharpness(image: Union[Image.Image, np.ndarray]) -> tuple[float, Image.Image]:
    """
    Computes Laplacian variance and gradient map to detect boundary softening
    and unnatural facial blur from face-swapping algorithms.
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("L"))
    else:
        img_arr = np.asarray(image)
        if img_arr.ndim == 3:
            img_np = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        else:
            img_np = img_arr.copy()

    laplacian = cv2.Laplacian(img_np, cv2.CV_64F)
    variance = float(laplacian.var())

    # Edge map visualization
    laplacian_abs = np.ascontiguousarray(
        np.absolute(laplacian),
        dtype=np.uint8,
    )
    laplacian_colored = cv2.applyColorMap(laplacian_abs, int(cv2.COLORMAP_VIRIDIS))
    laplacian_colored = cv2.cvtColor(laplacian_colored, cv2.COLOR_BGR2RGB)

    return variance, Image.fromarray(laplacian_colored)


def compute_temporal_jitter(face_pil_list: list[Image.Image]) -> list[float]:
    """
    Computes inter-frame mean squared difference across facial sequence
    to detect temporal flicker or unnatural motion discontinuities.
    """
    if len(face_pil_list) < 2:
        return [0.0]

    diffs: list[float] = []
    prev_gray = np.array(face_pil_list[0].convert("L"), dtype=float)

    for i in range(1, len(face_pil_list)):
        curr_gray = np.array(face_pil_list[i].convert("L"), dtype=float)
        # Normalized mean squared error between consecutive face frames
        mse = float(np.mean((curr_gray - prev_gray) ** 2))
        diffs.append(mse)
        prev_gray = curr_gray

    return diffs


def extract_forensic_descriptors(face_imgs_np: np.ndarray) -> np.ndarray:
    """
    Extract multi-domain forensic descriptors from (N, H, W, 3) face sequence:
    1. FFT Frequency Spectrum: 4 radial energy bands + high-to-low power ratio.
    2. Color Chrominance Boundary Discrepancy: Inner face vs outer margin Delta-Cb and Delta-Cr.
    3. Laplacian Texture Sharpness: Inner face vs blending boundary edge variance ratio.
    4. Error Level Analysis (ELA): JPEG compression residual distribution (mean, std, max).
    5. Temporal Dynamics: Inter-frame motion jitter statistics (mean, std, max).

    Returns:
        (33,) float32 multi-domain forensic feature vector.
    """
    if face_imgs_np.ndim == 3:
        face_imgs_np = np.expand_dims(face_imgs_np, axis=0)

    N, H, W, _ = face_imgs_np.shape
    fft_feats: list[list[float]] = []
    color_feats: list[list[float]] = []
    laplacian_feats: list[list[float]] = []
    ela_feats: list[list[float]] = []

    center_y1, center_y2 = int(0.2 * H), int(0.8 * H)
    center_x1, center_x2 = int(0.2 * W), int(0.8 * W)
    outer_mask = np.ones((H, W), dtype=bool)
    outer_mask[center_y1:center_y2, center_x1:center_x2] = False

    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2) + 1e-6
    r_norm = r / max_r

    for i in range(N):
        img_rgb = face_imgs_np[i]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 1. FFT Radial Spectrum
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.log(np.abs(fshift) + 1e-5)
        b1 = float(mag[r_norm < 0.25].mean())
        b2 = float(mag[(r_norm >= 0.25) & (r_norm < 0.50)].mean())
        b3 = float(mag[(r_norm >= 0.50) & (r_norm < 0.75)].mean())
        b4 = float(mag[r_norm >= 0.75].mean())
        high_low_ratio = float(b4 / (b1 + 1e-5))
        fft_feats.append([b1, b2, b3, b4, high_low_ratio])

        # 2. Color Chrominance Boundary Discrepancy (YCbCr)
        ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        cb = ycbcr[:, :, 1].astype(np.float32)
        cr = ycbcr[:, :, 2].astype(np.float32)
        inner_cb = float(cb[center_y1:center_y2, center_x1:center_x2].mean())
        inner_cr = float(cr[center_y1:center_y2, center_x1:center_x2].mean())
        outer_cb = float(cb[outer_mask].mean())
        outer_cr = float(cr[outer_mask].mean())
        delta_cb = abs(inner_cb - outer_cb)
        delta_cr = abs(inner_cr - outer_cr)
        color_feats.append([inner_cb, inner_cr, delta_cb, delta_cr])

        # 3. Laplacian Texture Sharpness
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap_inner = float(lap[center_y1:center_y2, center_x1:center_x2].var())
        lap_outer = float(lap[outer_mask].var())
        lap_ratio = float((lap_inner + 1e-5) / (lap_outer + 1e-5))
        laplacian_feats.append([lap_inner, lap_outer, lap_ratio])

        # 4. In-Memory ELA (JPEG Q=85 residual)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            decomp = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            ela_diff = np.abs(img_rgb.astype(np.float32) - decomp.astype(np.float32))
            ela_feats.append([float(ela_diff.mean()), float(ela_diff.std()), float(ela_diff.max())])
        else:
            ela_feats.append([0.0, 0.0, 0.0])

    fft_arr = np.array(fft_feats, dtype=np.float32)
    color_arr = np.array(color_feats, dtype=np.float32)
    lap_arr = np.array(laplacian_feats, dtype=np.float32)
    ela_arr = np.array(ela_feats, dtype=np.float32)

    # 5. Temporal Motion Jitter
    if N > 1:
        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in face_imgs_np]
        diffs = [float(np.mean((grays[k] - grays[k - 1]) ** 2)) for k in range(1, N)]
        temp_mean = float(np.mean(diffs))
        temp_std = float(np.std(diffs))
        temp_max = float(np.max(diffs))
    else:
        temp_mean, temp_std, temp_max = 0.0, 0.0, 0.0

    return np.concatenate([
        fft_arr.mean(axis=0), fft_arr.std(axis=0),
        color_arr.mean(axis=0), color_arr.std(axis=0),
        lap_arr.mean(axis=0), lap_arr.std(axis=0),
        ela_arr.mean(axis=0), ela_arr.std(axis=0),
        np.array([temp_mean, temp_std, temp_max], dtype=np.float32),
    ]).astype(np.float32)
