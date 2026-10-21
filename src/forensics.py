import hashlib
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io


def compute_video_hashes(video_path):
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
            "md5": md5.hexdigest()
        }
    except Exception as e:
        return {
            "sha256": "N/A",
            "md5": "N/A"
        }


def compute_fft_spectrum(image):
    """
    Computes the 2D Fast Fourier Transform magnitude spectrum of an image.
    Reveals periodic frequency artifacts typical of GAN/diffusion upsampling.
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("L"))
    else:
        if len(image.shape) == 3:
            img_np = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img_np = image.copy()

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
        cv2.NORM_MINMAX
    )
    norm_spec = np.ascontiguousarray(norm_spec, dtype=np.uint8)
    
    # Apply a high-contrast colormap for forensic visual inspection
    colored_spec = cv2.applyColorMap(
        norm_spec,
        cv2.COLORMAP_MAGMA
    )
    colored_spec = cv2.cvtColor(colored_spec, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(colored_spec)


def compute_ela(image, quality=90, scale=15):
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
    max_diff = float(np.asarray(diff, dtype=np.float32).max())
    if max_diff == 0:
        max_diff = 1
    multiplier = scale
    diff = ImageEnhance.Brightness(diff).enhance(multiplier)
    
    # Apply colormap to highlight artifacts
    diff_np = np.array(diff)
    gray_diff = cv2.cvtColor(diff_np, cv2.COLOR_RGB2GRAY)
    gray_diff = np.ascontiguousarray(gray_diff, dtype=np.uint8)
    ela_colored = cv2.applyColorMap(gray_diff, cv2.COLORMAP_INFERNO)
    ela_colored = cv2.cvtColor(ela_colored, cv2.COLOR_BGR2RGB)

    return Image.fromarray(ela_colored)


def compute_laplacian_sharpness(image):
    """
    Computes Laplacian variance and gradient map to detect boundary softening
    and unnatural facial blur from face-swapping algorithms.
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("L"))
    else:
        if len(image.shape) == 3:
            img_np = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img_np = image.copy()

    laplacian = cv2.Laplacian(img_np, cv2.CV_64F)
    variance = float(laplacian.var())
    
    # Edge map visualization
    laplacian_abs = np.ascontiguousarray(
        np.absolute(laplacian),
        dtype=np.uint8
    )
    laplacian_colored = cv2.applyColorMap(laplacian_abs, cv2.COLORMAP_VIRIDIS)
    laplacian_colored = cv2.cvtColor(laplacian_colored, cv2.COLOR_BGR2RGB)
    
    return variance, Image.fromarray(laplacian_colored)


def compute_temporal_jitter(face_pil_list):
    """
    Computes inter-frame mean squared difference across facial sequence
    to detect temporal flicker or unnatural motion discontinuities.
    """
    if len(face_pil_list) < 2:
        return [0.0]

    diffs = []
    prev_gray = np.array(face_pil_list[0].convert("L"), dtype=float)
    
    for i in range(1, len(face_pil_list)):
        curr_gray = np.array(face_pil_list[i].convert("L"), dtype=float)
        # Normalized mean squared error between consecutive face frames
        mse = float(np.mean((curr_gray - prev_gray) ** 2))
        diffs.append(mse)
        prev_gray = curr_gray
        
    return diffs
