"""
Image Preprocessing Module for Retail Shelf Monitoring (FR-2)
Handles file validation, dimensions normalization, letterbox resizing, and contrast enhancement.
Supports both OpenCV and Pillow/NumPy backends seamlessly.
"""

import os
from pathlib import Path
from typing import Tuple, Union, Optional, Any
import numpy as np
from PIL import Image, ImageEnhance
import sys

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import ALLOWED_EXTENSIONS, MAX_IMAGE_SIZE_MB, INPUT_WIDTH, INPUT_HEIGHT

class ImagePreprocessor:
    """Provides standard preprocessing pipelines for retail shelf images."""

    @staticmethod
    def validate_image_file(file_path_or_buffer: Union[str, Path, bytes]) -> Tuple[bool, str]:
        """Validates image extension, readability, and file size constraints (FR-1.2)."""
        if isinstance(file_path_or_buffer, (str, Path)):
            p = Path(file_path_or_buffer)
            if not p.exists():
                return False, f"File does not exist: {p}"
            if p.suffix.lower() not in ALLOWED_EXTENSIONS:
                return False, f"Unsupported format: {p.suffix}. Allowed: {ALLOWED_EXTENSIONS}"
            size_mb = p.stat().st_size / (1024 * 1024)
            if size_mb > MAX_IMAGE_SIZE_MB:
                return False, f"File size ({size_mb:.1f}MB) exceeds limit of {MAX_IMAGE_SIZE_MB}MB"
        return True, "Valid image"

    @staticmethod
    def load_image(image_source: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Loads an image from various input types into a standard RGB numpy array (H, W, 3).
        """
        if isinstance(image_source, Image.Image):
            return np.array(image_source.convert("RGB"))

        if isinstance(image_source, np.ndarray):
            if len(image_source.shape) == 2:
                return np.stack([image_source] * 3, axis=-1)
            return image_source.copy()

        if isinstance(image_source, (str, Path)):
            if HAS_CV2:
                bgr = cv2.imread(str(image_source))
                if bgr is not None:
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.open(str(image_source)).convert("RGB")
            return np.array(pil_img)

        if isinstance(image_source, (bytes, bytearray)):
            import io
            pil_img = Image.open(io.BytesIO(image_source)).convert("RGB")
            return np.array(pil_img)

        raise TypeError(f"Unsupported image input type: {type(image_source)}")

    @staticmethod
    def letterbox_resize(image: np.ndarray, target_shape: Tuple[int, int] = (INPUT_WIDTH, INPUT_HEIGHT)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resizes image preserving aspect ratio with padding (114, 114, 114).
        """
        target_w, target_h = target_shape
        h, w = image.shape[:2]
        scale = min(target_w / w, target_h / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

        if HAS_CV2:
            resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        else:
            pil_img = Image.fromarray(image)
            resized = np.array(pil_img.resize((nw, nh), Image.Resampling.BILINEAR))

        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        top = (target_h - nh) // 2
        left = (target_w - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized

        return canvas, scale, (left, top)

    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Applies adaptive contrast enhancement."""
        if HAS_CV2:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:
            pil_img = Image.fromarray(image)
            enhancer = ImageEnhance.Contrast(pil_img)
            return np.array(enhancer.enhance(1.2))

    @classmethod
    def preprocess_pipeline(cls, image_source: Any, enhance: bool = False) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
        """Full preprocessing pipeline returning original_rgb, model_input, scale, padding."""
        original = cls.load_image(image_source)
        working_img = cls.enhance_contrast(original) if enhance else original
        model_input, scale, padding = cls.letterbox_resize(working_img)
        return original, model_input, scale, padding
