"""
Visualization and Image Annotation Utility Module (FR-3.4, FR-6.3)
Renders color-coded bounding boxes, zone delimiters, empty gap highlights, and telemetry banners.
Supports OpenCV and Pillow rendering engines.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# RGB Color Palette
COLOR_CORRECT_RGB = (46, 204, 113)     # Emerald Green
COLOR_MISPLACED_RGB = (231, 76, 60)    # Bright Red / Crimson
COLOR_GAP_RGB = (243, 156, 18)         # Amber / Orange
COLOR_ZONE_RGB = (120, 130, 140)       # Slate Grey
COLOR_TAG_BG_RGB = (35, 40, 45)        # Dark Slate

def draw_shelf_annotations(image_rgb: np.ndarray,
                           detections: List[Dict[str, Any]],
                           zones: List[Dict[str, Any]],
                           empty_slots: Optional[List[Dict[str, Any]]] = None,
                           show_zones: bool = True,
                           show_gaps: bool = True) -> np.ndarray:
    """
    Produces an annotated shelf image with bounding boxes, zone lines, and gap markers.
    Always returns RGB numpy array.
    """
    pil_img = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(pil_img, "RGBA")
    w, h = pil_img.size

    # 1. Draw Shelf Zones
    if show_zones and zones:
        for zone in zones:
            zx1, zy1 = int(zone['x_min'] * w), int(zone['y_min'] * h)
            zx2, zy2 = int(zone['x_max'] * w), int(zone['y_max'] * h)

            # Zone boundary box
            draw.rectangle([(zx1, zy1), (zx2, zy2)], outline=(80, 90, 100, 180), width=2)

            # Zone label header
            label = f"{zone['zone_name']} | Expected: {zone['expected_category'].upper()}"
            draw.rectangle([(zx1, max(0, zy1 - 22)), (zx1 + 320, zy1)], fill=(40, 44, 52, 220))
            draw.text((zx1 + 6, max(2, zy1 - 18)), label, fill=(240, 240, 240))

    # 2. Draw Empty Space Gaps
    if show_gaps and empty_slots:
        for gap in empty_slots:
            gx1, gy1, gx2, gy2 = gap['bbox']
            # Semi-transparent amber overlay
            draw.rectangle([(gx1, gy1), (gx2, gy2)], fill=(243, 156, 18, 45),
                           outline=COLOR_GAP_RGB, width=2)
            draw.rectangle([(gx1, gy1), (gx1 + 105, gy1 + 18)], fill=(243, 156, 18, 230))
            draw.text((gx1 + 4, gy1 + 2), "[EMPTY GAP]", fill=(255, 255, 255))

    # 3. Draw Product Bounding Boxes
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        is_misplaced = det.get('is_misplaced', False)
        class_name = det.get('class_name', 'item')
        conf = det.get('confidence', 0.0)

        color = COLOR_MISPLACED_RGB if is_misplaced else COLOR_CORRECT_RGB
        width_px = 3 if is_misplaced else 2

        # Bounding box
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=width_px)

        # Label tag
        if is_misplaced:
            tag = f"! MISPLACED: {class_name.upper()} ({int(conf*100)}%)"
            fill_color = COLOR_MISPLACED_RGB
        else:
            tag = f"{class_name.capitalize()} {int(conf*100)}%"
            fill_color = COLOR_CORRECT_RGB

        tag_w = len(tag) * 8 + 10
        tag_y1 = max(0, y1 - 20)
        tag_y2 = y1

        draw.rectangle([(x1, tag_y1), (x1 + tag_w, tag_y2)], fill=fill_color)
        draw.text((x1 + 4, tag_y1 + 2), tag, fill=(255, 255, 255))

    return np.array(pil_img.convert("RGB"))

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Converts BGR to RGB."""
    if HAS_CV2:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img[:, :, ::-1]
