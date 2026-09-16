"""
System Configuration Module for Retail Shelf Monitoring
Defines system paths, thresholds, model settings, and default zone mappings.
"""

import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
UPLOADS_DIR = DATA_DIR / 'uploads'
EXPORTS_DIR = DATA_DIR / 'exports'
SAMPLES_DIR = DATA_DIR / 'samples'
MODELS_DIR = BASE_DIR / 'models'

# Ensure directories exist
for folder in [DATA_DIR, UPLOADS_DIR, EXPORTS_DIR, SAMPLES_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Database Configuration
DB_PATH = DATA_DIR / 'shelf_monitoring.db'

# Model & Inference Settings (YOLOv8n)
MODEL_NAME = 'yolov8n.pt'
LOCAL_MODEL_PATH = MODELS_DIR / MODEL_NAME
DEFAULT_CONFIDENCE_THRESHOLD = 0.40
DEFAULT_IOU_THRESHOLD = 0.45
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# Image Constraints
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MAX_IMAGE_SIZE_MB = 10.0

# Shelf Analysis & Thresholds
LOW_STOCK_THRESHOLD_PCT = 35.0      # Zones below 35% occupancy are flagged Low Stock
CRITICAL_STOCK_THRESHOLD_PCT = 15.0 # Zones below 15% occupancy are flagged Empty/Critical
MIN_GAP_CONTOUR_AREA = 1800         # Min area in pixels for empty space contour detection

# Shelf Health Score Status
HEALTH_STATUS_GOOD = 'Good'
HEALTH_STATUS_ATTENTION = 'Needs Attention'
HEALTH_STATUS_CRITICAL = 'Critical'

# Default 3-Tier Shelf Zones (Normalized coordinates [x_min, y_min, x_max, y_max])
DEFAULT_ZONES = [
    {
        'zone_id': 'ZONE_TOP',
        'zone_name': 'Top Shelf (Beverages)',
        'expected_category': 'bottle',
        'x_min': 0.05,
        'y_min': 0.08,
        'x_max': 0.95,
        'y_max': 0.36,
        'capacity': 8
    },
    {
        'zone_id': 'ZONE_MID',
        'zone_name': 'Middle Shelf (Cans & Cups)',
        'expected_category': 'can',
        'x_min': 0.05,
        'y_min': 0.38,
        'x_max': 0.95,
        'y_max': 0.66,
        'capacity': 8
    },
    {
        'zone_id': 'ZONE_BOT',
        'zone_name': 'Bottom Shelf (Snacks & Boxes)',
        'expected_category': 'box',
        'x_min': 0.05,
        'y_min': 0.68,
        'x_max': 0.95,
        'y_max': 0.96,
        'capacity': 8
    }
]

# Mapping common object classes to normalized product categories
CATEGORY_SYNONYMS = {
    'bottle': ['bottle', 'wine glass', 'water bottle', 'soft drink'],
    'can': ['can', 'tin', 'cup', 'soda can'],
    'box': ['box', 'cereal', 'package', 'carton', 'book', 'snack box']
}
