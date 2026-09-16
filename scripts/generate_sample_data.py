"""
Sample Data Generator for Retail Shelf Monitoring Demo
Generates synthetic shelf scenario images (Normal, Low-Stock, Misplaced Items)
using Pillow and OpenCV compatibility, and seeds the SQLite database.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys

# Try importing cv2 if available
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SAMPLES_DIR, DB_PATH
from database.db_manager import DatabaseManager

def create_shelf_canvas(width: int = 800, height: int = 800) -> Image.Image:
    """Generates a realistic supermarket shelf rack background."""
    img = Image.new("RGB", (width, height), (220, 225, 230))
    draw = ImageDraw.Draw(img)

    # Gradient background
    for y in range(height):
        ratio = y / height
        r = int(225 - ratio * 35)
        g = int(230 - ratio * 30)
        b = int(235 - ratio * 25)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Shelf tier planks (3 tiers)
    shelf_heights = [int(height * 0.36), int(height * 0.66), int(height * 0.96)]
    for sh in shelf_heights:
        # Metallic plank
        draw.rectangle([(20, sh - 14), (width - 20, sh)], fill=(130, 135, 140))
        # 3D Highlight & Shadow
        draw.line([(20, sh - 14), (width - 20, sh - 14)], fill=(200, 205, 210), width=2)
        draw.line([(20, sh), (width - 20, sh)], fill=(70, 75, 80), width=3)
        # Red price strip
        draw.rectangle([(30, sh - 11), (width - 30, sh - 2)], fill=(180, 40, 40))
        # Price tags
        for tx in range(60, width - 60, 90):
            draw.rectangle([(tx, sh - 10), (tx + 35, sh - 3)], fill=(255, 255, 255))

    return img

def draw_bottle(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, label: str = "Juice"):
    """Draws a bottle product."""
    cap_w, cap_h = int(w * 0.4), int(h * 0.12)
    neck_w, neck_h = int(w * 0.5), int(h * 0.20)
    body_y = y + neck_h

    # Bottle body
    draw.rectangle([(x, body_y), (x + w, y + h)], fill=(40, 140, 60), outline=(25, 100, 40), width=2)
    # Highlight
    draw.line([(x + 4, body_y + 4), (x + 4, y + h - 4)], fill=(110, 210, 130), width=2)
    # Neck
    nx = x + (w - neck_w) // 2
    draw.rectangle([(nx, y + cap_h), (nx + neck_w, body_y)], fill=(35, 130, 55))
    # Cap
    cx = x + (w - cap_w) // 2
    draw.rectangle([(cx, y), (cx + cap_w, y + cap_h)], fill=(200, 20, 20))
    # Label
    draw.rectangle([(x + 4, body_y + int(h * 0.2)), (x + w - 4, body_y + int(h * 0.6))], fill=(245, 245, 245))
    draw.text((x + 6, body_y + int(h * 0.25)), label[:4], fill=(10, 10, 10))

def draw_can(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, label: str = "Soda"):
    """Draws a beverage can."""
    # Body
    draw.rectangle([(x, y + 4), (x + w, y + h - 4)], fill=(180, 50, 40), outline=(130, 30, 25), width=2)
    # Top & bottom rims
    draw.ellipse([(x, y), (x + w, y + 10)], fill=(210, 210, 210))
    draw.ellipse([(x, y + h - 10), (x + w, y + h)], fill=(140, 140, 140))
    # Shine
    draw.line([(x + 6, y + 10), (x + 6, y + h - 10)], fill=(230, 130, 120), width=2)
    # Text
    draw.text((x + 6, y + h // 2 - 8), label[:4], fill=(255, 255, 255))

def draw_box(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, label: str = "Cereal"):
    """Draws a cereal/snack box."""
    draw.rectangle([(x, y), (x + w, y + h)], fill=(220, 130, 30), outline=(140, 75, 15), width=2)
    draw.rectangle([(x + 6, y + 10), (x + w - 6, y + int(h * 0.55))], fill=(250, 230, 160))
    draw.text((x + 8, y + h - 22), label[:6], fill=(255, 255, 255))

def generate_scenario_normal(output_path: Path):
    """Normal Condition: Perfectly stocked."""
    img = create_shelf_canvas(800, 800)
    draw = ImageDraw.Draw(img)

    # Top shelf (Zone A): 7 Bottles
    for i in range(7):
        draw_bottle(draw, 55 + i * 102, 110, 52, 165, label="Juice")

    # Middle shelf (Zone B): 7 Cans
    for i in range(7):
        draw_can(draw, 60 + i * 100, 360, 56, 155, label="Soda")

    # Bottom shelf (Zone C): 6 Boxes
    for i in range(6):
        draw_box(draw, 65 + i * 115, 600, 75, 155, label="Cereal")

    img.save(str(output_path))
    print(f"Generated: {output_path.name}")

def generate_scenario_low_stock(output_path: Path):
    """Low Stock Condition: Gaps & missing stock."""
    img = create_shelf_canvas(800, 800)
    draw = ImageDraw.Draw(img)

    # Top shelf: 2 bottles only (Low Stock)
    draw_bottle(draw, 60, 110, 52, 165, label="Juice")
    draw_bottle(draw, 165, 110, 52, 165, label="Juice")

    # Middle shelf: 3 cans with wide gap
    draw_can(draw, 65, 360, 56, 155, label="Soda")
    draw_can(draw, 165, 360, 56, 155, label="Soda")
    draw_can(draw, 660, 360, 56, 155, label="Soda")

    # Bottom shelf: 1 box (Critical)
    draw_box(draw, 660, 600, 75, 155, label="Cereal")

    img.save(str(output_path))
    print(f"Generated: {output_path.name}")

def generate_scenario_misplaced(output_path: Path):
    """Misplaced Condition: Item in wrong shelf tier."""
    img = create_shelf_canvas(800, 800)
    draw = ImageDraw.Draw(img)

    # Top shelf (Beverages/Bottles): Contains a MISPLACED SODA CAN
    for i in range(4):
        draw_bottle(draw, 55 + i * 105, 110, 52, 165, label="Juice")
    draw_can(draw, 55 + 4 * 105, 120, 56, 155, label="Soda")
    draw_bottle(draw, 55 + 5 * 105, 110, 52, 165, label="Juice")

    # Middle shelf: Cans
    for i in range(6):
        draw_can(draw, 65 + i * 105, 360, 56, 155, label="Soda")

    # Bottom shelf: Boxes + MISPLACED BOTTLE
    for i in range(4):
        draw_box(draw, 65 + i * 115, 600, 75, 155, label="Cereal")
    draw_bottle(draw, 65 + 4 * 115, 595, 52, 165, label="Juice")

    img.save(str(output_path))
    print(f"Generated: {output_path.name}")

def main():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print("--- Generating Retail Shelf Demonstration Images ---")
    generate_scenario_normal(SAMPLES_DIR / "shelf_normal.png")
    generate_scenario_low_stock(SAMPLES_DIR / "shelf_low_stock.png")
    generate_scenario_misplaced(SAMPLES_DIR / "shelf_misplaced.png")

    print("\n--- Initializing & Verifying Database ---")
    db = DatabaseManager()
    zones = db.get_all_zones()
    print(f"Database initialized at: {DB_PATH}")
    print(f"Total active shelf zones configured: {len(zones)}")
    for z in zones:
        print(f" - [{z['zone_id']}] {z['zone_name']}: Target = {z['expected_category']}, Capacity = {z['capacity']}")

    print("\nSample generation complete! Ready for live dashboard execution.")

if __name__ == '__main__':
    main()
