"""
Model Training & Fine-Tuning Script (TRD Section 7 & 13)
Provides transfer learning pipeline for fine-tuning YOLOv8n on retail shelf datasets
such as SKU-110K, Grocery Store Dataset, or custom labeled Roboflow datasets.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n on Retail Shelf Dataset")
    parser.add_argument('--data', type=str, default='dataset.yaml', help='Path to dataset.yaml config')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Pretrained base model (yolov8n.pt / yolov8s.pt)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image resolution')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='', help='Device: 0 for GPU, or cpu')
    parser.add_argument('--project', type=str, default=str(MODELS_DIR), help='Save directory')
    parser.add_argument('--name', type=str, default='shelf_yolov8n_exp', help='Experiment name')
    return parser.parse_args()

def create_sample_dataset_yaml(output_path: Path):
    """Creates a template dataset.yaml file for custom training."""
    template = """# Retail Shelf Monitoring Dataset Configuration
# Structure your data into train/val/test splits:
path: ./dataset # root dataset directory
train: images/train
val: images/val
test: images/test

# Retail classes
names:
  0: bottle
  1: can
  2: box
  3: snack
  4: package
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)
    print(f"[Train] Sample dataset YAML created at: {output_path}")

def train():
    args = parse_args()

    print("=========================================================")
    print("   RETAIL SHELF MONITORING - YOLOv8n FINE-TUNING PIPELINE")
    print("=========================================================")
    print(f"Base Model:    {args.model}")
    print(f"Dataset YAML:  {args.data}")
    print(f"Epochs:        {args.epochs}")
    print(f"Image Size:    {args.imgsz}")
    print(f"Batch Size:    {args.batch}")
    print(f"Destination:   {args.project}/{args.name}")
    print("=========================================================\n")

    yaml_path = Path(args.data)
    if not yaml_path.exists():
        print(f"[Train] Dataset config '{args.data}' not found. Generating default template...")
        create_sample_dataset_yaml(yaml_path)
        print("[Train] Please populate your annotated images and labels in ./dataset and re-run training.")
        return

    try:
        from ultralytics import YOLO
        import torch

        device = args.device
        if not device:
            device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"[Train] Executing training on target hardware device: {device}")

        # Load base model
        model = YOLO(args.model)

        # Train model with transfer learning
        results = model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=args.project,
            name=args.name,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            augment=True,
            val=True
        )

        print("\n[Train] Training complete!")
        print(f"[Train] Best model weights saved to: {args.project}/{args.name}/weights/best.pt")

        # Validate
        print("\n[Train] Running final validation...")
        metrics = model.val()
        print(f"[Train] Validation mAP@0.5: {metrics.box.map50:.4f}")
        print(f"[Train] Validation mAP@0.5:0.95: {metrics.box.map:.4f}")

    except ImportError:
        print("[Train] Error: 'ultralytics' or 'torch' is not installed in the current environment.")
        print("[Train] To install requirements: pip install ultralytics torch")
        print("[Train] On Google Colab, GPU acceleration is enabled automatically.")

if __name__ == '__main__':
    train()
