#!/usr/bin/env python3
"""
YOLO11-SimAM Training Pipeline (Minimal)
Author: Your Name
Date: 2025

Clean, focused script for training YOLO11-SimAM on any dataset.
Includes:
- Full SimAM training
- Optional quick setup test
- Validation
"""

import argparse
from pathlib import Path
from datetime import datetime
import torch
import yaml
from ultralytics import YOLO


class SimAMTrainer:
    """Simple training pipeline for YOLO11-SimAM."""

    def __init__(self, data_yaml, project_name="simam_training"):
        self.data_yaml = Path(data_yaml)
        self.project_name = project_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Data YAML not found: {self.data_yaml}")

        with open(self.data_yaml, "r") as f:
            self.data_cfg = yaml.safe_load(f)

        self.device = self._detect_device()
        self._print_summary()

    def _detect_device(self):
        """Detect CUDA or CPU."""
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ Using GPU: {gpu} ({mem:.1f} GB)")
            return "0"
        print("ℹ️  Using CPU (CUDA not available)")
        return "cpu"

    def _print_summary(self):
        print("=" * 80)
        print("YOLO11-SimAM TRAINING PIPELINE")
        print("=" * 80)
        print(f"Dataset YAML : {self.data_yaml}")
        print(f"Classes      : {self.data_cfg['names']}")
        print(f"Num classes  : {self.data_cfg['nc']}")
        print(f"Project      : {self.project_name}")
        print(f"Device       : {self.device}")
        print("=" * 80)

    def train(self, epochs=100, imgsz=640, batch=32, lr0=0.003, patience=15):
        """Train YOLO11-SimAM."""
        print("\n" + "=" * 80)
        print("TRAINING YOLO11-SimAM")
        print("=" * 80)

        model = YOLO("ultralytics/cfg/models/11/yolo11-simam.yaml")

        results = model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            project=self.project_name,
            name=f"simam_{self.timestamp}",
            lr0=lr0,
            patience=patience,
            optimizer="AdamW",
            weight_decay=0.0005,
            momentum=0.9,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            verbose=True,
            plots=True,
            save=True,
        )

        print("\n✓ Training complete!")
        print(f"Results saved to: {self.project_name}/simam_{self.timestamp}")
        return results

    def validate(self, weights_path, split="test"):
        """Validate a trained YOLO11-SimAM model."""
        print("\n" + "=" * 80)
        print(f"VALIDATING MODEL: {Path(weights_path).name}")
        print("=" * 80)

        model = YOLO(weights_path)
        results = model.val(
            data=str(self.data_yaml),
            split=split,
            batch=16,
            imgsz=640,
            save_json=True,
            plots=True,
            verbose=True,
        )
        print("\n✓ Validation complete!")
        return results

    def quick_test(self):
        """One-epoch quick test to verify setup."""
        print("\n" + "=" * 80)
        print("QUICK TEST (YOLO11-SimAM, 1 epoch)")
        print("=" * 80)

        model = YOLO("ultralytics/cfg/models/11/yolo11-simam.yaml")
        results = model.train(
            data=str(self.data_yaml),
            epochs=1,
            imgsz=640,
            batch=8,
            device=self.device,
            project=self.project_name,
            name="quick_test",
            verbose=True,
        )
        print("\n✓ Quick test complete — setup verified.")
        return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11-SimAM")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--lr0", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stop patience")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, 0, 1, ...)")
    parser.add_argument("--mode", choices=["train", "quick", "validate"], default="train")
    parser.add_argument("--weights", type=str, help="Weights path for validation")
    parser.add_argument("--project", default="simam_training", help="Project folder")
    args = parser.parse_args()

    trainer = SimAMTrainer(args.data, args.project)
    if args.device:
        trainer.device = args.device

    if args.mode == "quick":
        trainer.quick_test()
    elif args.mode == "validate":
        if not args.weights:
            print("❌ Please specify --weights for validation.")
            return
        trainer.validate(args.weights)
    else:
        trainer.train(args.epochs, args.imgsz, args.batch, args.lr0, args.patience)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
