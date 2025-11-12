import torch
from pathlib import Path
from ultralytics import YOLO
import yaml
import argparse
import csv
import time
from datetime import datetime


def train_model(model_config, data_yaml, model_name, output_dir):
    """Train YOLO model with custom configuration."""
    device = '0' if torch.cuda.is_available() else 'cpu'
    save_dir = Path(output_dir)

    print(f"\n{'='*80}")
    print(f"TRAINING {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Model config: {model_config}")
    print(f"Device: {device}")
    print(f"Output directory: {save_dir}\n")

    model = YOLO(model_config)

    # Optimized training configuration
    config = {
        "epochs": 150,
        "batch": 32,
        "imgsz": 640,
        "lr0": 0.005 if model_name != 'baseline' else 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 5.0 if model_name != 'baseline' else 3.0,
        "warmup_momentum": 0.8,
        "cos_lr": True,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10.0 if model_name != 'baseline' else 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1 if model_name != 'baseline' else 0.0,
        "patience": 50,
    }

    print(f"Training Parameters:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("="*80 + "\n")

    start_time = time.time()
    results = model.train(
        data=data_yaml,
        project=str(save_dir),
        name="train",
        device=device,
        exist_ok=True,
        verbose=True,
        plots=True,
        **config
    )
    end_time = time.time()
    duration = (end_time - start_time) / 60

    print(f"\n✓ {model_name.upper()} training complete!")
    print(f"Time taken: {duration:.2f} minutes")
    best_weights = save_dir / "train/weights/best.pt"
    print(f"Best weights: {best_weights}\n")

    return best_weights


def validate_model(weights_path, data_yaml, model_name, output_dir):
    """Validate a trained YOLO model and save metrics."""
    device = '0' if torch.cuda.is_available() else 'cpu'
    save_dir = Path(output_dir)

    print(f"\n{'='*80}")
    print(f"VALIDATING {model_name.upper()}")
    print(f"{'='*80}")

    model = YOLO(str(weights_path))

    results = model.val(
        data=data_yaml,
        project=str(save_dir),
        name="validation",
        split="test",
        batch=16,
        device=device,
        plots=True,
        save_json=True,
        exist_ok=True
    )

    # Collect metrics
    metrics = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": model_name,
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "Precision": results.box.mp,
        "Recall": results.box.mr,
        "F1": (2 * results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-6),
        "Inference Speed (ms/img)": results.speed['inference'],
        "VRAM (MB)": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    }

    print(f"\nMetrics for {model_name.upper()}:")
    for k, v in metrics.items():
        if k not in ["Model", "Timestamp"]:
            print(f"  {k}: {v:.4f}")
    print("="*80 + "\n")

    # Save metrics to CSV
    csv_path = Path(output_dir).parent / "training_results.csv"
    file_exists = csv_path.exists()

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

    print(f"✓ Metrics saved to {csv_path}\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Custom YOLO11 Training Script with Timestamped Output")
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--model', type=str, required=True,
                        choices=['baseline', 'simam', 'cbam', 'cbam+simam'],
                        help='Model to train (baseline/simam/cbam/cbam+simam)')
    parser.add_argument('--output', type=str, default='results_custom', help='Base output directory')
    args = parser.parse_args()

    # Define model configs
    model_configs = {
        'baseline': 'yolo11n.yaml',
        'simam': 'ultralytics/cfg/models/11/yolo11-simam.yaml',
        'cbam': 'ultralytics/cfg/models/11/yolo11-cbam.yaml',
        'cbam+simam': 'ultralytics/cfg/models/11/yolo11-cs.yaml',
    }

    # Timestamped subfolder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = Path(args.output) / f"{args.model}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*80}")
    print(f" STARTING TRAINING FOR MODEL: {args.model.upper()} ")
    print(f" Output Directory: {run_dir}")
    print(f"{'#'*80}\n")

    best_weights = train_model(
        model_configs[args.model],
        args.data,
        args.model,
        run_dir
    )

    validate_model(best_weights, args.data, args.model, run_dir)


if __name__ == '__main__':
    main()
