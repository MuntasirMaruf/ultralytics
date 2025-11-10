#!/usr/bin/env python3
"""
Training Pipeline for YOLO11n, YOLO11-SimAM, and YOLO11-CBAM
Simplified version with fixed configuration
"""

import os
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
import yaml


class MultiModelTrainer:
    """Train and compare YOLO11n, SimAM, and CBAM models."""
    
    def __init__(self, data_yaml, output_dir="results"):
        self.data_yaml = data_yaml
        self.output_dir = Path(output_dir)
        
        # Fixed training configuration
        self.config = {
            "epochs": 100,
            "batch": 32,
            "imgsz": 640,
            "lr0": 0.003,
            "patience": 15,
            "optimizer": "AdamW",
            "weight_decay": 0.0005,
            "warmup_epochs": 3,
        }
        
        # Detect device
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        
        # Load data config
        with open(data_yaml, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        print("="*80)
        print("MULTI-MODEL TRAINING PIPELINE")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Classes: {self.data_config['nc']}")
        print(f"Configuration: {self.config}")
        print("="*80 + "\n")
    
    def train_model(self, model_config, model_name):
        """
        Train a single model.
        
        Args:
            model_config (str): Path to model YAML or model name
            model_name (str): Name for saving results (baseline/simam/cbam)
        """
        print(f"\n{'='*80}")
        print(f"TRAINING {model_name.upper()}")
        print(f"{'='*80}")
        
        try:
            model = YOLO(model_config)
            save_dir = self.output_dir / model_name
            
            results = model.train(
                data=self.data_yaml,
                project=str(save_dir),
                name="train",
                device=self.device,
                exist_ok=True,
                **self.config
            )
            
            print(f"\n✓ {model_name.upper()} training complete!")
            print(f"Results saved to: {save_dir}/train")
            
            return results, save_dir / "train" / "weights" / "best.pt"
            
        except Exception as e:
            print(f"✗ Error training {model_name}: {e}")
            raise
    
    def validate_model(self, weights_path, model_name):
        """
        Validate a trained model.
        
        Args:
            weights_path (Path): Path to model weights
            model_name (str): Model name for saving results
        """
        print(f"\n{'='*80}")
        print(f"VALIDATING {model_name.upper()}")
        print(f"{'='*80}")
        
        model = YOLO(str(weights_path))
        save_dir = self.output_dir / model_name
        
        results = model.val(
            data=self.data_yaml,
            project=str(save_dir),
            name="validation",
            split="test",
            batch=16,
            device=self.device,
            plots=True,
            save_json=True,
            exist_ok=True
        )
        
        print(f"\n✓ {model_name.upper()} validation complete!")
        print(f"Results saved to: {save_dir}/validation")
        
        return results
    
    def compare_models(self, results_dict):
        """
        Compare results from all models.
        
        Args:
            results_dict (dict): Dictionary with model names as keys and results as values
        """
        print(f"\n{'='*80}")
        print("RESULTS COMPARISON")
        print(f"{'='*80}\n")
        
        # Print header
        print(f"{'Model':<15} {'mAP50':<12} {'mAP50-95':<12} {'Precision':<12} {'Recall':<12}")
        print("-"*80)
        
        # Store baseline for comparison
        baseline_map50 = results_dict['baseline'].box.map50
        baseline_map = results_dict['baseline'].box.map
        
        # Print results for each model
        for name, results in results_dict.items():
            map50 = results.box.map50
            map_val = results.box.map
            precision = results.box.mp
            recall = results.box.mr
            
            print(f"{name.upper():<15} {map50:<12.4f} {map_val:<12.4f} {precision:<12.4f} {recall:<12.4f}")
        
        print("-"*80)
        
        # Print improvements over baseline
        print("\nIMPROVEMENTS OVER BASELINE:")
        print("-"*80)
        for name, results in results_dict.items():
            if name == 'baseline':
                continue
            
            map50 = results.box.map50
            map_val = results.box.map
            
            map50_imp = ((map50 - baseline_map50) / baseline_map50) * 100
            map_imp = ((map_val - baseline_map) / baseline_map) * 100
            
            print(f"{name.upper():<15} mAP50: {map50_imp:+.2f}%  |  mAP50-95: {map_imp:+.2f}%")
        
        print("="*80 + "\n")
        
        # Per-class comparison
        print("PER-CLASS mAP50 COMPARISON:")
        print("-"*80)
        print(f"{'Class':<15} {'Baseline':<12} {'SimAM':<12} {'CBAM':<12}")
        print("-"*80)
        
        for i, class_name in enumerate(self.data_config['names']):
            baseline_val = results_dict['baseline'].box.maps[i]
            simam_val = results_dict['simam'].box.maps[i]
            cbam_val = results_dict['cbam'].box.maps[i]
            
            print(f"{class_name:<15} {baseline_val:<12.4f} {simam_val:<12.4f} {cbam_val:<12.4f}")
        
        print("="*80 + "\n")
    
    def run_all(self):
        """Train and validate all models, then compare results."""
        models = {
            'baseline': 'yolo11n.yaml',
            'simam': 'ultralytics/cfg/models/11/yolo11-simam.yaml',
            'cbam': 'ultralytics/cfg/models/11/yolo11-cbam.yaml'
        }
        
        weights = {}
        results = {}
        
        # Train all models
        for name, config in models.items():
            _, weights[name] = self.train_model(config, name)
        
        # Validate all models
        for name, weight_path in weights.items():
            results[name] = self.validate_model(weight_path, name)
        
        # Compare results
        self.compare_models(results)
        
        print("\n" + "="*80)
        print("ALL MODELS TRAINED AND VALIDATED SUCCESSFULLY")
        print("="*80)
        print(f"\nResults structure:")
        print(f"  {self.output_dir}/")
        print(f"    ├── baseline/")
        print(f"    │   ├── train/")
        print(f"    │   └── validation/")
        print(f"    ├── simam/")
        print(f"    │   ├── train/")
        print(f"    │   └── validation/")
        print(f"    └── cbam/")
        print(f"        ├── train/")
        print(f"        └── validation/")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train YOLO11n, SimAM, and CBAM')
    parser.add_argument('--data', type=str, default='data.yaml',
                       help='Path to data.yaml')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'baseline', 'simam', 'cbam'],
                       help='Which model(s) to train')
    
    args = parser.parse_args()
    
    trainer = MultiModelTrainer(args.data, args.output)
    
    if args.model == 'all':
        trainer.run_all()
    else:
        # Train single model
        models = {
            'baseline': 'yolo11n.yaml',
            'simam': 'ultralytics/cfg/models/11/yolo11-simam.yaml',
            'cbam': 'ultralytics/cfg/models/11/yolo11-cbam.yaml'
        }
        
        _, weights = trainer.train_model(models[args.model], args.model)
        trainer.validate_model(weights, args.model)


if __name__ == '__main__':
    main()