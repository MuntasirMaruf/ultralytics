#!/usr/bin/env python3
"""
Complete Training Pipeline for YOLO11-SimAM on Vehicle Dataset
Author: Your Name
Date: 2025

This script provides a comprehensive training pipeline with:
- Baseline and SimAM model training
- Automatic validation
- Results comparison
- Visualization
"""

import os
import sys
from pathlib import Path
import argparse
import torch
from ultralytics import YOLO
import yaml
from datetime import datetime


class VehicleDetectionTrainer:
    """Complete training pipeline for vehicle detection with SimAM."""
    
    def __init__(self, data_yaml, project_name="vehicle_detection"):
        """
        Initialize the trainer.
        
        Args:
            data_yaml (str): Path to data.yaml file
            project_name (str): Project name for organizing results
        """
        self.data_yaml = data_yaml
        self.project_name = project_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if data.yaml exists
        if not Path(data_yaml).exists():
            raise FileNotFoundError(f"Data YAML not found: {data_yaml}")
        
        # Load data config
        with open(data_yaml, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        # Detect available device
        self.device = self._detect_device()
        
        print("="*80)
        print("VEHICLE DETECTION TRAINING PIPELINE")
        print("="*80)
        print(f"Data YAML: {data_yaml}")
        print(f"Number of classes: {self.data_config['nc']}")
        print(f"Classes: {self.data_config['names']}")
        print(f"Project: {project_name}")
        print(f"Device: {self.device}")
        print("="*80 + "\n")
    
    def _detect_device(self):
        """Detect and return available device."""
        if torch.cuda.is_available():
            device = '0'  # Use first GPU
            print(f"✓ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = 'cpu'
            print("ℹ️  Using CPU (CUDA not available)")
        return device
    
    def train_baseline(self, epochs=100, imgsz=640, batch=32, device=None, lr0=0.003, patience=15, **kwargs):
        """
        Train baseline YOLO11n model with custom configuration.
        
        Args:
            epochs (int): Number of training epochs
            imgsz (int): Image size
            batch (int): Batch size
            device (str): Device ('cpu', '0', '1', etc.)
            lr0 (float): Initial learning rate
            patience (int): Early stopping patience
            **kwargs: Additional training arguments
        """
        print("\n" + "="*80)
        print("TRAINING BASELINE YOLO11n")
        print("="*80)
        print(f"Configuration: epochs={epochs}, batch={batch}, lr0={lr0}, patience={patience}")
        print("="*80)
        
        # Use detected device if not specified
        if device is None:
            device = self.device
        
        print(f"Using device: {device}")
        
        try:
            model = YOLO('yolo11n.yaml')
            
            results = model.train(
                data=self.data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                project=self.project_name,
                name=f'baseline_{self.timestamp}',
                patience=patience,
                lr0=lr0,
                save=True,
                plots=True,
                verbose=True,
                # Additional optimizations
                optimizer='AdamW',  # Better optimizer for convergence
                weight_decay=0.0005,  # Regularization
                momentum=0.9,  # Momentum for SGD-like optimizers
                warmup_epochs=3,  # Learning rate warmup
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                box=7.5,  # Box loss gain
                cls=0.5,  # Class loss gain
                dfl=1.5,  # Distribution Focal Loss gain
                **kwargs
            )
            
            print("\n✓ Baseline training complete!")
            print(f"Results saved to: {self.project_name}/baseline_{self.timestamp}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error during baseline training: {e}")
            # Fallback to CPU if GPU fails
            if device != 'cpu':
                print("🔄 Retrying with CPU...")
                return self.train_baseline(epochs, imgsz, batch, 'cpu', lr0, patience, **kwargs)
            raise
    
    def train_simam(self, epochs=100, imgsz=640, batch=32, device=None, lr0=0.003, patience=15, **kwargs):
        """
        Train YOLO11-SimAM model with custom configuration.
        
        Args:
            epochs (int): Number of training epochs
            imgsz (int): Image size
            batch (int): Batch size
            device (str): Device ('cpu', '0', '1', etc.)
            lr0 (float): Initial learning rate
            patience (int): Early stopping patience
            **kwargs: Additional training arguments
        """
        print("\n" + "="*80)
        print("TRAINING YOLO11-SimAM")
        print("="*80)
        print(f"Configuration: epochs={epochs}, batch={batch}, lr0={lr0}, patience={patience}")
        print("="*80)
        
        # Use detected device if not specified
        if device is None:
            device = self.device
        
        print(f"Using device: {device}")
        
        try:
            model = YOLO('ultralytics/cfg/models/11/yolo11-simam.yaml')
            
            results = model.train(
                data=self.data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                project=self.project_name,
                name=f'simam_{self.timestamp}',
                patience=patience,
                lr0=lr0,
                save=True,
                plots=True,
                verbose=True,
                # Additional optimizations
                optimizer='AdamW',  # Better optimizer for convergence
                weight_decay=0.0005,  # Regularization
                momentum=0.9,  # Momentum for SGD-like optimizers
                warmup_epochs=3,  # Learning rate warmup
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                box=7.5,  # Box loss gain
                cls=0.5,  # Class loss gain
                dfl=1.5,  # Distribution Focal Loss gain
                **kwargs
            )
            
            print("\n✓ SimAM training complete!")
            print(f"Results saved to: {self.project_name}/simam_{self.timestamp}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error during SimAM training: {e}")
            # Fallback to CPU if GPU fails
            if device != 'cpu':
                print("🔄 Retrying with CPU...")
                return self.train_simam(epochs, imgsz, batch, 'cpu', lr0, patience, **kwargs)
            raise
    
    def validate_model(self, weights_path, split='test'):
        """
        Validate a trained model.
        
        Args:
            weights_path (str): Path to model weights
            split (str): Dataset split to validate on ('val' or 'test')
        """
        print("\n" + "="*80)
        print(f"VALIDATING MODEL: {Path(weights_path).name}")
        print("="*80)
        
        model = YOLO(weights_path)
        
        results = model.val(
            data=self.data_yaml,
            split=split,
            batch=16,
            imgsz=640,
            plots=True,
            save_json=True,
            verbose=True
        )
        
        print("\n✓ Validation complete!")
        return results
    
    def compare_results(self, baseline_weights, simam_weights):
        """
        Compare baseline and SimAM model results.
        
        Args:
            baseline_weights (str): Path to baseline weights
            simam_weights (str): Path to SimAM weights
        """
        print("\n" + "="*80)
        print("COMPARING MODELS")
        print("="*80)
        
        # Validate both models
        print("\n1. Validating Baseline...")
        baseline_results = self.validate_model(baseline_weights)
        
        print("\n2. Validating SimAM...")
        simam_results = self.validate_model(simam_weights)
        
        # Extract metrics
        baseline_map50 = baseline_results.box.map50
        baseline_map = baseline_results.box.map
        
        simam_map50 = simam_results.box.map50
        simam_map = simam_results.box.map
        
        # Calculate improvements
        map50_improvement = ((simam_map50 - baseline_map50) / baseline_map50) * 100
        map_improvement = ((simam_map - baseline_map) / baseline_map) * 100
        
        # Print comparison
        print("\n" + "="*80)
        print("RESULTS COMPARISON")
        print("="*80)
        print(f"\n{'Metric':<20} {'Baseline':<15} {'SimAM':<15} {'Improvement':<15}")
        print("-"*80)
        print(f"{'mAP50':<20} {baseline_map50:<15.4f} {simam_map50:<15.4f} {map50_improvement:+.2f}%")
        print(f"{'mAP50-95':<20} {baseline_map:<15.4f} {simam_map:<15.4f} {map_improvement:+.2f}%")
        print("="*80)
        
        # Per-class results
        print("\nPER-CLASS RESULTS (mAP50):")
        print("-"*80)
        for i, class_name in enumerate(self.data_config['names']):
            baseline_class_map = baseline_results.box.maps[i]
            simam_class_map = simam_results.box.maps[i]
            improvement = ((simam_class_map - baseline_class_map) / baseline_class_map) * 100 if baseline_class_map > 0 else 0
            print(f"{class_name:<15} {baseline_class_map:>8.4f} -> {simam_class_map:>8.4f} ({improvement:+6.2f}%)")
        
        print("="*80 + "\n")
        
        return {
            'baseline': baseline_results,
            'simam': simam_results,
            'improvement': {
                'map50': map50_improvement,
                'map': map_improvement
            }
        }
    
    def quick_test(self, epochs=1):
        """
        Quick training test with 1 epoch to verify setup.
        
        Args:
            epochs (int): Number of epochs (default: 1)
        """
        print("\n" + "="*80)
        print("QUICK TRAINING TEST (1 EPOCH)")
        print("="*80)
        
        model = YOLO('ultralytics/cfg/models/11/yolo11-simam.yaml')
        
        results = model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=640,
            batch=8,
            device=self.device,
            project=self.project_name,
            name='quick_test',
            verbose=True
        )
        
        print("\n✓ Quick test complete! Setup is working correctly.")
        print("You can now run full training.")
        
        return results


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train YOLO11-SimAM on Vehicle Dataset')
    parser.add_argument('--data', type=str, default='data.yaml', 
                       help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=32, 
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, 
                       help='Image size')
    parser.add_argument('--lr0', type=float, default=0.003,
                       help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, default=None, 
                       help='Device (cpu, 0, 1, 2). Auto-detect if not specified')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['baseline', 'simam', 'both', 'compare', 'quick'],
                       help='Training mode')
    parser.add_argument('--baseline-weights', type=str, default=None,
                       help='Path to baseline weights for comparison')
    parser.add_argument('--simam-weights', type=str, default=None,
                       help='Path to SimAM weights for comparison')
    parser.add_argument('--project', type=str, default='vehicle_detection',
                       help='Project name for results')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = VehicleDetectionTrainer(args.data, args.project)
    
    # Override device if specified
    if args.device is not None:
        trainer.device = args.device
    
    # Quick test mode
    if args.mode == 'quick':
        trainer.quick_test(epochs=1)
        return
    
    # Training modes
    if args.mode == 'baseline':
        trainer.train_baseline(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            lr0=args.lr0,
            patience=args.patience
        )
    
    elif args.mode == 'simam':
        trainer.train_simam(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            lr0=args.lr0,
            patience=args.patience
        )
    
    elif args.mode == 'both':
        # Train both models sequentially
        print("\n🚀 Training both models sequentially...\n")
        
        # Train baseline first
        baseline_results = trainer.train_baseline(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            lr0=args.lr0,
            patience=args.patience
        )
        
        # Then train SimAM
        simam_results = trainer.train_simam(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            lr0=args.lr0,
            patience=args.patience
        )
        
        print("\n✓ Both models trained successfully!")
        print(f"\nResults saved in: {args.project}/")
        print("\nTo compare results, run:")
        print(f"python train_simam.py --mode compare --baseline-weights <path> --simam-weights <path>")
    
    elif args.mode == 'compare':
        if not args.baseline_weights or not args.simam_weights:
            print("Error: Please provide both --baseline-weights and --simam-weights for comparison")
            return
        
        trainer.compare_results(args.baseline_weights, args.simam_weights)
    
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()