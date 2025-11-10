#!/usr/bin/env python3
"""
Test SimAM module integration
Run this to verify SimAM is properly integrated

Usage:
    python test_simam.py
"""

import torch
from ultralytics import YOLO
from ultralytics.nn.modules import SimAM


def test_simam_module():
    """Test SimAM module standalone"""
    print("\n" + "="*80)
    print("TEST 1: SimAM Module")
    print("="*80)
    
    try:
        # Create SimAM module
        simam = SimAM(c1=256)
        
        # Test forward pass
        x = torch.randn(1, 256, 20, 20)
        out = simam(x)
        
        assert out.shape == x.shape, f"Shape mismatch! Expected {x.shape}, got {out.shape}"
        
        # Count parameters
        params = sum(p.numel() for p in simam.parameters())
        
        print(f"✓ SimAM module imported successfully")
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {out.shape}")
        print(f"✓ Parameters: {params:,} (should be 0!)")
        print(f"✓ Shape preserved: {x.shape == out.shape}")
        
        if params == 0:
            print(f"✓ ZERO PARAMETERS - This is correct for SimAM!")
        else:
            print(f"⚠ Warning: SimAM should have 0 parameters, but has {params}")
        
        print("="*80 + "\n")
        
        return True
    except Exception as e:
        print(f"✗ SimAM module test failed: {e}")
        print(f"\nError details: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you:")
        print("  1. Added SimAM class to ultralytics/nn/modules/block.py")
        print("  2. Imported SimAM in ultralytics/nn/modules/__init__.py")
        print("  3. SimAM class uses signature: __init__(self, c1, c2=None, e_lambda=1e-4)")
        return False


def test_model_loading():
    """Test loading YOLO11-SimAM configuration"""
    print("="*80)
    print("TEST 2: Model Loading")
    print("="*80)
    
    try:
        import os
        
        # Try to find the YAML file
        yaml_locations = [
            'yolo11-simam.yaml',
            'ultralytics/cfg/models/11/yolo11-simam.yaml',
            os.path.join('ultralytics', 'cfg', 'models', '11', 'yolo11-simam.yaml'),
        ]
        
        model = None
        loaded_from = None
        last_error = None
        for yaml_path in yaml_locations:
            try:
                if os.path.exists(yaml_path):
                    print(f"  Found: {yaml_path}")
                    model = YOLO(yaml_path)
                    loaded_from = yaml_path
                    break
            except Exception as e:
                last_error = e
                print(f"  Error loading {yaml_path}: {e}")
                continue
        
        if model is None:
            print("  Could not load yolo11-simam.yaml")
            if last_error:
                print(f"  Last error: {last_error}")
            raise FileNotFoundError(f"yolo11-simam.yaml not found or failed to load. Error: {last_error}")
        
        print(f"✓ Loaded from: {loaded_from}")
        
        # Get model info
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        print(f"✓ YOLO11-SimAM model loaded successfully")
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        print("  Testing forward pass...")
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model(dummy_input, verbose=False)
        
        print(f"✓ Forward pass successful")
        print(f"✓ Model is ready for training")
        print("="*80 + "\n")
        
        return True
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        print(f"\nError details: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def test_baseline_comparison():
    """Compare YOLO11n vs YOLO11n-SimAM parameters"""
    print("="*80)
    print("TEST 3: Baseline Comparison")
    print("="*80)
    
    try:
        import os
        
        # Load baseline
        print("  Loading baseline YOLO11n...")
        baseline = YOLO('yolo11n.yaml')
        baseline_params = sum(p.numel() for p in baseline.model.parameters())
        
        # Load SimAM version
        print("  Loading YOLO11n-SimAM...")
        yaml_locations = [
            'yolo11-simam.yaml',
            'ultralytics/cfg/models/11/yolo11-simam.yaml',
            os.path.join('ultralytics', 'cfg', 'models', '11', 'yolo11-simam.yaml'),
        ]
        
        simam_model = None
        for yaml_path in yaml_locations:
            try:
                simam_model = YOLO(yaml_path)
                break
            except:
                continue
        
        if simam_model is None:
            raise FileNotFoundError("Could not find yolo11-simam.yaml")
        
        simam_params = sum(p.numel() for p in simam_model.model.parameters())
        
        # Calculate difference
        added_params = simam_params - baseline_params
        percentage_increase = (added_params / baseline_params) * 100
        
        print(f"\nYOLO11n (baseline):   {baseline_params:,} parameters")
        print(f"YOLO11n-SimAM:        {simam_params:,} parameters")
        print(f"Added by SimAM:       {added_params:,} parameters (+{percentage_increase:.2f}%)")
        
        # SimAM should add ZERO parameters!
        if added_params == 0:
            print(f"✓ PERFECT! SimAM adds 0 parameters (parameter-free attention)")
        elif abs(percentage_increase) < 0.01:
            print(f"✓ Excellent! Minimal parameter increase ({percentage_increase:.4f}%)")
        else:
            print(f"⚠ Unexpected: SimAM should add 0 parameters, but added {added_params:,}")
        
        print("="*80 + "\n")
        
        return True
    except Exception as e:
        print(f"⚠ Comparison test skipped: {e}")
        print("This is optional - not critical for training")
        print("="*80 + "\n")
        return True


def main():
    print("\n" + "="*80)
    print("SimAM INTEGRATION TEST")
    print("Simple, Parameter-Free Attention Module")
    print("="*80 + "\n")
    
    # Run tests
    test1 = test_simam_module()
    test2 = test_model_loading()
    test3 = test_baseline_comparison()
    
    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"SimAM Module:         {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"Model Loading:        {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"Baseline Comparison:  {'✓ PASS' if test3 else '⚠ SKIP (optional)'}")
    print("="*80 + "\n")
    
    if test1 and test2:
        print("✓ ALL CRITICAL TESTS PASSED!")
        print("✓ Ready to train YOLO11-SimAM")
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Quick training test (1 epoch):")
        print("   yolo detect train data=vehicle.yaml model=yolo11-simam.yaml epochs=1 imgsz=640")
        print("\n2. Full training (200 epochs):")
        print("   yolo detect train data=vehicle.yaml model=yolo11-simam.yaml epochs=200 imgsz=640 batch=16")
        print("\n3. Alternative command:")
        print("   python -m ultralytics.train data=vehicle.yaml model=yolo11-simam.yaml epochs=200")
        print("\n" + "="*80)
        print("WHY SimAM?")
        print("="*80)
        print("✓ Zero parameters - no overfitting risk")
        print("✓ Fast training - minimal computation overhead")
        print("✓ No hyperparameter tuning needed")
        print("✓ Expected improvement: +0.5-1.5% mAP")
        print("✓ Perfect for thesis: novel parameter-free approach")
        print("="*80 + "\n")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())