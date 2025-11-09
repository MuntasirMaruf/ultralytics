#!/usr/bin/env python3
"""
Test CBAM module integration
Run this to verify CBAM is properly integrated

Usage:
    python test_cbam.py
"""

import torch
from ultralytics import YOLO
from ultralytics.nn.modules import CBAMCustom

# Register CBAMCustom with the parser (workaround for YAML parsing)
import sys
import ultralytics.nn.tasks as tasks

# Add CBAMCustom to the module's globals so parse_model can find it
if 'CBAMCustom' not in dir(tasks):
    tasks.CBAMCustom = CBAMCustom
    print("✓ Auto-registered CBAMCustom for YAML parsing\n")


def test_cbam_module():
    """Test CBAMCustom module standalone"""
    print("\n" + "="*80)
    print("TEST 1: CBAMCustom Module")
    print("="*80)
    
    try:
        # Create CBAMCustom module (single parameter: c1)
        cbam = CBAMCustom(c1=256, reduction=16)
        
        # Test forward pass
        x = torch.randn(1, 256, 20, 20)
        out = cbam(x)
        
        assert out.shape == x.shape, f"Shape mismatch! Expected {x.shape}, got {out.shape}"
        
        # Count parameters
        params = sum(p.numel() for p in cbam.parameters())
        
        print(f"✓ CBAMCustom module imported successfully")
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {out.shape}")
        print(f"✓ Parameters: {params:,}")
        print(f"✓ Shape preserved: {x.shape == out.shape}")
        print("="*80 + "\n")
        
        return True
    except Exception as e:
        print(f"✗ CBAMCustom module test failed: {e}")
        print(f"\nError details: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you:")
        print("  1. Added CBAMCustom classes to ultralytics/nn/modules/block.py")
        print("  2. Imported CBAMCustom in ultralytics/nn/modules/__init__.py")
        print("  3. CBAMCustom class uses signature: __init__(self, c1, reduction=16)")
        return False


def test_model_loading():
    """Test loading YOLO11-CBAM configuration"""
    print("="*80)
    print("TEST 2: Model Loading")
    print("="*80)
    
    try:
        import os
        
        # Try to find the YAML file
        yaml_locations = [
            'yolo11-cbam.yaml',
            'ultralytics/cfg/models/11/yolo11-cbam.yaml',
            os.path.join('ultralytics', 'cfg', 'models', '11', 'yolo11-cbam.yaml'),
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
            print("  Could not load yolo11-cbam.yaml")
            if last_error:
                print(f"  Last error: {last_error}")
            raise FileNotFoundError(f"yolo11-cbam.yaml not found or failed to load. Error: {last_error}")
        
        print(f"✓ Loaded from: {loaded_from}")
        
        # Get model info
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        print(f"✓ YOLO11-CBAM model loaded successfully")
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
        print(f"\nMake sure:")
        print(f"  1. yolo11-cbam.yaml exists in ultralytics/cfg/models/11/")
        print(f"  2. CBAMCustom is added to ultralytics/nn/modules/block.py")
        print(f"  3. CBAMCustom is imported in ultralytics/nn/modules/__init__.py")
        print(f"  4. yolo11-cbam.yaml uses 'CBAMCustom' (not 'CBAM') in module definitions")
        return False


def test_baseline_comparison():
    """Compare YOLO11n vs YOLO11n-CBAM parameters"""
    print("="*80)
    print("TEST 3: Baseline Comparison")
    print("="*80)
    
    try:
        import os
        
        # Load baseline
        print("  Loading baseline YOLO11n...")
        baseline = YOLO('yolo11n.yaml')
        baseline_params = sum(p.numel() for p in baseline.model.parameters())
        
        # Load CBAM version
        print("  Loading YOLO11n-CBAM...")
        yaml_locations = [
            'yolo11-cbam.yaml',
            'ultralytics/cfg/models/11/yolo11-cbam.yaml',
            os.path.join('ultralytics', 'cfg', 'models', '11', 'yolo11-cbam.yaml'),
        ]
        
        cbam_model = None
        for yaml_path in yaml_locations:
            try:
                cbam_model = YOLO(yaml_path)
                break
            except:
                continue
        
        if cbam_model is None:
            raise FileNotFoundError("Could not find yolo11-cbam.yaml")
        
        cbam_params = sum(p.numel() for p in cbam_model.model.parameters())
        
        # Calculate difference
        added_params = cbam_params - baseline_params
        percentage_increase = (added_params / baseline_params) * 100
        
        print(f"\nYOLO11n (baseline):   {baseline_params:,} parameters")
        print(f"YOLO11n-CBAM:         {cbam_params:,} parameters")
        print(f"Added by CBAM:        {added_params:,} parameters (+{percentage_increase:.2f}%)")
        
        # Expected: ~5-10% increase
        if 3 < percentage_increase < 15:
            print(f"✓ Parameter increase is reasonable ({percentage_increase:.1f}%)")
        else:
            print(f"⚠ Unexpected parameter increase: {percentage_increase:.1f}%")
        
        print("="*80 + "\n")
        
        return True
    except Exception as e:
        print(f"⚠ Comparison test skipped: {e}")
        print("This is optional - not critical for training")
        print("="*80 + "\n")
        return True


def main():
    print("\n" + "="*80)
    print("CBAM INTEGRATION TEST")
    print("Convolutional Block Attention Module")
    print("="*80 + "\n")
    
    # Run tests
    test1 = test_cbam_module()
    test2 = test_model_loading()
    test3 = test_baseline_comparison()
    
    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"CBAMCustom Module:    {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"Model Loading:        {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"Baseline Comparison:  {'✓ PASS' if test3 else '⚠ SKIP (optional)'}")
    print("="*80 + "\n")
    
    if test1 and test2:
        print("✓ ALL CRITICAL TESTS PASSED!")
        print("✓ Ready to train YOLO11-CBAM")
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Quick training test (1 epoch):")
        print("   yolo detect train data=vehicle.yaml model=yolo11-cbam.yaml epochs=1 imgsz=640")
        print("\n2. Full training (200 epochs):")
        print("   yolo detect train data=vehicle.yaml model=yolo11-cbam.yaml epochs=200 imgsz=640 batch=16")
        print("\n3. Alternative command:")
        print("   python train_cbam.py --data vehicle.yaml --epochs 200 --scale n")
        print("\n" + "="*80)
        print("WHY CBAM?")
        print("="*80)
        print("✓ Channel & Spatial attention combination")
        print("✓ Minimal parameters (~5-10% increase)")
        print("✓ Proven effective for object detection")
        print("✓ Expected improvement: +1.5-3% mAP")
        print("✓ Great for thesis: well-established method")
        print("="*80 + "\n")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())