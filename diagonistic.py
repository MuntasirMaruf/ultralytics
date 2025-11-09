#!/usr/bin/env python3
"""
Diagnostic script to check CBAM integration
"""

print("="*80)
print("DIAGNOSTIC CHECK")
print("="*80 + "\n")

# Test 1: Import CBAMCustom
print("1. Testing CBAMCustom import...")
try:
    from ultralytics.nn.modules import CBAMCustom
    print("   ✓ CBAMCustom imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import CBAMCustom: {e}")
    exit(1)

# Test 2: Check if it's in the registry
print("\n2. Checking module registry...")
try:
    from ultralytics.nn.tasks import parse_model
    import ultralytics.nn.modules as modules
    
    # Check if CBAMCustom is accessible
    if hasattr(modules, 'CBAMCustom'):
        print("   ✓ CBAMCustom found in modules")
    else:
        print("   ✗ CBAMCustom NOT in modules")
    
    # Try to get it directly
    cbam_class = getattr(modules, 'CBAMCustom', None)
    if cbam_class:
        print(f"   ✓ CBAMCustom class: {cbam_class}")
    else:
        print("   ✗ Could not get CBAMCustom class")
        
except Exception as e:
    print(f"   ✗ Registry check failed: {e}")

# Test 3: Check YAML file
print("\n3. Checking YAML file...")
import os
yaml_path = 'ultralytics/cfg/models/11/yolo11-cbam.yaml'
if os.path.exists(yaml_path):
    print(f"   ✓ YAML file exists: {yaml_path}")
    
    # Read and check content
    with open(yaml_path, 'r') as f:
        content = f.read()
        if 'CBAMCustom' in content:
            print("   ✓ 'CBAMCustom' found in YAML")
            # Count occurrences
            count = content.count('CBAMCustom')
            print(f"   ✓ CBAMCustom appears {count} times")
        else:
            print("   ✗ 'CBAMCustom' NOT found in YAML")
            print("   Looking for 'CBAM'...")
            if 'CBAM' in content:
                print("   ! Found 'CBAM' - should be 'CBAMCustom'!")
else:
    print(f"   ✗ YAML file not found: {yaml_path}")

# Test 4: Try to manually parse
print("\n4. Testing manual model creation...")
try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import yaml_model_load
    
    # Read the yaml
    import yaml as pyyaml
    with open(yaml_path, 'r') as f:
        yaml_dict = pyyaml.safe_load(f)
    
    print(f"   ✓ YAML loaded successfully")
    print(f"   Model has {len(yaml_dict.get('backbone', []))} backbone layers")
    
    # Check for CBAMCustom in backbone
    backbone = yaml_dict.get('backbone', [])
    cbam_layers = [i for i, layer in enumerate(backbone) if 'CBAMCustom' in str(layer)]
    if cbam_layers:
        print(f"   ✓ Found CBAMCustom in layers: {cbam_layers}")
        print(f"   Example layer: {backbone[cbam_layers[0]]}")
    else:
        print("   ✗ No CBAMCustom layers found in backbone")
        
except Exception as e:
    print(f"   ✗ Manual parsing failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Check if it's a parsing issue
print("\n5. Testing module instantiation...")
try:
    import torch
    cbam = CBAMCustom(c1=256)
    x = torch.randn(1, 256, 20, 20)
    out = cbam(x)
    print(f"   ✓ CBAMCustom instantiation works")
    print(f"   ✓ Forward pass works: {x.shape} -> {out.shape}")
except Exception as e:
    print(f"   ✗ Instantiation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)