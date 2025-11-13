"""
Training Scripts for Hybrid Ensemble Approach
Run these in separate terminals for parallel training
"""

from ultralytics import YOLO
import torch

# ============================================
# MODEL A: YOLOv11s-CBAM (640px, Attention)
# ============================================
def train_model_a_cbam():
    """
    Model A: YOLOv11s with CBAM attention
    Focus: Balanced performance with attention mechanism
    """
    print("="*70)
    print("TRAINING MODEL A: YOLOv11s-CBAM @ 640px")
    print("="*70)
    
    # Load custom architecture with COCO pretrained weights
    model = YOLO('yolo11s.pt')  # Load weights first
    model = YOLO('/content/ultralytics/ultralytics/cfg/models/11/yolo11s-cbam.yaml').load('yolo11s.pt')  # Transfer to custom architecture
    
    results = model.train(
        # Data — UPDATED PATH
        data='/content/road_ds_new/data.yaml',
        
        # Training duration
        epochs=300,
        patience=50,
        
        # Image settings
        imgsz=640,
        batch=32,
        workers=8,
        device=0,
        
        # Optimizer
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Augmentation - BALANCED
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.2,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Validation
        val=True,
        plots=True,
        save=True,
        save_period=10,
        
        # Close mosaic in final epochs
        close_mosaic=10,
        
        # Project
        project='hybrid_ensemble',
        name='model_a_cbam_640',
        exist_ok=False,
        
        # Advanced
        amp=True,
        fraction=1.0,
        profile=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        seed=0,
    )
    
    print("\n✓ Model A training complete!")
    return model, results


# ============================================
# MODEL B: YOLOv8m (800px, High Accuracy)
# ============================================
def train_model_b_accuracy():
    """
    Model B: YOLOv8m at higher resolution
    Focus: Maximum accuracy, catches small objects
    """
    print("="*70)
    print("TRAINING MODEL B: YOLOv8m @ 800px")
    print("="*70)
    
    model = YOLO('yolov8m.pt')
    
    results = model.train(
        # Data — UPDATED PATH
        data='/content/road_ds_new/data.yaml',
        
        # Training duration
        epochs=300,
        patience=50,
        
        # Image settings - HIGHER RESOLUTION
        imgsz=800,
        batch=16,  # Smaller batch due to higher resolution
        workers=8,
        device=0,
        
        # Optimizer - SGD for stability at high resolution
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Augmentation - CONSERVATIVE (high res = less need for aug)
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.0,
        copy_paste=0.1,
        
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Validation
        val=True,
        plots=True,
        save=True,
        save_period=10,
        
        # Close mosaic
        close_mosaic=10,
        
        # Project
        project='hybrid_ensemble',
        name='model_b_yolov8m_800',
        exist_ok=False,
        
        # Advanced
        amp=True,
        fraction=1.0,
    )
    
    print("\n✓ Model B training complete!")
    return model, results


# ============================================
# MODEL C: YOLOv11s (640px, Minority Specialist)
# ============================================
def train_model_c_minority():
    """
    Model C: YOLOv11s with aggressive augmentation
    Focus: Minority class performance (person, two_wheeler, suv, pickup_truck)
    """
    print("="*70)
    print("TRAINING MODEL C: YOLOv11s @ 640px (Minority Specialist)")
    print("="*70)
    
    model = YOLO('yolo11s.pt')
    
    results = model.train(
        # Data — UPDATED PATH
        data='/content/road_ds_new/data.yaml',
        
        # Training duration
        epochs=300,
        patience=50,
        
        # Image settings
        imgsz=640,
        batch=32,
        workers=8,
        device=0,
        
        # Optimizer
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Augmentation - AGGRESSIVE for minority classes
        hsv_h=0.02,
        hsv_s=0.9,
        hsv_v=0.5,
        degrees=15.0,
        translate=0.3,
        scale=0.7,
        shear=3.0,
        perspective=0.0001,
        flipud=0.1,  # Enable vertical flip
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.3,  # High mixup
        copy_paste=0.5,  # Critical for minority classes
        
        # Loss weights - emphasize classification
        box=7.5,
        cls=1.0,  # Higher class loss weight
        dfl=1.5,
        
        # Validation
        val=True,
        plots=True,
        save=True,
        save_period=10,
        
        # Close mosaic
        close_mosaic=10,
        
        # Project
        project='hybrid_ensemble',
        name='model_c_minority_640',
        exist_ok=False,
        
        # Advanced
        amp=True,
        fraction=1.0,
    )
    
    print("\n✓ Model C training complete!")
    return model, results


# ============================================
# MAIN: Launch All Training
# ============================================
if __name__ == "__main__":
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║        HYBRID ENSEMBLE TRAINING - OPTION C                           ║
║        3 Models in Parallel for Maximum Performance                  ║
╚══════════════════════════════════════════════════════════════════════╝

Models to train:
  A. YOLOv11s-CBAM @ 640px  (Attention + Balance)
  B. YOLOv8m @ 800px        (High Accuracy)
  C. YOLOv11s @ 640px       (Minority Specialist)

Usage:
  python train.py A  # Train Model A
  python train.py B  # Train Model B
  python train.py C  # Train Model C
  python train.py ALL  # Train all sequentially (not recommended)

Recommended: Open 3 terminals and run each model separately!

Expected Training Time: 6-8 hours per model
Expected Improvement: +6-10% mAP with ensemble
""")
    
    if len(sys.argv) < 2:
        print("Please specify which model to train: A, B, C, or ALL")
        sys.exit(1)
    
    model_choice = sys.argv[1].upper()
    
    if model_choice == 'A':
        train_model_a_cbam()
    elif model_choice == 'B':
        train_model_b_accuracy()
    elif model_choice == 'C':
        train_model_c_minority()
    elif model_choice == 'ALL':
        print("\n⚠️  Training all models sequentially will take 18-24 hours!")
        print("Recommended: Run in parallel using 3 terminals instead.\n")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() == 'yes':
            train_model_a_cbam()
            train_model_b_accuracy()
            train_model_c_minority()
        else:
            print("Cancelled. Use separate terminals for parallel training.")
    else:
        print(f"Unknown option: {model_choice}")
        print("Use: A, B, C, or ALL")