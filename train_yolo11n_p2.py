"""
Training script for YOLO11n with P2 head
Optimized for dashcam vehicle detection
"""

from ultralytics import YOLO
import torch

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Load the modified model
model = YOLO('ultralytics/cfg/models/11/yolo11n-p2.yaml')  # Make sure this file is in the same directory

# Training configuration
results = model.train(
    # Dataset
    data='data.yaml',  # CHANGE THIS to your dataset config path
    
    # Training duration
    epochs=150,
    
    # Image settings
    imgsz=640,
    batch=16,  # Adjust based on your GPU memory (8/16/32)
    
    # Optimizer settings
    optimizer='AdamW',  # AdamW often works better than SGD
    lr0=0.001,          # Initial learning rate
    lrf=0.01,           # Final learning rate (OneCycleLR)
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    
    # Loss weights (CRITICAL FOR BBOX IMPROVEMENT)
    box=7.5,            # Box loss gain (increased from default)
    cls=0.5,            # Class loss gain
    dfl=1.5,            # Distribution Focal Loss gain (increased)
    
    # Augmentation (optimized for dashcam)
    hsv_h=0.015,        # HSV-Hue augmentation
    hsv_s=0.7,          # HSV-Saturation
    hsv_v=0.4,          # HSV-Value
    degrees=2.0,        # Rotation augmentation (small for dashcam)
    translate=0.1,      # Translation augmentation
    scale=0.5,          # Scale augmentation
    shear=0.0,          # Shear augmentation
    perspective=0.0,    # Perspective augmentation
    flipud=0.0,         # Flip up-down probability
    fliplr=0.5,         # Flip left-right probability
    mosaic=1.0,         # Mosaic augmentation probability
    mixup=0.1,          # Mixup augmentation probability (helps small objects)
    copy_paste=0.0,     # Copy-paste augmentation
    
    # Learning rate scheduler
    cos_lr=True,        # Use cosine learning rate scheduler
    
    # Hardware
    device=0,           # GPU device (0 for first GPU, 'cpu' for CPU)
    workers=8,          # Number of worker threads for data loading
    
    # Saving
    project='thesis_experiments',
    name='yolo11n_p2_optimized',
    exist_ok=False,
    pretrained=False,   # Start from scratch
    
    # Validation
    val=True,
    save=True,
    save_period=-1,     # Save checkpoint every x epochs (-1 = only save last)
    
    # Verbosity
    verbose=True,
    plots=True,         # Generate training plots
)

# Print final results
print("\n" + "="*50)
print("TRAINING COMPLETED")
print("="*50)
print(f"Best mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
print(f"Best mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
print(f"Precision: {results.results_dict['metrics/precision(B)']:.4f}")
print(f"Recall: {results.results_dict['metrics/recall(B)']:.4f}")
print("="*50)

# Validate on test set
print("\nValidating on test set...")
metrics = model.val()
print(f"\nTest Results:")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"mAP per class: {metrics.box.maps}")  # Shows AP for each vehicle class