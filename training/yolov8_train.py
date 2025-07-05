# scripts/advanced_train.py (Modified for command-line arguments)
from ultralytics import YOLO
import torch
import os
from datetime import datetime
import argparse  # Import the argparse module

def configure_training(imgsz, epochs, model_size): # Function now takes arguments directly
    # Create unique run identifier
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'dataset', 'data.yaml')
    output_dir = os.path.join(base_dir, 'runs', 'detect', f'train_{model_size}_augmented_res{imgsz}_ep{epochs}_{run_id}') # Dynamic output directory name

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Debugging path information
    print(f"🗂️ Dataset Path: {dataset_path}")
    print(f"📁 Output Directory: {output_dir}")
    print(f"🖼️ Image Size: {imgsz}")
    print(f"⏳ Epochs: {epochs}")
    print(f"🤖 Model Size: {model_size}")

    # Verify paths exist
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset configuration not found: {dataset_path}")

    # Load pre-trained YOLOv8 model based on model_size parameter
    model = YOLO(f'{model_size}.pt')

    # Comprehensive training configuration with augmentations
    results = model.train(
        data=dataset_path,
        epochs=epochs,             # Use the epochs parameter from command line
        patience=15,             # Early stopping to prevent overfitting
        imgsz=imgsz,             # Use the imgsz parameter from command line
        batch=4,                 # Conservative batch size for 6GB GPU (adjust if needed for higher imgsz)
        device=0,                # Primary GPU
        workers=2,               # Match multi-processor count
        augment=True,            # Enable data augmentation
        degrees=10.0,            # Add slight rotation
        translate=0.1,           # Add slight translation
        scale=0.1,               # Add slight scaling
        shear=5.0,               # Add slight shear
        perspective=0.0,         # Keep perspective augmentation off initially
        flipud=0.0,              # No vertical flips for sign language usually
        fliplr=0.5,              # Horizontal flips might be useful
        mosaic=1.0,              # Keep mosaic augmentation
        mixup=0.1,               # Keep mixup augmentation
        copy_paste=0.0,          # Copy-paste augmentation is also good, but let's start with basics
        lr0=0.01,                # Initial learning rate
        lrf=0.1,                 # Learning rate decay factor
        project=output_dir,      # Custom output directory
        name=f'{model_size}_augmented_res{imgsz}_ep{epochs}_training',  # Dynamic run name
        save_period=10,          # Save checkpoint every 10 epochs
        plots=True,              # Generate performance plots
        verbose=True             # Detailed logging
    )

    return results

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Train YOLOv8 model with configurable image size and epochs.")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size for training (default: 640)")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument('--model_size', type=str, default='yolov8s', choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'], help="YOLOv8 model size (default: yolov8s, options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)") # Added model_size argument
    args = parser.parse_args()

    imgsz = args.imgsz
    epochs = args.epochs
    model_size = args.model_size

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Ensure GPU is configured correctly.")
        return

    # Print GPU information
    print(f"🖥️ Training on: {torch.cuda.get_device_name(0)}")
    print(f"📊 Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Start training
    try:
        results = configure_training(imgsz=imgsz, epochs=epochs, model_size=model_size) # Pass arguments to configure_training
        print("✅ Training Completed. Detailed results available in training directory.")
    except Exception as e:
        print(f"❌ Training Failed: {e}")

if __name__ == "__main__":
    main()
