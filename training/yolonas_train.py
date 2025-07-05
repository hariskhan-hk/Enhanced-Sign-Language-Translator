import os
import torch
import cv2
import matplotlib.pyplot as plt
import yaml
import argparse
import glob
import warnings
from pathlib import Path

# --- Framework-Specific Imports ---
try:
    from super_gradients.training import Trainer, models
    from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_val
    from super_gradients.training.metrics import DetectionMetrics_050, DetectionMetrics
    from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
    _SUPER_GRADIENTS_AVAILABLE = True
except ImportError:
    _SUPER_GRADIENTS_AVAILABLE = False
    warnings.warn("SuperGradients not found. YOLO-NAS evaluation will not be available.")

try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False
    warnings.warn("Ultralytics not found. YOLOv8 evaluation will not be available.")

# --- Configuration ---
# Adjust these paths based on your project structure
YOLONAS_PROJECT_ROOT = '/home/chaoder/urdu_sign_language_yolonas'
YOLOV8_PROJECT_ROOT = '/home/chaoder/yolo_project'

# Shared Dataset Configuration (Assuming same dataset used for both)
# Use the dataset path relevant to the model being tested OR a common path
# We will determine the specific dataset path based on the model_type argument later
DATASET_NAME = 'dataset' # Subdirectory name within project roots

# Shared Classes (MUST be identical and in the correct order for both models)
CLASSES = ['C1', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C2', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C3', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
NUM_CLASSES = len(CLASSES)

# Model Specific Defaults (Can be overridden by args)
YOLONAS_DEFAULT_ARCH = 'yolo_nas_s'
YOLONAS_DEFAULT_EXP_NAME = 'yolonas_urdu_sign_language'
YOLOV8_DEFAULT_MODEL_SIZE = 's' # e.g., 'n', 's', 'm', 'l', 'x' - used if finding latest dynamically

# --- Helper Functions ---

def find_latest_yolonas_checkpoint(checkpoint_dir, experiment_name):
    """Find the most recent YOLO-NAS checkpoint (best preferred)."""
    if not _SUPER_GRADIENTS_AVAILABLE:
        raise ImportError("SuperGradients is required for YOLO-NAS.")

    experiment_dir = os.path.join(checkpoint_dir, experiment_name)
    if not os.path.isdir(experiment_dir):
        raise FileNotFoundError(f"YOLO-NAS experiment directory not found: {experiment_dir}")

    run_dirs = [d for d in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, d)) and d.startswith('RUN_')]
    if not run_dirs:
        raise FileNotFoundError(f"No runs found in {experiment_dir}")

    try:
        run_dirs.sort(key=lambda d: d.split('_')[1] + d.split('_')[2], reverse=True)
    except IndexError:
        run_dirs.sort(reverse=True) # Fallback sort

    latest_run_dir = os.path.join(experiment_dir, run_dirs[0])

    # Prioritize 'best', then 'average'
    best_ckpt_path = os.path.join(latest_run_dir, 'ckpt_best.pth')
    avg_ckpt_path = os.path.join(latest_run_dir, 'average_model.pth')

    # if os.path.exists(best_ckpt_path):
    #     print(f"Found best YOLO-NAS checkpoint: {best_ckpt_path}")
    #     return best_ckpt_path
    if os.path.exists(avg_ckpt_path):
        print(f"Found average YOLO-NAS checkpoint (best not found): {avg_ckpt_path}")
        return avg_ckpt_path
    else:
        raise FileNotFoundError(f"No 'ckpt_best.pth' or 'average_model.pth' found in the latest run directory: {latest_run_dir}")

def find_latest_yolov8_checkpoint(runs_dir):
    """Find the most recent YOLOv8 'best.pt' checkpoint."""
    if not _ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics is required for YOLOv8.")

    detect_dir = os.path.join(runs_dir, 'detect')
    if not os.path.isdir(detect_dir):
        raise FileNotFoundError(f"YOLOv8 runs/detect directory not found: {detect_dir}")

    train_dirs = glob.glob(os.path.join(detect_dir, 'train*')) # Find all 'train*' directories
    if not train_dirs:
        raise FileNotFoundError(f"No 'train*' directories found in {detect_dir}")

    # Sort by modification time or name to find the latest
    try:
        # Try sorting by embedded timestamp if present (e.g., train_YYYYMMDD_HHMMSS)
        # This depends heavily on the exact naming convention used by Ultralytics versions
        latest_train_dir = max(train_dirs, key=os.path.getmtime) # Safer bet: sort by modification time
    except Exception:
         latest_train_dir = sorted(train_dirs)[-1] # Fallback: simple alphabetical sort

    # Check inside common structures: direct 'weights' or nested 'yolov8_training/weights'
    potential_paths = [
        os.path.join(latest_train_dir, 'weights', 'best.pt'),
        os.path.join(latest_train_dir, 'yolov8_training', 'weights', 'best.pt'), # As seen in user's log
    ]

    for model_path in potential_paths:
        if os.path.exists(model_path):
            print(f"Found latest YOLOv8 checkpoint: {model_path}")
            return model_path

    raise FileNotFoundError(f"No 'best.pt' found in the latest YOLOv8 run directory: {latest_train_dir} or its subdirs.")


def create_temp_yolov8_data_yaml(dataset_root, classes_list, temp_yaml_path):
    """Creates a temporary data.yaml for YOLOv8 evaluation with absolute paths."""
    test_img_dir = os.path.join(dataset_root, 'images', 'test')
    test_lbl_dir = os.path.join(dataset_root, 'labels', 'test') # Needed for ground truth

    # Check if test images AND labels exist
    if not os.path.isdir(test_img_dir):
         raise FileNotFoundError(f"YOLOv8 test image directory not found: {test_img_dir}")
    # If test labels don't exist, we can't evaluate metrics properly.
    # Ultralytics .val() might fall back to val set or error out if test labels are missing.
    # We assume test labels *do* exist here, matching the images.
    if not os.path.isdir(test_lbl_dir):
         warnings.warn(f"YOLOv8 test label directory not found: {test_lbl_dir}. Evaluation metrics might be incorrect if ground truth is missing.")
         # Option: Fallback to validation set if test labels missing?
         # test_img_dir = os.path.join(dataset_root, 'images', 'val')
         # print("Warning: Test labels not found. Using validation set for evaluation.")
         # if not os.path.isdir(test_img_dir):
         #    raise FileNotFoundError(f"YOLOv8 validation image directory not found either: {test_img_dir}")


    data_config = {
        'path': dataset_root, # Optional: Used by Ultralytics to resolve relative paths if needed
        'train': os.path.join(dataset_root, 'images', 'train'), # Placeholder, not used for test
        'val': os.path.join(dataset_root, 'images', 'val'),     # Placeholder, not used for test
        'test': test_img_dir, # Path to test images
        'nc': len(classes_list),
        'names': classes_list
    }

    with open(temp_yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=None, sort_keys=False)
    print(f"Created temporary YOLOv8 data YAML: {temp_yaml_path}")
    return temp_yaml_path

# --- Evaluation Functions ---

def evaluate_yolonas(checkpoint_path, dataset_root, classes, device, imgsz=640):
    """Evaluates a YOLO-NAS model."""
    if not _SUPER_GRADIENTS_AVAILABLE:
        raise ImportError("SuperGradients is required for YOLO-NAS.")

    print("\n--- Evaluating YOLO-NAS ---")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset Root: {dataset_root}")
    print(f"Image Size: {imgsz}x{imgsz}") # Note: SG might override this based on model/training

    # Trainer needed for evaluation context
    # Using a temporary dir to avoid polluting original checkpoints
    temp_ckpt_dir = os.path.join(os.path.dirname(checkpoint_path), "temp_eval_logs")
    trainer = Trainer(experiment_name="yolonas_fair_eval", ckpt_root_dir=temp_ckpt_dir)

    model = models.get(
        YOLONAS_DEFAULT_ARCH, # Assuming 's' model, could make this configurable
        num_classes=len(classes),
        checkpoint_path=checkpoint_path
    ).to(device)
    model.eval()

    # Prepare Test Data Loader
    test_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_root,
            'images_dir': os.path.join('images', 'test'), # Relative to data_dir
            'labels_dir': os.path.join('labels', 'test'), # Relative to data_dir
            'classes': classes,
            # Add input_dim if needed, SG often infers it
            # 'input_dim': (imgsz, imgsz)
        },
        dataloader_params={
            'batch_size': 16,
            'num_workers': 2,
            'shuffle': False,
            # Ensure dataloader uses the target image size if possible
            # This might depend on the specific dataset adapter in SG
        }
    )

    # Define Test Metrics (Aligning with common COCO metrics)
    test_metrics_list = [
        DetectionMetrics_050( # Corresponds to mAP@0.50
            score_thres=0.1, # Low threshold for metric calculation range
            top_k_predictions=300,
            num_cls=len(classes),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01, # Lower threshold for considering predictions
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7 # Standard NMS threshold
            )
        ),
        DetectionMetrics( # Corresponds to mAP@0.50:0.95 (COCO primary)
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(classes),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ]

    # Run Testing
    print("📊 Evaluating YOLO-NAS on Test Set...")
    results = trainer.test(model=model,
                           test_loader=test_data,
                           test_metrics_list=test_metrics_list)

    # Extract and Standardize Metrics
    map50 = results.get('mAP@0.50', 'N/A')
    map50_95 = results.get('mAP@0.50:0.95', 'N/A')
    precision = results.get('Precision', 'N/A') # May still be N/A depending on SG version/config
    recall = results.get('Recall', 'N/A')       # May still be N/A

    print("\n📊 YOLO-NAS Performance Metrics:")
    print(f"mAP@0.50: {map50:.4f}" if isinstance(map50, float) else f"mAP@0.50: {map50}")
    print(f"mAP@0.50:0.95: {map50_95:.4f}" if isinstance(map50_95, float) else f"mAP@0.50:0.95: {map50_95}")
    # print(f"Overall Precision: {precision:.4f}" if isinstance(precision, float) else f"Overall Precision: {precision}")
    # print(f"Overall Recall: {recall:.4f}" if isinstance(recall, float) else f"Overall Recall: {recall}")

    # Cleanup temp log dir if empty or desired
    # try: os.rmdir(temp_ckpt_dir) except OSError: pass

    return {"mAP@0.50": map50, "mAP@0.50:0.95": map50_95}, model


def evaluate_yolov8(checkpoint_path, dataset_root, classes, device, imgsz=640):
    """Evaluates a YOLOv8 model."""
    if not _ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics is required for YOLOv8.")

    print("\n--- Evaluating YOLOv8 ---")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset Root: {dataset_root}")
    print(f"Image Size: {imgsz}x{imgsz}")

    model = YOLO(checkpoint_path)
    model.to(device)

    # Create temporary YAML for evaluation
    temp_yaml_path = Path(os.path.dirname(__file__)) / 'temp_eval_data.yaml' # Place temp file near script
    try:
        data_yaml_path = create_temp_yolov8_data_yaml(dataset_root, classes, str(temp_yaml_path))

        # Run Validation (using the 'test' split defined in our temp yaml)
        print("📊 Evaluating YOLOv8 on Test Set...")
        results = model.val(
            data=data_yaml_path,
            imgsz=imgsz,
            batch=16, # Adjust as needed
            conf=0.001, # Standard threshold for COCO metric calculation
            iou=0.7,   # Standard IoU threshold for NMS in COCO eval
            split='test', # Explicitly use the test split
            plots=False, # Don't save plots during metric calculation
            device=device
        )

        # Extract and Standardize Metrics
        # Naming convention might vary slightly between ultralytics versions
        # Check results.box.map50 and results.box.map if results_dict is different
        map50 = results.results_dict.get('metrics/mAP50(B)', results.box.map50 if hasattr(results, 'box') else 'N/A')
        map50_95 = results.results_dict.get('metrics/mAP50-95(B)', results.box.map if hasattr(results, 'box') else 'N/A')
        # precision = results.results_dict.get('metrics/precision(B)', 'N/A')
        # recall = results.results_dict.get('metrics/recall(B)', 'N/A')

        print("\n📊 YOLOv8 Performance Metrics:")
        print(f"mAP@0.50: {map50:.4f}" if isinstance(map50, float) else f"mAP@0.50: {map50}")
        print(f"mAP@0.50:0.95: {map50_95:.4f}" if isinstance(map50_95, float) else f"mAP@0.50:0.95: {map50_95}")
        # print(f"Overall Precision: {precision:.4f}" if isinstance(precision, float) else f"Overall Precision: {precision}")
        # print(f"Overall Recall: {recall:.4f}" if isinstance(recall, float) else f"Overall Recall: {recall}")

        return {"mAP@0.50": map50, "mAP@0.50:0.95": map50_95}, model

    finally:
        # Clean up the temporary YAML file
        if temp_yaml_path.exists():
            os.remove(str(temp_yaml_path))
            print(f"Removed temporary YAML: {temp_yaml_path}")


def predict_and_visualize(model, model_type, dataset_root, device, num_images=5, conf_thresh=0.3):
    """Runs prediction and visualizes results for a few test images."""
    print(f"\n🖼️ Running predictions on {num_images} test images (conf_thresh={conf_thresh})...")
    sample_dir = os.path.join(dataset_root, 'images', 'test')

    if not os.path.isdir(sample_dir):
        print(f"❌ Test image directory not found: {sample_dir}")
        return

    image_files = sorted([
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])[:num_images]

    if not image_files:
        print(f"❌ No test images found in {sample_dir}")
        return

    if model_type == 'yolonas':
        if not _SUPER_GRADIENTS_AVAILABLE: return
        # SG predict handles list of images and has built-in show
        predictions = model.predict(image_files, conf=conf_thresh)
        predictions.show() # Opens matplotlib windows

    elif model_type == 'yolov8':
        if not _ULTRALYTICS_AVAILABLE: return
        # YOLOv8 predict can take a list, results contain plotted images
        results = model.predict(source=image_files, conf=conf_thresh, device=device)

        plt.figure(figsize=(15, 10))
        num_cols = 3
        num_rows = (num_images + num_cols - 1) // num_cols
        for i, r in enumerate(results):
            img_bgr = r.plot() # Returns annotated image as numpy array (BGR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(img_rgb)
            plt.title(f"YOLOv8 Pred: {os.path.basename(image_files[i])}")
            plt.axis('off')
        plt.tight_layout()
        plt.show() # Show matplotlib window

    else:
        print(f"❌ Visualization not implemented for model type: {model_type}")

    print("Prediction visualization complete.")


# --- Main Execution ---

def main(args):
    # --- Device Setup ---
    if args.cpu:
        device = torch.device("cpu")
        print("INFO: Using CPU as requested.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"INFO: Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ WARNING: CUDA not available. Running on CPU.")

    results = {}
    model = None # Keep track of the loaded model for visualization

    try:
        if args.model_type == 'yolonas':
            if not _SUPER_GRADIENTS_AVAILABLE:
                print("❌ Cannot evaluate YOLO-NAS: SuperGradients library not found.")
                return
            project_root = YOLONAS_PROJECT_ROOT
            dataset_root = os.path.join(project_root, DATASET_NAME)
            checkpoint_dir = os.path.join(project_root, 'scripts', 'checkpoints') # As per original script
            exp_name = args.yolonas_exp_name or YOLONAS_DEFAULT_EXP_NAME
            checkpoint_path = args.checkpoint or find_latest_yolonas_checkpoint(checkpoint_dir, exp_name)
            results['yolonas'], model = evaluate_yolonas(checkpoint_path, dataset_root, CLASSES, device, args.imgsz)

        elif args.model_type == 'yolov8':
            if not _ULTRALYTICS_AVAILABLE:
                print("❌ Cannot evaluate YOLOv8: Ultralytics library not found.")
                return
            project_root = YOLOV8_PROJECT_ROOT
            dataset_root = os.path.join(project_root, DATASET_NAME)
            runs_dir = os.path.join(project_root, 'runs') # Standard YOLOv8 runs location
            checkpoint_path = args.checkpoint or find_latest_yolov8_checkpoint(runs_dir)
            results['yolov8'], model = evaluate_yolov8(checkpoint_path, dataset_root, CLASSES, device, args.imgsz)

        else:
            print(f"❌ Unknown model type: {args.model_type}")
            return

        # --- Print Summary ---
        print("\n--- Evaluation Summary ---")
        if 'yolonas' in results:
            print(f"YOLO-NAS mAP@0.50:       {results['yolonas']['mAP@0.50']:.4f}" if isinstance(results['yolonas']['mAP@0.50'], float) else results['yolonas']['mAP@0.50'])
            print(f"YOLO-NAS mAP@0.50:0.95: {results['yolonas']['mAP@0.50:0.95']:.4f}" if isinstance(results['yolonas']['mAP@0.50:0.95'], float) else results['yolonas']['mAP@0.50:0.95'])
        if 'yolov8' in results:
            print(f"YOLOv8 mAP@0.50:         {results['yolov8']['mAP@0.50']:.4f}" if isinstance(results['yolov8']['mAP@0.50'], float) else results['yolov8']['mAP@0.50'])
            print(f"YOLOv8 mAP@0.50:0.95:   {results['yolov8']['mAP@0.50:0.95']:.4f}" if isinstance(results['yolov8']['mAP@0.50:0.95'], float) else results['yolov8']['mAP@0.50:0.95'])

        # --- Visualization (Optional) ---
        if args.visualize and model:
             # Determine correct dataset root again for visualization function
             current_project_root = YOLONAS_PROJECT_ROOT if args.model_type == 'yolonas' else YOLOV8_PROJECT_ROOT
             current_dataset_root = os.path.join(current_project_root, DATASET_NAME)
             predict_and_visualize(model, args.model_type, current_dataset_root, device,
                                   num_images=args.num_vis, conf_thresh=args.conf_thresh)

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure the project structure, dataset, and checkpoints exist.")
    except ImportError as e:
        print(f"❌ Error: Missing library dependency - {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fairly Evaluate YOLO-NAS or YOLOv8 Models")
    parser.add_argument('--model_type', type=str, required=True, choices=['yolonas', 'yolov8'],
                        help="Specify which model type to evaluate.")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to a specific model checkpoint file (optional, overrides auto-find).")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size for evaluation (e.g., 640).")
    parser.add_argument('--cpu', action='store_true', help="Force use CPU even if CUDA is available.")

    # YOLO-NAS specific args (only used if model_type is yolonas)
    parser.add_argument('--yolonas_exp_name', type=str, default=None,
                        help="YOLO-NAS experiment name (if different from default).")

    # Visualization args
    parser.add_argument('--visualize', action='store_true', help="Run prediction on sample images and show results.")
    parser.add_argument('--num_vis', type=int, default=5, help="Number of images to visualize.")
    parser.add_argument('--conf_thresh', type=float, default=0.3, help="Confidence threshold for visualization.")

    # Add arguments for project roots if they differ significantly or are not fixed
    # parser.add_argument('--yolonas_root', type=str, default=YOLONAS_PROJECT_ROOT)
    # parser.add_argument('--yolov8_root', type=str, default=YOLOV8_PROJECT_ROOT)

    args = parser.parse_args()
    main(args)