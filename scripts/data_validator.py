import os
import cv2

def validate_dataset(base_path):
    # Specify paths for images and labels
    image_dir = os.path.join(base_path, 'images', 'train')
    label_dir = os.path.join(base_path, 'labels', 'train')
    
    # List image and label files
    images = os.listdir(image_dir)
    labels = os.listdir(label_dir)
    
    # Print basic dataset information
    print(f"Total Images: {len(images)}")
    print(f"Total Labels: {len(labels)}")
    
    # Validate label format
    for label_file in labels:
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Check label format (class_id, x_center, y_center, width, height)
                assert len(parts) == 5, f"Invalid label format in {label_file}"
                
                # Validate normalized coordinates (should be between 0 and 1)
                assert 0 <= float(parts[1]) <= 1, "Invalid x coordinate"
                assert 0 <= float(parts[2]) <= 1, "Invalid y coordinate"
                assert 0 <= float(parts[3]) <= 1, "Invalid width"
                assert 0 <= float(parts[4]) <= 1, "Invalid height"
    
    print("Dataset validation successful!")

# Run validation on the dataset
validate_dataset('../dataset')
