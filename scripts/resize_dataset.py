# ~/yolo_project/scripts/resize_dataset.py
import os
import cv2
from tqdm import tqdm  # For progress bar

def resize_dataset(base_path, target_size=(640, 640)):  # Target size set to 640x640
    """Resizes images and updates bounding box annotations."""

    for split in ['train', 'val', 'test']:
        image_dir = os.path.join(base_path, 'images', split)
        label_dir = os.path.join(base_path, 'labels', split)
        new_image_dir = os.path.join(base_path, 'images_resized', split)
        new_label_dir = os.path.join(base_path, 'labels_resized', split)
        os.makedirs(new_image_dir, exist_ok=True)
        os.makedirs(new_label_dir, exist_ok=True)

        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        for image_file in tqdm(image_files, desc=f"Resizing {split} set"):
            image_path = os.path.join(image_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)

            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}. Skipping.")
                continue
            h, w = img.shape[:2]
            img_resized = cv2.resize(img, target_size)

            cv2.imwrite(os.path.join(new_image_dir, image_file), img_resized)

            # Update bounding box coordinates
            with open(label_path, 'r') as f_in:
                lines = f_in.readlines()

            with open(os.path.join(new_label_dir, label_file), 'w') as f_out:
                for line in lines:
                    parts = line.strip().split()
                    class_id, x_center, y_center, width, height = parts
                    x_center = float(x_center) * target_size[0] / w
                    y_center = float(y_center) * target_size[1] / h
                    width = float(width) * target_size[0] / w
                    height = float(height) * target_size[1] / h

                    f_out.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'dataset')
    resize_dataset(dataset_path)
    print("Resized dataset created in 'images_resized' and 'labels_resized'.")
