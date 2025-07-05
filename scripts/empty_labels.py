# find_empty_labels.py
import os

def find_empty_label_files(label_dir):
    empty_files = []
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(label_dir, filename)
            with open(filepath, 'r') as f:
                content = f.read().strip()  # Read and remove whitespace
                if not content:  # Check if the content is empty
                    empty_files.append(filename)
    return empty_files

if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_root = os.path.join(project_root, 'dataset')
    label_dirs = [
        os.path.join(dataset_root, 'labels', 'train'),
        os.path.join(dataset_root, 'labels', 'val'),
        os.path.join(dataset_root, 'labels', 'test'),
    ]

    for label_dir in label_dirs:
        empty_files = find_empty_label_files(label_dir)
        if empty_files:
            print(f"Empty or whitespace-only label files in {label_dir}:")
            for filename in empty_files:
                print(f"  - {filename}")
        else:
            print(f"No empty label files found in {label_dir}")
