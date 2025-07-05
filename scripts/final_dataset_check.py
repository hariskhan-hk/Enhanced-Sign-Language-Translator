# ~/yolo_project/scripts/final_dataset_check.py
import os
import yaml

def validate_dataset_structure():
    # Base path to your dataset
    base_path = '../dataset'
    
    # Paths to check
    dataset_paths = {
        'train_images': os.path.join(base_path, 'images', 'train'),
        'train_labels': os.path.join(base_path, 'labels', 'train'),
        'val_images': os.path.join(base_path, 'images', 'val'),
        'val_labels': os.path.join(base_path, 'labels', 'val'),
        'test_images': os.path.join(base_path, 'images', 'test'),
        'test_labels': os.path.join(base_path, 'labels', 'test')
    }
    
    # Detailed validation
    print("🔍 Dataset Structure Validation:")
    
    # Check if directories exist
    for name, path in dataset_paths.items():
        if os.path.exists(path):
            contents = os.listdir(path)
            print(f"✓ {name}: {len(contents)} files")
        else:
            print(f"❌ {name}: Directory not found!")
    
    # Validate data.yaml
    data_yaml_path = os.path.join(base_path, 'data.yaml')
    
    try:
        with open(data_yaml_path, 'r') as file:
            data_config = yaml.safe_load(file)
        
        print("\n📋 data.yaml Configuration:")
        print(f"Classes: {data_config.get('nc', 'Not specified')}")
        print("Class Names:", data_config.get('names', 'Not specified'))
        
        # Verify paths in data.yaml
        print("\n🗂️ Configuration Paths:")
        for key in ['train', 'val', 'test']:
            print(f"{key.capitalize()} Path: {data_config.get(key, 'Not specified')}")
    
    except Exception as e:
        print(f"❌ Error reading data.yaml: {e}")

if __name__ == "__main__":
    validate_dataset_structure()
