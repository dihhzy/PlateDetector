import os
import yaml
import glob
from roboflow import Roboflow
from ultralytics import YOLO
from dotenv import load_dotenv
load_dotenv()

def train_new_dataset():
    print("---DOWNLOADING DATASET---")
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("❌ ROBOFLOW_API_KEY environment variable not set!")
    
    dataset = project.version(2).download("yolov11")
    dataset_path = dataset.location
    print(f"Dataset downloaded to: {dataset_path}")

    yaml_path = os.path.join(dataset_path, "data.yaml")
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    class_names = config['names']
    print(f"   Classes found: {class_names}")

    target_id = -1
    for idx, name in enumerate(class_names):
        name_clean = name.lower()
        if 'plat' in name_clean or 'license' in name_clean or 'number' in name_clean:
            target_id = idx
            print(f"   🎯 Target Class Found: '{name}' (ID: {target_id})")
            break
    
    if target_id == -1:
        print("Label name error")
        target_id = 0 
        print(f"-> Defaulting to Class ID {target_id}")

    
    # Cleaning Labels
    txt_files = glob.glob(f"{dataset_path}/**/*.txt", recursive=True)
    for file_path in txt_files:
        if file_path.endswith("classes.txt"): continue
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            try:
                cid = int(parts[0])
                if cid == target_id:
                    # Change to 0 for single-class training
                    parts[0] = "0"
                    new_lines.append(" ".join(parts) + "\n")
            except: continue
        
        with open(file_path, 'w') as f:
            f.writelines(new_lines)

    # Update YAML to reflect 1 class
    config['nc'] = 1
    config['names'] = ['license_plate']

    config['path'] = dataset_path # Root dir
    config['train'] = os.path.join(dataset_path, "train", "images")
    
    # Handle 'valid' vs 'val' folder naming
    if os.path.exists(os.path.join(dataset_path, "valid")):
        config['val'] = os.path.join(dataset_path, "valid", "images")
    elif os.path.exists(os.path.join(dataset_path, "val")):
        config['val'] = os.path.join(dataset_path, "val", "images")
    else:
        print("No validation set found. Using training set for validation.")
        config['val'] = os.path.join(dataset_path, "train", "images")

    # clear test to avoid errors if missing
    if 'test' in config:
        del config['test']

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)

    # ---TRAIN ---
    print("\n--- STARTING TRAINING---")
    model = YOLO('yolo11n.pt')
    
    model.train(
        data=yaml_path,
        epochs=50, 
        imgsz=640,
        batch=16,
        patience=10,
        plots=True,
        name='indo_plate_v2'
    )
    
    print("\nTRAINING FINISHED!")

if __name__ == "__main__":
    train_new_dataset()