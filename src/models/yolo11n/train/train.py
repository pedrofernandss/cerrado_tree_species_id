import sys
import os

user_site = "/mnt/sdb-seagate/graduacao/python_userbase/ana_pedro/lib/python3.9/site-packages"
if user_site not in sys.path:
    sys.path.insert(0, user_site) 

import gc
import torch
import argparse
from ultralytics import YOLO

def train_and_validate():
    parser = argparse.ArgumentParser(description="YOLO Training & Validation Pipeline")
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--name', type=str, required=True, help='Unique run name')
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))

    model_path = os.path.join(project_root, "models", "yolo11n.pt")
    model = YOLO(model_path)
    
    model.train(
        data=args.data,
        name=args.name,
        exist_ok=True,
        save=True,     
        plots=True,
        device=1, 
        project=os.path.join(project_root, "runs/yolo11n"),

        # Deactivate online augmentation maked by Yolo
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        scale=0.0, degrees=0.0, translate=0.0,
        fliplr=0.0, flipud=0.0, 
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    train_and_validate()