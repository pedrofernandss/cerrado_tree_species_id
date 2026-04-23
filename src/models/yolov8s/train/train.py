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
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))

    model = YOLO('yolov8s.pt')

    model.train(
        data=args.data,
        name=args.name,
        task='detect',
        exist_ok=True,
        save=True,
        plots=True,
        device=1,
        epochs=100,
        batch=12,
        imgsz=640,
        pretrained=False,
        seed=args.seed,
        project=os.path.join(project_root, "runs", f"seed{args.seed}", "yolov8s")
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    train_and_validate()