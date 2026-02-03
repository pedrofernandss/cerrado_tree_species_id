import sys
import os

user_site = "/mnt/sdb-seagate/graduacao/python_userbase/ana_pedro/lib/python3.9/site-packages"
if user_site not in sys.path:
    sys.path.insert(0, user_site) 

import gc
import torch
import argparse
from ultralytics import YOLO

import mlflow

def train_and_validate():
    parser = argparse.ArgumentParser(description="YOLO Training & Validation Pipeline")
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--name', type=str, required=True, help='Unique run name')
    parser.add_argument('--tags', type=str, help='Tags')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=1000)
    args = parser.parse_args()

    os.environ["ULTRALYTICS_MLFLOW"] = "True"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))

    mlflow.set_tracking_uri(f"file://{os.path.join(project_root, 'mlruns')}")
    mlflow.set_experiment("cerrado_tree_identifier")

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name=args.name):
        model_path = os.path.join(project_root, "models", "yolo11s.pt")
        model = YOLO(model_path)

        if args.tags:
            tags_list = args.tags.split(',')
            for tag in tags_list:
                key, value = tag.split(':')
                mlflow.set_tag(key, value)
        
        mlflow.set_tag("model_version", "yolov11s")
        mlflow.set_tag("project_phase", "no_augmentation")
    
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            imgsz=args.imgsz,
            name=args.name,
            exist_ok=True,
            save=True,     
            plots=True,
            device=0,
            project=os.path.join(project_root, "runs/yolo11s/no_augmentation")      
        )

        if results is not None:
            metrics = results.results_dict
            for k, v in metrics.items():
                clean_name = k.replace("metrics/", "")
                clean_name = clean_name.replace("(", "").replace(")", "")
                clean_name = clean_name.replace(" ", "_")
                mlflow.log_metric(clean_name, float(v))

        mlflow.log_artifacts(f"runs/detect/{args.name}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    train_and_validate()