import os
import gc
import argparse
import mlflow
from ultralytics import YOLO
import torch

os.environ["ULTRALYTICS_MLFLOW"] = "True"

def evaluate():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    reports_dir = os.path.join(project_root, 'reports', 'evaluations')

    mlflow.set_tracking_uri(f"file://{os.path.join(project_root, 'mlruns')}")
    mlflow.set_experiment("cerrado_tree_identifier")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path for best.pt')
    parser.add_argument('--data', type=str, required=True, help='Path for data.yaml')
    parser.add_argument('--name', type=str, required=True, help='MLflow run name')
    parser.add_argument('--tags', type=str, required=True)
    parser.add_argument('--aug', type=str, required=True, help='yes ou no')
    args = parser.parse_args()

    model = YOLO(args.model)

    with mlflow.start_run(run_name=args.name):
        mlflow.set_tag("model_version", "yolov11s")
        mlflow.set_tag("phase", "evaluation")
        mlflow.set_tag("is_augmented", args.aug)

        if args.tags:
            tags_list = args.tags.split(',')
            for tag in tags_list:
                key, value = tag.split(':')
                mlflow.set_tag(key, value)

        results = model.val(
            data=args.data,
            split='test',
            imgsz=1000,
            batch=8,
            project=reports_dir,
            name=args.name,
            save=True,
            plots=True
        )

        metrics = results.results_dict
        for k, v in metrics.items():
            clean_name = f"test_{k.replace('metrics/', '').replace('(', '').replace(')', '').replace(' ', '_')}"
            mlflow.log_metric(clean_name, float(v))
    
    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    evaluate()