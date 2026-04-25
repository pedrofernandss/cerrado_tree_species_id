import os
import gc
import argparse
from ultralytics import YOLO
import torch

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path for best.pt')
    parser.add_argument('--data', type=str, required=True, help='Path for data.yaml')
    parser.add_argument('--seed', type=int, required=True, help='Seed number')
    parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g. yolov5n)')
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
    reports_dir = os.path.join(project_root, 'reports', 'evaluations', f'seed{args.seed}', args.model_name)

    model = YOLO(args.model)

    model.val(
        data=args.data,
        split='test',
        task='detect',
        imgsz=640,
        project=reports_dir,
        save=True,
        plots=True,
        device=1,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    evaluate()