import os
import gc
import argparse
from ultralytics import YOLO
import torch

def evaluate():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
    reports_dir = os.path.join(project_root, 'reports', 'evaluations', 'yolo5s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path for best.pt')
    parser.add_argument('--data', type=str, required=True, help='Path for data.yaml')
    args = parser.parse_args()

    model = YOLO(args.model)

    model.val(
        data=args.data,
        split='test',
        task='detect',
        imgsz=640,
        project=reports_dir,
        save=True,
        plots=True
    )
    
    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    evaluate()