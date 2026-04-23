import sys
import os

import gc
import torch
import argparse
from rfdetr import RFDETRSmall

def train_and_validate():
    parser = argparse.ArgumentParser(description="Large RT-DETR Training & Validation Pipeline")
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--name', type=str, required=True, help='Unique run name')
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))

    model = RFDETRSmall(pretrain_weights=None, num_classes=16)
    
    model.train(
        dataset_dir=args.data,
        name=args.name,
        exist_ok=True,
        save=True,     
        plots=True,
        early_stopping=True,
        batch_size=4,
        grad_accum_steps=12,
        epochs=100,
        output_dir=os.path.join(project_root, "runs/rfdetr-small"),  
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    train_and_validate()