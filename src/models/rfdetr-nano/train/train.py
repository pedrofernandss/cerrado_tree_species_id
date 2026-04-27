import os
import gc
import torch
import argparse
from rfdetr import RFDETRNano


def train_and_validate():
    parser = argparse.ArgumentParser(description="RF-DETR Nano Training Pipeline (no pretrained)")
    parser.add_argument('--data', type=str, required=True, help='Path to dataset dir')
    parser.add_argument('--name', type=str, required=True, help='Unique run name')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))

    model = RFDETRNano(pretrain_weights=None, num_classes=16)

    model.train(
        dataset_dir=args.data,
        output_dir=os.path.join(project_root, "runs", f"seed{args.seed}", "rfdetr-nano", args.name),
        epochs=50,
        batch_size=4,
        grad_accum_steps=12,
        early_stopping=True,
        early_stopping_patience=20,
        run_test=True,
        devices=1,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    train_and_validate()