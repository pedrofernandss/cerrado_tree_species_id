from ultralytics import RTDETR
import argparse
import torch
import gc
import sys
import os

user_site = "/mnt/sdb-seagate/graduacao/python_userbase/ana_pedro/lib/python3.9/site-packages"
if user_site not in sys.path:
    sys.path.insert(0, user_site)


def train_and_validate():
    parser = argparse.ArgumentParser(description="rtdetr Training Pipeline")
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data.yaml')
    parser.add_argument('--name', type=str, required=True,
                        help='Unique run name')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))

    model = RTDETR("rtdetr-l.pt")

    model.train(
        data=args.data,
        name=args.name,
        exist_ok=True,
        save=True,
        plots=True,
        device=1,
        batch=8,
        epochs=50,
        seed=args.seed,
        pretrained=False,
        project=os.path.join(project_root, "runs",
                             f"seed{args.seed}", "rtdetr-l")
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    train_and_validate()
