import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

DATASET = "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited"
BASE_NAME = "processed-DJI_20250127101349_0248_D"

RGB_CHECKPOINT   = "/mnt/sdb-seagate/graduacao/home/ana_pedro/projetos/cerrado_tree_identifier/runs/seed1/yolov5s/rgb_seed1/weights/best.pt"
FUSED_CHECKPOINT = "/mnt/sdb-seagate/graduacao/home/ana_pedro/projetos/cerrado_tree_identifier/runs/seed1/yolov5s/fused_seed1/weights/best.pt"

RGB_IMAGE_PATH   = f"{DATASET}/rgb/test/images/{BASE_NAME}.jpg"
FUSED_IMAGE_PATH = f"{DATASET}/fused/test/images/{BASE_NAME}_FUSED.jpg"

INPUT_SIZE = 640
TARGET_LAYER_INDEX = -4
CONF_THRESHOLD = 0.25

OUTPUT_PATH = "../../assets/explainability/gradcam_comparison.png"

class YOLOWrapper(nn.Module):
    """Wraps the Ultralytics YOLO model so it returns a single tensor
    instead of a tuple — required by pytorch_grad_cam."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)

        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs


def build_cam(checkpoint_path):
    model = YOLO(checkpoint_path)
    model.model.eval()
    wrapped = YOLOWrapper(model.model)
    target_layers = [model.model.model[TARGET_LAYER_INDEX]]
    cam = EigenCAM(wrapped, target_layers)
    return model, cam


def get_heatmap(cam, image_path, input_size=INPUT_SIZE):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (input_size, input_size))
    img_float = np.float32(img_rgb) / 255.0

    input_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).float()

    grayscale_cam = cam(input_tensor)[0, :, :]
    overlay = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    return img_rgb, overlay


def get_detections(model, image_path, input_size=INPUT_SIZE):
    """Run inference and draw bounding boxes on the image."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (input_size, input_size))

    results = model.predict(image_path, imgsz=input_size, conf=CONF_THRESHOLD, verbose=False)
    det_img = results[0].plot()  # BGR with boxes drawn
    det_img = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
    det_img = cv2.resize(det_img, (input_size, input_size))
    return det_img

def main():

    rgb_model, rgb_cam = build_cam(RGB_CHECKPOINT)
    rgb_original, rgb_overlay = get_heatmap(rgb_cam, RGB_IMAGE_PATH)
    rgb_detections = get_detections(rgb_model, RGB_IMAGE_PATH)

    fused_model, fused_cam = build_cam(FUSED_CHECKPOINT)
    fused_original, fused_overlay = get_heatmap(fused_cam, FUSED_IMAGE_PATH)
    fused_detections = get_detections(fused_model, FUSED_IMAGE_PATH)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    row1 = [
        (rgb_original, "RGB input"),
        (rgb_detections, "RGB — detections"),
        (rgb_overlay, "RGB — Eigen-CAM"),
    ]
    for ax, (img, title) in zip(axes[0], row1):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    row2 = [
        (fused_original, "Fused (R-G-NIR) input"),
        (fused_detections, "Fused — detections"),
        (fused_overlay, "Fused — Eigen-CAM"),
    ]
    for ax, (img, title) in zip(axes[1], row2):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()