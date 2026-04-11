import random
import re
import traceback
import albumentations as A
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

BASE_PATH = Path(
    "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_stratified/")

PRIMARY_CHANNEL = "rgb"

CHANNEL_FILENAME_MAP = {
    "ndre":      ("_NDRE",       ".TIF"),
    "ndvi":      ("_NDVI",       ".TIF"),
    "fused":     ("_D_FUSED",    ".TIF"),
    "fused-ndre": ("_FUSED_NDRE", ".TIF"),
    "fused-ndvi": ("_FUSED_NDVI", ".TIF"),
    "rgb-ndre":  ("_RGB_NDRE",   ".TIF"),
    "rgb-ndvi":  ("_RGB_NDVI",   ".TIF"),
}

NUM_AUGMENTATIONS = 2
MAX_RETRY_ATTEMPTS = 10

VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

transform = A.ReplayCompose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            scale=(1.0, 1.15),
            rotate=(-15, 15),
            translate_percent=(0.0, 0.0),
            p=1.0,
            border_mode=cv2.BORDER_REFLECT
        ),
    ],
    bbox_params=A.BboxParams(
        format='yolo',
        clip=True,
        min_area=100,
        min_visibility=0.3,
        label_fields=['class_labels']
    )
)

def extract_base_id(rgb_stem):
    return re.sub(r'_D$', '', rgb_stem)


def clamp_boxes(boxes, class_labels):
    """Convert to corners, clamp to [0,1], convert back to YOLO center format."""
    clamped_boxes, clamped_labels = [], []
    for (x_c, y_c, w, h), label in zip(boxes, class_labels):
        x_min = max(x_c - w / 2, 0.0)
        y_min = max(y_c - h / 2, 0.0)
        x_max = min(x_c + w / 2, 1.0)
        y_max = min(y_c + h / 2, 1.0)
        new_w = x_max - x_min
        new_h = y_max - y_min
        if new_w > 1e-4 and new_h > 1e-4:
            clamped_boxes.append([
                (x_min + x_max) / 2,
                (y_min + y_max) / 2,
                new_w,
                new_h
            ])
            clamped_labels.append(label)
    return clamped_boxes, clamped_labels


def read_yolo_label(label_path):
    boxes, class_labels = [], []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_labels.append(int(float(parts[0])))
            boxes.append([float(x) for x in parts[1:]])
    return boxes, class_labels


def save_yolo_label(save_path, boxes, class_labels):
    with open(save_path, 'w') as f:
        for bbox, label in zip(boxes, class_labels):
            line = f"{label} {' '.join(f'{x:.6f}' for x in bbox)}\n"
            f.write(line)


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 3 and img.dtype == np.uint8:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(path, image):
    if image.ndim == 3 and image.dtype == np.uint8:
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(path), image)

primary_images_dir = BASE_PATH / PRIMARY_CHANNEL / "train" / "images"
primary_labels_dir = BASE_PATH / PRIMARY_CHANNEL / "train" / "labels"

if not primary_images_dir.exists():
    raise FileNotFoundError(
        f"Primary images dir not found: {primary_images_dir}")
if not primary_labels_dir.exists():
    raise FileNotFoundError(
        f"Primary labels dir not found: {primary_labels_dir}")

image_paths = sorted([
    p for p in primary_images_dir.iterdir()
    if p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    and '_aug' not in p.stem
])

total_sets = 0

for img_idx, img_path in enumerate(tqdm(image_paths)):
    label_path = primary_labels_dir / (img_path.stem + ".txt")

    if not label_path.exists():
        print(f"  Warning: no label for {img_path.name}, skipping.")
        continue

    try:
        primary_image = read_image(img_path)
        boxes, class_labels = read_yolo_label(label_path)
        boxes, class_labels = clamp_boxes(
            boxes, class_labels)  # fix bad labels

        if not boxes:
            print(
                f"  Warning: no valid boxes after clamping for {img_path.name}, skipping.")
            continue

        base_id = extract_base_id(img_path.stem)

        for aug_i in range(NUM_AUGMENTATIONS):

            replay_data = None
            aug_boxes = None
            aug_labels = None

            for attempt in range(MAX_RETRY_ATTEMPTS):
                seed = img_idx * 1000 + aug_i * 100 + attempt
                random.seed(seed)
                np.random.seed(seed)

                result = transform(image=primary_image,
                                   bboxes=boxes, class_labels=class_labels)

                if len(result['bboxes']) > 0:
                    replay_data = result['replay']
                    aug_labels = result['class_labels']
                    aug_boxes = [list(b) for b in result['bboxes']]
                    break

            if replay_data is None:
                print(
                    f"  Warning: no valid aug found for {img_path.name} aug{aug_i}, skipping.")
                continue

            new_rgb_stem = f"{img_path.stem}_aug{aug_i}"

            save_image(primary_images_dir /
                       f"{new_rgb_stem}{img_path.suffix}", result['image'])
            save_yolo_label(primary_labels_dir /
                            f"{new_rgb_stem}.txt", aug_boxes, aug_labels)

            for channel, (ch_suffix, ch_ext) in CHANNEL_FILENAME_MAP.items():
                channel_images_dir = BASE_PATH / channel / "train" / "images"
                channel_labels_dir = BASE_PATH / channel / "train" / "labels"

                original_filename = f"{base_id}{ch_suffix}{ch_ext}"
                channel_img_path = channel_images_dir / original_filename

                if not channel_img_path.exists():
                    print(
                        f"  Warning: {channel}/{original_filename} not found, skipping.")
                    continue

                channel_image = read_image(channel_img_path)
                replayed = A.ReplayCompose.replay(
                    replay_data,
                    image=channel_image,
                    bboxes=boxes,
                    class_labels=class_labels
                )

                new_channel_stem = f"{base_id}{ch_suffix}_aug{aug_i}"
                save_image(
                    channel_images_dir / f"{new_channel_stem}{ch_ext}",
                    replayed['image']
                )
                save_yolo_label(
                    channel_labels_dir / f"{new_channel_stem}.txt",
                    aug_boxes,
                    aug_labels
                )

            total_sets += 1

    except Exception as e:
        print(f"  Error on {img_path.name}: {e}")
        traceback.print_exc()

print(f"\nDone. Total augmented sets generated: {total_sets}")
