import albumentations as A
import cv2
import os
from pathlib import Path
from tqdm import tqdm

base_path = Path("/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/datasets_augmented/")
total_images = 0

transform = A.Compose(
    [
        A.RandomCrop(width=500, height=500, p=1.0), 
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ],
    bbox_params=A.BboxParams(
        format='yolo',
        min_area=2500, 
        min_visibility=0.3,
        label_fields=['class_labels']
    )
)

def read_yolo_label(label_path):
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            class_labels.append(int(float(parts[0])))
            # x_center, y_center, width, height
            bboxes.append([float(x) for x in parts[1:]])
    return bboxes, class_labels

def save_yolo_label(save_path, bboxes, class_labels):
    with open(save_path, 'w') as f:
        for bbox, label in zip(bboxes, class_labels):
            # Format: class x_c y_c w h
            line = f"{label} {' '.join(f'{x:.6f}' for x in bbox)}\n"
            f.write(line)

# itera pelos datasets
for dataset_path in base_path.iterdir():
    train_path = dataset_path / "train"
    images_dir = train_path / "images"
    labels_dir = train_path / "labels"

    print(f"Aumentando dados para {dataset_path.name}/{train_path.name}")

    for img_path in tqdm(images_dir.iterdir()):
        # encontra label correspondente
        label_name = img_path.stem + ".txt"
        label_path = labels_dir / label_name
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        bboxes, class_labels = read_yolo_label(label_path)

        try:
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = transformed['image']
            aug_bboxes = transformed['bboxes']
            aug_labels = transformed['class_labels']
            
            # se crop removeu todas as labels, pula
            if len(aug_bboxes) == 0:
                continue 

            new_filename = f"{img_path.stem}_aug{img_path.suffix}"
            new_img_path = images_dir / new_filename
            new_label_path = labels_dir / f"{img_path.stem}_aug.txt"

            # convertendo de volta para opencv salvar corretamente
            cv2.imwrite(str(new_img_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            save_yolo_label(new_label_path, aug_bboxes, aug_labels)
            total_images += 1

        except Exception as e:
            print(f"Error augmenting {img_path.name}: {e}")
        
print(f"Completo. Total de imagens geradas: {total_images}")