import re
import shutil
import numpy as np
from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def extract_core_id(filename):
    """
    Extract image identifier (ex: 'processed-DJI_20240828141944_0061').
    Funciona com nomes: 
    - processed-DJI_20240828141944_0061_D.JPG
    - processed-DJI_20240828141944_0061_RGB_NDVI.TIF
    - processed-DJI_20240828141944_0061_NDVI.TIF
    """
    name = filename.split('.')[0]

    for suffix in ['_D_FUSED', '_RGB_NDVI', '_RGB_NDRE', '_FUSED_NDVI', '_FUSED_NDRE', '_NDVI', '_NDRE', '_D']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    match = re.search(r"(processed-DJI_\d+_\d+)", name)
    if match:
        return match.group(1)
    return None


def build_file_map(dataset_path):
    img_folder = dataset_path / "images"
    file_map = {}

    if not img_folder.exists():
        return file_map

    for f in img_folder.iterdir():
        if f.suffix.lower() in ['.tif', '.jpg', '.jpeg', '.png']:
            core_id = extract_core_id(f.name)
            if core_id:
                file_map[core_id] = f

    return file_map


def process_dataset(dataset_folder, split_ids, split_name):
    file_map = build_file_map(dataset_folder)

    dst_root = out_base / dataset_folder.name
    dst_split_img = dst_root / split_name / 'images'
    dst_split_lbl = dst_root / split_name / 'labels'

    dst_split_img.mkdir(parents=True, exist_ok=True)
    dst_split_lbl.mkdir(parents=True, exist_ok=True)

    src_lbl_folder = dataset_folder / "labels"

    count = 0
    for core_id in split_ids:
        if core_id not in file_map:
            continue

        src_img = file_map[core_id]

        src_lbl = None
        for lbl_file in src_lbl_folder.glob(f"{core_id}*"):
            if lbl_file.suffix == '.txt':
                src_lbl = lbl_file
                break

        if src_lbl is None or not src_lbl.exists():
            master_lbl_folder = (dataset_folder.parent / "fused" / "labels")
            for lbl_file in master_lbl_folder.glob(f"{core_id}*"):
                if lbl_file.suffix == '.txt':
                    src_lbl = lbl_file
                    break

        if src_lbl and src_lbl.exists():
            shutil.copy(src_img, dst_split_img / src_img.name)
            shutil.copy(src_lbl, dst_split_lbl / src_lbl.name)
            count += 1

    return count


if __name__ == "__main__":
    base_root = Path(
        "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset")
    out_base = Path(
        "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited")

    num_classes = 16

    val_ratio = 0.10
    test_ratio = 0.10

    dataset_types = [
        "fused", "ndre", "ndvi", "rgb",
        "rgb-ndvi", "rgb-ndre", "fused-ndvi", "fused-ndre"
    ]

    available_datasets = [
        base_root / ds_type for ds_type in dataset_types
        if (base_root / ds_type / "images").exists()
    ]

    master_dataset = base_root / "fused"
    ref_labels_dir = master_dataset / "labels"

    core_ids_list = []
    labels_matrix = []

    for txt_path in ref_labels_dir.glob("*.txt"):
        core_id = extract_core_id(txt_path.name)
        if not core_id:
            continue

        class_presence = np.zeros(num_classes)

        with open(txt_path, 'r') as f:
            for linha in f:
                linha = linha.strip()
                if linha:
                    class_id = int(linha.split()[0])
                    if class_id < num_classes:
                        class_presence[class_id] = 1

        core_ids_list.append(core_id)
        labels_matrix.append(class_presence)

    X = np.array(core_ids_list)
    Y = np.array(labels_matrix)

    # First split: Train x (Val+Test)
    msss_train = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=(val_ratio + test_ratio), random_state=42)
    train_idx, rest_idx = next(msss_train.split(X, Y))

    X_train_core, Y_train = X[train_idx], Y[train_idx]
    X_rest, Y_rest = X[rest_idx], Y[rest_idx]

    # Second split: Val vs Test
    test_fraction = test_ratio / (val_ratio + test_ratio)
    msss_val = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_fraction, random_state=42)
    val_idx, test_idx = next(msss_val.split(X_rest, Y_rest))

    val_ids = set(X_rest[val_idx])
    test_ids = set(X_rest[test_idx])
    train_ids = set(X_train_core)

    data_yaml_src = base_root / "data.yaml"

    for ds in available_datasets:
        n_train = process_dataset(ds, train_ids, 'train')
        n_val = process_dataset(ds, val_ids, 'val')
        n_test = process_dataset(ds, test_ids, 'test')
        dst_root = out_base / ds.name
        if data_yaml_src.exists():
            with open(data_yaml_src, 'r') as f:
                content = f.read()

            content = content.replace('../train/images', 'train/images')
            content = content.replace('../valid/images', 'val/images')
            content = content.replace('../test/images', 'test/images')

            abs_path = dst_root.resolve()

            if 'path:' in content:
                content = re.sub(
                    r'path:.*?\n', f'path: {abs_path}\n', content, count=1)
            else:
                content = f"path: {abs_path}\n" + content

            dst_yaml = dst_root / "data.yaml"
            with open(dst_yaml, 'w') as f:
                f.write(content)
