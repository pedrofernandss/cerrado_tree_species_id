import os
from collections import Counter

BASE = "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/fused"

SPLITS = {
    "train": os.path.join(BASE, "train", "labels"),
    "valid": os.path.join(BASE, "valid", "labels"),
    "test":  os.path.join(BASE, "test",  "labels"),
}

CLASS_NAMES = {
    0: '1', 1: '10', 2: '13', 3: '14', 4: '15', 5: '17',
    6: '19', 7: '2', 8: '20', 9: '21', 10: '23', 11: '25',
    12: '4', 13: '5', 14: '7', 15: '9'
}

counters = {split: Counter() for split in SPLITS}
images = {split: 0 for split in SPLITS}

for split_name, labels_folder in SPLITS.items():
    for file in sorted(os.listdir(labels_folder)):
        if not file.endswith(".txt"):
            continue
        if file.endswith("_aug0.txt") or file.endswith("_aug1.txt"):
            continue
        images[split_name] += 1
        with open(os.path.join(labels_folder, file), 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                class_id = int(line.split()[0])
                counters[split_name][class_id] += 1

all_classes = sorted(set(
    k for c in counters.values() for k in c.keys()
))

# Cabeçalho
print(f"{'ID':<5} | {'Species':<10} | {'Train':>7} | {'Valid':>7} | {'Test':>7} | {'Total':>7}")
print("-" * 55)

total_train = total_valid = total_test = 0

for class_id in all_classes:
    nome = CLASS_NAMES.get(class_id, f"class_{class_id}")
    train = counters["train"][class_id]
    valid = counters["valid"][class_id]
    test  = counters["test"][class_id]
    total = train + valid + test
    total_train += train
    total_valid += valid
    total_test  += test
    print(f"{class_id:<5} | {nome:<10} | {train:>7} | {valid:>7} | {test:>7} | {total:>7}")

print("-" * 55)
grand_total = total_train + total_valid + total_test
print(f"{'':5}   {'TOTAL':<10} | {total_train:>7} | {total_valid:>7} | {total_test:>7} | {grand_total:>7}")
print()
print(f"Images — Train: {images['train']} | Valid: {images['valid']} | Test: {images['test']} | Total: {sum(images.values())}")