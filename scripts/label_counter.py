import os
from collections import Counter

labels_folder = "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/fotos-rotuladas/fused/labels"

classes = {}

classes_counter = Counter()
total_bounding_boxes = 0

for file in os.listdir(labels_folder):
    if file.endswith(".txt"):
        complete_path = os.path.join(labels_folder, file)

        with open(complete_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue  # Skip empty line
            
                class_id = int(line.split()[0])
                classes_counter[class_id] += 1
                total_bounding_boxes += 1

print("-" * 40)
print(f"{'ID':<5} | {'CLASS':<15} | {'QUANTITY'}")
print("-" * 40)

for class_id in sorted(classes_counter.keys()):
    quantidade = classes_counter[class_id]
    nome = classes.get(class_id, f"Classe {class_id}")
    print(f"{class_id:<5} | {nome:<15} | {quantidade}")

print("-" * 40)
print(f"NUMBER OF TREES (Bounding Boxes): {total_bounding_boxes}")
print(f"NUMBER OF IMAGES: {len([f for f in os.listdir(labels_folder) if f.endswith('.txt')])}")