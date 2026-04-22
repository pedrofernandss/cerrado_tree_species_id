from pathlib import Path

BASE = "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited"

for txt in Path(BASE).rglob("*.txt"):
    lines = txt.read_text().splitlines()
    corrigidas = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        parts[0] = str(int(float(parts[0])))
        corrigidas.append(" ".join(parts))
    txt.write_text("\n".join(corrigidas) + "\n")

print("Labels id transformed to int values!")