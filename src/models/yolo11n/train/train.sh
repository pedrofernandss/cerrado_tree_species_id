#!/bin/bash

SEED=${1:-1}

LOG_DIR="/mnt/sdb-seagate/graduacao/logs/ana_pedro/train/seed${SEED}/yolo11n"
mkdir -p "$LOG_DIR"

PYTHON="/mnt/sdb-seagate/graduacao/home/ana_pedro/projetos/cerrado_tree_identifier/.venv/bin/python3"

cd /mnt/sdb-seagate/graduacao/home/ana_pedro/projetos/cerrado_tree_identifier/src/models/yolo11n/train/

datasets=(
    "fused" "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/fused/data.yaml"
    "rgb"   "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/rgb/data.yaml"
    "ndre"  "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/ndre/data.yaml"
    "ndvi"  "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/ndvi/data.yaml"
)

for ((i=0; i<${#datasets[@]}; i+=2)); do
    NAME="${datasets[$i]}"
    YAML="${datasets[$i+1]}"
    echo "Iniciando: $NAME | seed: $SEED"
    $PYTHON train.py --data "$YAML" --name "${NAME}_seed${SEED}" --seed "$SEED" > "$LOG_DIR/$NAME.log" 2>&1
    echo "Finalizado: $NAME"
    sleep 10
done

echo "yolo11n seed=$SEED concluído!"