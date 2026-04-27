#!/bin/bash

SEED=${1:-1}

LOG_DIR="/mnt/sdb-seagate/graduacao/logs/ana_pedro/train/seed${SEED}/rfdetr-nano"
mkdir -p "$LOG_DIR"

PYTHON="/mnt/sdb-seagate/graduacao/home/ana_pedro/.conda/envs/rfdetr/bin/python"

cd /mnt/sdb-seagate/graduacao/home/ana_pedro/projetos/cerrado_tree_identifier/src/models/rfdetr-nano/train/

datasets=(
    "fused" "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/fused"
    "rgb"   "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/rgb"
    "ndre"  "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/ndre"
    "ndvi"  "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/ndvi"
)

for ((i=0; i<${#datasets[@]}; i+=2)); do
    NAME="${datasets[$i]}"
    DATA="${datasets[$i+1]}"
    echo "Starting: $NAME | seed: $SEED"
    $PYTHON train.py --data "$DATA" --name "${NAME}_seed${SEED}" --seed "$SEED" > "$LOG_DIR/$NAME.log" 2>&1
    echo "Finishing: $NAME"
    sleep 10
done

echo "rfdetr-nano seed=$SEED completed!"