#!/bin/bash

LOG_DIR="/mnt/sdb-seagate/graduacao/logs/ana_pedro/train/rfdetr-small"
mkdir -p "$LOG_DIR"

PYTHON="/mnt/sdb-seagate/graduacao/home/ana_pedro/.conda/envs/rfdetr/bin/python"

datasets=(
    "fused" "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/fused" 
    "fused-ndre" "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/fused-ndre" 
    "fused-ndvi"  "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/fused-ndvi" 
    "ndre"  "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/ndre" 
    "ndvi"  "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/ndvi" 
    "rgb"  "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/rgb" 
    "rgb-ndre"  "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/rgb-ndre"
    "rgb-ndvi" "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/rgb-ndvi" 
)

for ((i=0; i<${#datasets[@]}; i+=2)); do
    NAME="${datasets[$i]}"
    DATA="${datasets[$i+1]}"

    echo "----------------------------------------------------------"
    echo "Iniciando treinamento do $NAME"
    echo "----------------------------------------------------------"

    $PYTHON train.py --data "$DATA" --name "$NAME" > "$LOG_DIR/$NAME.log" 2>&1

    echo "Finalizado: $NAME"
    sleep 10 
done

echo "Todos os 8 treinamentos foram concluídos!"