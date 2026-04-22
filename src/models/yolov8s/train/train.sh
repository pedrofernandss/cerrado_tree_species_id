#!/bin/bash

LOG_DIR="/mnt/sdb-seagate/graduacao/logs/ana_pedro/train/yolo8s"
mkdir -p "$LOG_DIR"

datasets=(
    "fused" "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/fused/data.yaml" 
    "fused-ndre" "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/fused-ndre/data.yaml" 
    "fused-ndvi"  "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/fused-ndvi/data.yaml" 
    "ndre"  "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/ndre/data.yaml" 
    "ndvi"  "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/ndvi/data.yaml" 
    "rgb"  "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/rgb/data.yaml" 
    "rgb-ndre"  "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/rgb-ndre/data.yaml" 
    "rgb-ndvi" "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited/rgb-ndvi/data.yaml"
)

for ((i=0; i<${#datasets[@]}; i+=2)); do
    NAME="${datasets[$i]}"
    YAML="${datasets[$i+1]}"

    echo "----------------------------------------------------------"
    echo "Iniciando treinamendo do $NAME com o arquivo $YAML"
    echo "----------------------------------------------------------"

    python3 train.py --data "$YAML" --name "$NAME" > "$LOG_DIR/$NAME.log" 2>&1

    echo "Finalizado: $NAME"
    sleep 10 
done

echo "Todos os 8 treinamentos foram concluídos!"