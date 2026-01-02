#!/bin/bash

LOG_DIR="/mnt/sdb-seagate/graduacao/logs/ana_pedro/yolo5s/no_augmentation"
mkdir -p "$LOG_DIR"

datasets=(
    "fused" "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/datasets_no_augmentation/fused/data.yaml" "index:FUSED"
    "fused-ndre" "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/datasets_no_augmentation/fused-ndre/data.yaml" "index:FUSED+NDRE"
    "fused-ndvi"  "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/datasets_no_augmentation/fused-ndvi/data.yaml" "index:FUSED+NDVI"
    "ndre"  "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/datasets_no_augmentation/ndre/data.yaml" "index:NDRE"
    "ndvi"  "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/datasets_no_augmentation/ndvi/data.yaml" "index:NDVI"
    "rgb"  "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/datasets_no_augmentation/rgb/data.yaml" "index:RGB"
    "rgb-ndre"  "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/datasets_no_augmentation/rgb-ndre/data.yaml" "index:RGB+NDRE"
    "rgb-ndvi" "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/datasets_no_augmentation/rgb-ndvi/data.yaml" "index:RGB+NDVI"
)

for ((i=0; i<${#datasets[@]}; i+=3)); do
    NAME="${datasets[$i]}"
    YAML="${datasets[$i+1]}"
    TAGS="${datasets[$i+2]}"

    echo "----------------------------------------------------------"
    echo "Iniciando treinamendo do $NAME com o arquivo $YAML"
    echo "----------------------------------------------------------"

    python3 train.py --data "$YAML" --name "$NAME" --tags "$TAGS" > "$LOG_DIR/$NAME.log" 2>&1

    echo "Finalizado: $NAME"
    sleep 10 
done

echo "Todos os 8 treinamentos foram concluídos!"