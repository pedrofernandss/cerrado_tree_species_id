#!/bin/bash

LOG_DIR="/mnt/sdb-seagate/graduacao/logs/ana_pedro/evaluations/yolo8s"
mkdir -p "$LOG_DIR"

DATASET="/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited"

datasets=(
    "fused" 
    "fused-ndre"
    "fused-ndvi" 
    "ndre" 
    "ndvi" 
    "rgb" 
    "rgb-ndre" 
    "rgb-ndvi" 
)

for ((i=0; i<${#datasets[@]}; i+=1)); do
    NAME="${datasets[$i]}"

    YAML="$DATASET/$NAME/data.yaml"
    MODEL="../../../../runs/yolov8s/$NAME/weights/best.pt"
    RUN_NAME="test_$NAME"

    if [ -f "$MODEL" ]; then
        echo "Realizando teste : $NAME"
        find "$DATASET/$NAME" -name "*.cache" -delete
        python3 eval.py \
            --model "$MODEL" \
            --data "$YAML" > "$LOG_DIR/$RUN_NAME.log" 2>&1
    else
        echo "ERRO: Modelo não encontrado para $NAME em $MODEL"
    fi

    echo "Progresso: $NAME finalizado."
    sleep 10 
done

echo "Processo de avaliação finalizado!"