#!/bin/bash

SEED=${1:-1}

LOG_DIR="/mnt/sdb-seagate/graduacao/logs/ana_pedro/evaluations/seed${SEED}/yolov8n"
mkdir -p "$LOG_DIR"

PYTHON="/mnt/sdb-seagate/graduacao/home/ana_pedro/projetos/cerrado_tree_identifier/.venv/bin/python3"
DATASET="/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/dataset_splited"
RUNS="/mnt/sdb-seagate/graduacao/home/ana_pedro/projetos/cerrado_tree_identifier/runs"

cd /mnt/sdb-seagate/graduacao/home/ana_pedro/projetos/cerrado_tree_identifier/src/models/yolov8n/eval/

datasets=("fused" "rgb" "ndre" "ndvi")

for NAME in "${datasets[@]}"; do
    YAML="$DATASET/$NAME/data.yaml"
    MODEL="$RUNS/seed${SEED}/yolov8n/${NAME}_seed${SEED}/weights/best.pt"

    if [ -f "$MODEL" ]; then
        echo "Avaliando: $NAME | seed: $SEED"
        find "$DATASET/$NAME" -name "*.cache" -delete
        $PYTHON eval.py --model "$MODEL" --data "$YAML" --seed "$SEED" --model_name "yolov8n" > "$LOG_DIR/$NAME.log" 2>&1
        echo "Finishing: $NAME"
    else
        echo "ERRO: Modelo não encontrado: $MODEL"
    fi
    sleep 5
done

echo "yolov8n seed=$SEED avaliação concluída!"