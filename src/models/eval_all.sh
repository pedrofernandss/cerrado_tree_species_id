#!/bin/bash

BASE="/mnt/sdb-seagate/graduacao/home/ana_pedro/projetos/cerrado_tree_identifier/src/models"

MODELOS=(
    # "yolo5n"
    # "yolo5s"
    # "yolov8n"
    # "yolov8s"
    # "yolo11n"
    "yolo11s"
    # "rtdetr-l"
)

SEEDS=(1 2 3)

for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Evaluating seed: $SEED"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    for MODELO in "${MODELOS[@]}"; do
        echo "------------------------------------------"
        echo "Evaluating: $MODELO | seed: $SEED"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "------------------------------------------"

        bash "$BASE/$MODELO/eval/eval.sh" "$SEED"

        echo "Complete: $MODELO | seed: $SEED"
        sleep 10
    done
done

echo "=========================================="
echo "All evaluations are completed!"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="