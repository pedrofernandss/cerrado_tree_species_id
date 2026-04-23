#!/bin/bash

BASE="/mnt/sdb-seagate/graduacao/home/ana_pedro/projetos/cerrado_tree_identifier/src/models"

MODELOS=(
    "yolo5n"
    "yolo5s"
    "yolov8n"
    "yolov8s"
    "yolo11n"
    "yolo11s"
)

SEEDS=(1 2 3)

for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Iniciando seed: $SEED"
    echo "Horário: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    for MODELO in "${MODELOS[@]}"; do
        echo "------------------------------------------"
        echo "Iniciando: $MODELO | seed: $SEED"
        echo "Horário: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "------------------------------------------"

        bash "$BASE/$MODELO/train/train.sh" "$SEED"

        echo "Finalizado: $MODELO | seed: $SEED"
        echo "Horário: $(date '+%Y-%m-%d %H:%M:%S')"
        sleep 30
    done
done

echo "=========================================="
echo "Todos os treinamentos concluídos!"
echo "Horário: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="