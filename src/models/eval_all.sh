#!/bin/bash

BASE="/mnt/sdb-seagate/graduacao/home/ana_pedro/projetos/cerrado_tree_identifier/src/models"

MODELOS=(
    "yolo5n"
    "yolo5s"
    "yolo8n"
    "yolo8s"
    "yolo11n"
    "yolo11s"
)

SEEDS=(1 2 3)

for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Avaliando seed: $SEED"
    echo "Horário: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    for MODELO in "${MODELOS[@]}"; do
        echo "------------------------------------------"
        echo "Avaliando: $MODELO | seed: $SEED"
        echo "Horário: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "------------------------------------------"

        bash "$BASE/$MODELO/eval/eval.sh" "$SEED"

        echo "Finalizado: $MODELO | seed: $SEED"
        sleep 10
    done
done

echo "=========================================="
echo "Todas as avaliações concluídas!"
echo "Horário: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="