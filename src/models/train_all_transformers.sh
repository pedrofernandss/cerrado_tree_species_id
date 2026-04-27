#!/bin/bash

BASE="/mnt/sdb-seagate/graduacao/home/ana_pedro/projetos/cerrado_tree_identifier/src/models"

SEEDS=(1 2 3)

for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Starting seed: $SEED"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    echo "--- rtdetr-l | seed: $SEED ---"
    bash "$BASE/rtdetr-l/train/train.sh" "$SEED"
    echo "Finising rtdetr-l | $(date '+%Y-%m-%d %H:%M:%S')"
    sleep 30

    echo "--- rfdetr-nano | seed: $SEED ---"
    bash "$BASE/rfdetr-nano/train/train.sh" "$SEED"
    echo "Finising rfdetr-nano | $(date '+%Y-%m-%d %H:%M:%S')"
    sleep 15

done

echo "=========================================="
echo "The trainig for all models is complete!"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="