#!/bin/bash

LOG_DIR="/mnt/sdb-seagate/graduacao/logs/ana_pedro/evaluations/yolo11n"
mkdir -p "$LOG_DIR"

DATASET_BASE_NO_AUG="/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/datasets_no_augmentation"
DATASET_BASE_AUG="/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado/datasets_augmented"

datasets=(
    "fused" "index:FUSED"
    "fused-ndre" "index:FUSED+NDRE"
    "fused-ndvi" "index:FUSED+NDVI"
    "ndre" "index:NDRE"
    "ndvi" "index:NDVI"
    "rgb" "index:RGB"
    "rgb-ndre" "index:RGB+NDRE"
    "rgb-ndvi" "index:RGB+NDVI"
)

for ((i=0; i<${#datasets[@]}; i+=2)); do
    NAME="${datasets[$i]}"
    TAG="${datasets[$i+1]}"

    YAML_NO_AUG="$DATASET_BASE_NO_AUG/$NAME/data.yaml"
    MODEL_NO_AUG="../../../runs/yolo11n/no_augmentation/adjust-dataset/$NAME/weights/best.pt"
    RUN_NAME_NO_AUG="test_no_augmented_$NAME"

    if [ -f "$MODEL_NO_AUG" ]; then
        echo "Lançando Teste Sem Data Augmentation : $NAME"
        find "$DATASET_BASE_NO_AUG/$NAME" -name "*.cache" -delete
        python3 eval.py \
            --model "$MODEL_NO_AUG" \
            --data "$YAML_NO_AUG" \
            --name "$RUN_NAME_NO_AUG" \
            --aug "no" \
            --tags "$TAG" > "$LOG_DIR/$RUN_NAME_NO_AUG.log" 2>&1
    else
        echo "ERRO: Modelo não encontrado para $NAME em $MODEL_NO_AUG"
    fi

    YAML_AUG="$DATASET_BASE_AUG/$NAME/data.yaml"
    MODEL_AUG="../../../runs/yolo11n/augmented/$NAME/weights/best.pt"
    RUN_NAME_AUG="test_augmented_$NAME"

    if [ -f "$MODEL_AUG" ]; then
        echo "Lançando Teste Augmented: $NAME"
        find "$DATASET_BASE_AUG/$NAME" -name "*.cache" -delete
        python3 eval.py \
            --model "$MODEL_AUG" \
            --data "$YAML_AUG" \
            --name "$RUN_NAME_AUG" \
            --aug "yes" \
            --tags "$TAG" > "$LOG_DIR/$RUN_NAME_AUG.log" 2>&1
    else
        echo "ERRO: Modelo Augmented não encontrado para $NAME em $MODEL_AUG"
    fi

    echo "Progresso: $NAME finalizado (Baseline & Augmented)."
    sleep 10 
done

echo "Processo de avaliação finalizado! Confira os resultados no MLflow."