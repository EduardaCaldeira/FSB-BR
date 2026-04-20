#!/bin/bash

NETWORKS=("danet-ps" "fastscnn-ps" "fcn-ps" "bisenet-ps") #"fpn-ps-224" "bisenet-ps" "danet-ps" "fastscnn-ps"
DATASETS=("frgc")

for DATASET in "${DATASETS[@]}"; do
    for NETWORK in "${NETWORKS[@]}"; do
        CONFIG="local_configs/easyportrait_experiments_v2/$NETWORK/$NETWORK.py"
        CHECKPOINT="local_paths/$NETWORK.pth"
        OUT_FILE="../../../../../../data/mcaldeir/exit_entry/$DATASET/original/${DATASET}_$NETWORK"

        if [[ "$DATASET" == "IJBC" ]]; then
            DATASET_PATH="../../../../../../data/mcaldeir/IJB_release/IJBC/loose_crop"
        else
            DATASET_PATH="../../../../../../data/mcaldeir/exit_entry/$DATASET/original/${DATASET}"
        fi

        PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
        python $(dirname "$0")/demo/image_demo.py \
            "$DATASET_PATH" \
            "$CONFIG" \
            "$CHECKPOINT" \
            --device cpu \
            --out_dir $OUT_FILE 
    done
done