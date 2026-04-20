#!/bin/bash

NETWORKS=("fpn-ps-224" "segformer-ps-224" "bisenet-ps" "danet-ps" "fastscnn-ps" "fcn-ps") #"segformer-ps-224" "bisenet-ps" "danet-ps" "fastscnn-ps" "fcn-ps"

for NETWORK in "${NETWORKS[@]}"; do
    CONFIG="local_configs/easyportrait_experiments_v2/$NETWORK/$NETWORK.py"
    CHECKPOINT="local_paths/$NETWORK.pth"
    DATASET_PATH="/data/mcaldeir/exit_entry/CelebAMask-HQ/CelebA-HQ-img-test"
    GT_PATH="/data/mcaldeir/exit_entry/CelebAMask-HQ/CelebAMask-unified-masks"

    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python $(dirname "$0")/demo/segmentation_eval.py \
        "$DATASET_PATH" \
        "$GT_PATH" \
        "$CONFIG" \
        "$CHECKPOINT" \
        --device cpu 
done
