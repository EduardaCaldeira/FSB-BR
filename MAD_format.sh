#!/bin/bash

NETWORKS=("sam_bb_extra")
DATASETS=("frgc")

for DATASET in "${DATASETS[@]}"; do
    for NETWORK in "${NETWORKS[@]}"; do
        IMAGE_DIR="/data/mcaldeir/exit_entry/${DATASET}/MAD_crop/${DATASET}"
        OUTPUT_FILE="../../../../../../data/mcaldeir/exit_entry/${DATASET}/MAD_crop/${DATASET}_${NETWORK}.csv"

        # Run the script
        python MAD_format.py \
            --image_dir $IMAGE_DIR \
            --output_excel $OUTPUT_FILE \
            --network $NETWORK
    done
done