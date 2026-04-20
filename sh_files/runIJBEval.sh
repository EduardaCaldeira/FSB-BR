#!/usr/bin/env bash

NETWORKS=("IJBC_fpn-ps-224" "IJBC_segformer-ps-224" "IJBC_fastscnn-ps" "IJBC_fcn-ps" "IJBC_danet-ps" "IJBC_sam_full_no_ctr" "IJBC_bisenet-ps") # "IJBC_fpn-ps-224" "IJBC_segformer-ps-224" "IJBC_fastscnn-ps" "IJBC_fcn-ps" "IJBC_danet-ps" "IJBC_sam_full_no_ctr" "IJBC_bisenet-ps"
FR_MODELS=("ms1mv3_arcface_r100_fp16") # "ElasticCos" "ms1mv3_arcface_r100_fp16" "TransFace-L" "SwinFace"

for NETWORK in "${NETWORKS[@]}"; do
    for FR_MODEL in "${FR_MODELS[@]}"; do
        if [[ "$FR_MODEL" == "SwinFace" || "$FR_MODEL" == "TransFace-L" ]]; then
            MPATH="/home/mcaldeir/projects/exit_entry/FR_models/$FR_MODEL.pt"
            if [[ "$FR_MODEL" == "SwinFace" ]]; then
                NET="swin_t"
            else
                NET="ViT-L"
            fi
        else
            MPATH="/home/mcaldeir/projects/exit_entry/FR_models/$FR_MODEL.pth"
            NET="iresnet100"
        fi

        if [ "$NETWORK" == "IJBC" ]; then
            IMAGE_PATH="/data/mcaldeir/IJB_release/IJBC/loose_crop"
        else
            IMAGE_PATH="/data/mcaldeir/exit_entry/IJBC/original/$NETWORK"
        fi

        OUTPUT="/data/mcaldeir/exit_entry/IJBC/original/FR/$NETWORK/$FR_MODEL"
        CUDA_VISIBLE_DEVICES=0 python eval_ijbc.py \
            --model-prefix $MPATH \
            --general_path "/data/mcaldeir/IJB_release/IJBC" \
            --image-path $IMAGE_PATH \
            --job $OUTPUT \
            --target IJBC \
            --network $NET
    done
done