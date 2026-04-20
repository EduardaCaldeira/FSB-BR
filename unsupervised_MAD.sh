#!/bin/bash
export OMP_NUM_THREADS=2

ROOT=/data/mcaldeir/exit_entry/frgc/MAD_crop
FILE_LIST=("frgc_sam_bb_extra")

for FILE in "${FILE_LIST[@]}"; do
    CUDA_VISIBLE_DEVICES=2 python3 -m torch.distributed.launch \
        --nproc_per_node=1 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr="localhost" \
        --master_port=2231 \
        unsupervised_MAD/unsupervised_MAD.py \
        --test_csv="$ROOT/$FILE.csv" \
        --model_path="unsupervised_MAD/casia_smdd.pth" \
        --output_path="/igd/a1/home/mcaldeir/MAD_exit_entry/SPL/$FILE.csv" \
        --method="eval"
done