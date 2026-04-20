#!/bin/bash
export OMP_NUM_THREADS=2

ROOT=/data/mcaldeir/exit_entry/frgc/MAD_crop
FILE_LIST=("frgc_sam_bb_extra")

#ROOT=/data/mcaldeir/FaceMAD/Protocols
#FILE_LIST=("SYN_MAD22")

for FILE in "${FILE_LIST[@]}"; do
    python supervised_MAD/main.py \
    --test_csv_path "$ROOT/$FILE.csv" \
    --model_path 'supervised_MAD/mixfacenet_SMDD.pth' \
    --is_train False \
    --is_test True \
    --output_dir "/igd/a1/home/mcaldeir/MAD_exit_entry/MixFaceNet-MAD/$FILE.csv" \
    --method "eval"
done