export OMP_NUM_THREADS=2

DATASET="frgc"

if [[ "$DATASET" == "IJBC" ]]; then
    FOLDER="/data/mcaldeir/IJB_release/IJBC/loose_crop"
    OUT_DIR="/data/mcaldeir/exit_entry/IJBC/original/IJBC"
    BOXES_PATH="/data/mcaldeir/IJB_release/IJBC/meta/ijbc_name_box_score_5pts.txt"
else
    FOLDER="/data/mcaldeir/exit_entry/$DATASET/original/$DATASET"
    OUT_DIR="/data/mcaldeir/exit_entry/$DATASET/original/$DATASET"
    BOXES_PATH="/data/mcaldeir/exit_entry/$DATASET/${DATASET}_bb.txt"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port=21229 SAM/run_sam.py \
    --boxes_path $BOXES_PATH \
    --folder $FOLDER \
    --out_dir $OUT_DIR \
    --method bb_extra