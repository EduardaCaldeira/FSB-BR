export OMP_NUM_THREADS=2

DATASET_PATH="/data/mcaldeir/exit_entry/CelebAMask-HQ/CelebA-HQ-img-test"
GT_PATH="/data/mcaldeir/exit_entry/CelebAMask-HQ/CelebAMask-unified-masks"
BOXES_PATH="/data/mcaldeir/exit_entry/CelebAMask-HQ/CelebA-HQ-img-test_bb.txt"
OUT_SEG_DIR="/data/mcaldeir/exit_entry/CelebAMask-HQ/segmentation/sam_full_no_ctr"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port=38129 SAM/eval_sam.py \
    --boxes_path $BOXES_PATH \
    --dataset $DATASET_PATH \
    --gt $GT_PATH \
    --method full_no_ctr \
    --out_seg_dir $OUT_SEG_DIR