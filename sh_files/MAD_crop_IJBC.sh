export OMP_NUM_THREADS=2

NETWORKS=("bisenet-ps") #"fpn-ps-224" "segformer-ps-224"

for NETWORK in "${NETWORKS[@]}"; do
    CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --nnodes=1 \
    --node_rank=0 --master_addr="127.0.0.1" --master_port=14782 MAD_crop_IJBC.py \
        --boxes_path /data/mcaldeir/IJB_release/IJBC/meta/ijbc_name_box_score_5pts.txt \
        --network $NETWORK
done