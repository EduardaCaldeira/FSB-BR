export OMP_NUM_THREADS=2

NETWORKS=("none" "fpn-ps-224")
DATASETS=("feret")

for DATASET in "${DATASETS[@]}"; do
    for NETWORK in "${NETWORKS[@]}"; do
        CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --nnodes=1 \
        --node_rank=0 --master_addr="127.0.0.1" --master_port=48321 retinaface_alignment.py \
            --dataset $DATASET \
            --network $NETWORK
    done
done