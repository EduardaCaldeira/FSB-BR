export OMP_NUM_THREADS=2

NETWORKS=("sam_bb_extra")
DATASETS=("frgc")
FR_MODELS=("ElasticCos" "ms1mv3_arcface_r100_fp16" "TransFace-L" "SwinFace")
for DATASET in "${DATASETS[@]}"; do
    for NETWORK in "${NETWORKS[@]}"; do
        CUDA_VISIBLE_DEVICES=4 python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 \
        --node_rank=0 --master_addr="127.0.0.1" --master_port=22122 save_FR_embeddings.py \
        --dataset $DATASET \
        --segmentation_network $NETWORK \
        --FR_networks "${FR_MODELS[@]}"
    done
done