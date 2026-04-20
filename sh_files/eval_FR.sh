export OMP_NUM_THREADS=2

NETWORKS=("sam_bb_extra") # "none" "fastscnn-ps" "fcn-ps" "danet-ps" "bisenet-ps" "sam_full_no_ctr"
DATASETS=("frgc")
FR_MODELS=("ElasticCos" "ms1mv3_arcface_r100_fp16" "TransFace-L" "SwinFace") # "ElasticCos" "ms1mv3_arcface_r100_fp16" "TransFace-L" "SwinFace"

for DATASET in "${DATASETS[@]}"; do
    for NETWORK in "${NETWORKS[@]}"; do
        for FR_MODEL in "${FR_MODELS[@]}"; do
            CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 \
            --node_rank=0 --master_addr="127.0.0.1" --master_port=21812 efficient_eval_FR.py \
            --dataset $DATASET \
            --segmentation_network $NETWORK \
            --FR_network $FR_MODEL \
            --method dist
        done
    done
done