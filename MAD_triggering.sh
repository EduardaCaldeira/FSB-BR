# Add both root and src to PYTHONPATH to solve the "utils.utils" error
export PYTHONPATH=$PYTHONPATH:/home/mcaldeir/projects/exit_entry:/home/mcaldeir/projects/exit_entry/src

export TORCH_CUDA_ARCH_LIST="12.0" 
export CUDA_MODULE_LOADING=LAZY
export TORCH_CUDA_RUN_EVENT_SYNC=1

export OMP_NUM_THREADS=2
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_HOME="./cache/"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

backbone_size="ViT-L/14"
training_type="test_clip"
dataset_names=("frgc_sam_bb_extra")

for dataset_name in "${dataset_names[@]}"; do
    CUDA_VISIBLE_DEVICES=5 python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=13924 \
    src/config.py \
    --debug=True \
    --backbone_size="$backbone_size" \
    --dataset_name="$dataset_name" \
    --model_name="MADPromptS" \
    --training_type="$training_type" \
    --eval_method="eval"
done