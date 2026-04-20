export OMP_NUM_THREADS=2

NETWORKS=("fastscnn-ps" "fcn-ps")

for NETWORK in "${NETWORKS[@]}"; do
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 \
    --node_rank=0 --master_addr="127.0.0.1" --master_port=48321 clean_frgc.py \
        --file_list /data/mcaldeir/exit_entry/frgc/remove_files.txt \
        --in_dir /data/mcaldeir/exit_entry/frgc \
        --network $NETWORK
done