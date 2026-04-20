export OMP_NUM_THREADS=2

NETWORKS=("sam_bb_extra") #"fpn-ps-224" "segformer-ps-224" "sam_bb"
DATASETS=("frgc")

for DATASET in "${DATASETS[@]}"; do
    for NETWORK in "${NETWORKS[@]}"; do
        if [[ "$NETWORK" == "none" ]]; then
            CSV_DIR="/data/mcaldeir/exit_entry/$DATASET/MAD_crop/$DATASET.csv"
            SCORE_FILE_NAME="$DATASET.csv"
        else
            CSV_DIR="/data/mcaldeir/exit_entry/$DATASET/MAD_crop/${DATASET}_${NETWORK}.csv"
            SCORE_FILE_NAME="${DATASET}_${NETWORK}.csv"
        fi

        python CR-FIQA/getQualityScore.py \
        --csv_dir $CSV_DIR \
        --model_path "CR-FIQA" \
        --backbone "iresnet100" \
        --model_id "181952" \
        --score_file_name $SCORE_FILE_NAME \
        --color_channel "RGB" 
    done
done