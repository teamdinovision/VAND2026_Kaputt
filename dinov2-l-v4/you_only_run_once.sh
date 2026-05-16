#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export UV_OFFLINE=1
export UV_NO_SYNC=1
export PORT=29527


#==========================================================
# 0: set data root and parquet root here
#    datapath: $DATA_ROOT/data/[train, validation, test]/query-data/crop
#    parquetpath: $PARQUET_ROOT/*.parquet
#==========================================================
export DATA_ROOT='/data/public/dataset/kaputt'
export PARQUET_ROOT='/data/public/dataset/kaputt'
export DATA2_ROOT='/data/public/dataset/kaputt2'


#==========================================================
# 1: train first model with train/validation/test data
#==========================================================
sed -i "s|^data_root =.*|data_root = \"${DATA_ROOT}\"|" configs/large_728_mlp.py
sed -i "s|^parquet_root =.*|parquet_root = \"${PARQUET_ROOT}\"|" configs/large_728_mlp.py
uv run torchrun --nproc_per_node=4 --master-port $PORT \
                train_vit_mm.py configs/large_728_mlp.py \
                --work-dir work_dirs/large_728_mlp


#==========================================================
# 2: clean mislabeled data using model trained in step 1
#==========================================================
uv run python clean_mislabel.py \
            --config configs/large_728_mlp.py \
            --checkpoint work_dirs/large_728_mlp/epoch_10.pth \
            --thresh-up 0.75 --thresh-down 0.3 \
            --output-dir ./data


#==========================================================
# 3: re-train model with clean data 
#==========================================================
sed -i "s|^data_root =.*|data_root = \"${DATA_ROOT}\"|" configs/large_728_mlp_clean2.py
sed -i "s|^parquet_root =.*|parquet_root = \"${PARQUET_ROOT}\"|" configs/large_728_mlp_clean2.py
uv run torchrun --nproc_per_node=4 --master-port $PORT \
                train_vit_mm.py configs/large_728_mlp_clean2.py \
                --work-dir work_dirs/large_728_mlp_clean2


#==========================================================
# 4: evaluate model with clean data
#==========================================================
uv run python evaluate_vit_mm_moretta.py \
        --config configs/large_728_mlp_clean2.py \
        --checkpoint work_dirs/large_728_mlp_clean2/epoch_10.pth \
        --predict \
        --image-dir ${DATA2_ROOT}/data/test/query-data/crop/ \
        --output-dir work_dirs/ \
        --tta \
        --multi-gpu \
        --tta-scales 728 868 1008 1148 1288