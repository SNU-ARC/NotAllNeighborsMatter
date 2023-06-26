#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

export BATCH_SIZE=${BATCH_SIZE:-6}
export MAX_ITER=${MAX_ITER:-120000}
export MODEL=${MODEL:-Res16UNet34}
export DATASET=${DATASET:-ScannetVoxelization2cmDataset}

export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

python -m main \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --lr 1e-1 \
    --batch_size $BATCH_SIZE \
    --scheduler PolyLR \
    --max_iter $MAX_ITER \
    --is_train True \
    --conv1_kernel_size 3 \
    --train_limit_numpoints 1200000 \
    --train_phase train \
    $2 $3 \
    $4 $5 \
    $6 $7 \
    $8 $9 \
    ${10} ${11} \
    --log_dir $LOG_DIR \
    2>&1 | tee -a "$LOG"
