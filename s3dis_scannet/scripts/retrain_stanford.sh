#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

export BATCH_SIZE=${BATCH_SIZE:-6}

export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export LOG_DIR=./retrain_outputs/StanfordArea5Dataset_retrain/$TIME

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

# For train (Original 60000 iter)
python -m main \
    --dataset StanfordArea5Dataset \
    --batch_size $BATCH_SIZE \
    --scheduler PolyLR \
    --model Res16UNet18D \
    --conv1_kernel_size 3 \
    --log_dir $LOG_DIR \
    --lr 3e-2 \
    --max_iter 12000 \
    --is_train True \
    --data_aug_color_trans_ratio 0.05 \
    --data_aug_color_jitter_std 0.005 \
    --train_phase train \
    --val_freq 20 \
    --retrain \
    $2 $3 \
    $4 $5 \
    $6 $7 \
    $8 $9 \
    ${10} ${11} \
    2>&1 | tee -a "$LOG"
