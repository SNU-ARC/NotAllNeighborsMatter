#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export LOG_DIR=./outputs/StanfordArea5Dataset/$TIME

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

# For test only
python -m main \
       	--dataset StanfordArea5Dataset \
       	--batch_size 1 \
       	--model Res16UNet18D \
       	--conv1_kernel_size 3 \
       	--data_aug_color_trans_ratio 0.05 \
       	--data_aug_color_jitter_std 0.005 \
        $2 $3 \
        $4 $5 \
        $6 $7 \
        $8 $9 \
        ${10} ${11} \
       	2>&1 | tee -a "$LOG"
