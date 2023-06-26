#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

export MODEL=${MODEL:-Res16UNet18D}
export DATASET=${DATASET:-ScannetVoxelization2cmDataset}
export LOG_DIR=./outputs/$DATASET/$MODEL-b$BATCH_SIZE-$MAX_ITER/$TIME

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

# For test only
python -m main \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --conv1_kernel_size 3 \
    $2 $3 \
    $4 $5 \
    $6 $7 \
    $8 $9 \
    ${10} ${11} \
    2>&1 | tee -a "$LOG"
