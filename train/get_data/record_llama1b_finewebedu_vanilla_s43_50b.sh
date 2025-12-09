#!/bin/bash
#
# Record training data samples for llama1b-finewebedu-vanilla-s43-50b run
#
# This script creates the exact same data iterator as the training run
# and records a sample of sequences (0.1% by default) to analyze what
# data was seen during training.
#
# IMPORTANT: Run this in the same environment as training to ensure
# the exact same data ordering. The key parameters that determine
# data order are:
#   - grain_train_files: The data files
#   - data_shuffle_seed: 43 (same as training)
#   - enable_data_shuffling: true
#   - packing: true
#   - max_target_length: 8192
#   - per_device_batch_size: 4
#   - gradient_accumulation_steps: 1
#   - num_devices: Number of TPU/GPU devices (must match training)
#   - dataloading_host_index/count: Must match training setup
#
# Usage:
#   bash record_llama1b_finewebedu_vanilla_s43_50b.sh [num_devices] [host_index] [host_count]
#
# Examples:
#   # Single host with 8 devices (e.g., v4-8 TPU)
#   bash record_llama1b_finewebedu_vanilla_s43_50b.sh 8 0 1
#
#   # Multi-host setup: host 0 of 2 hosts, each with 4 devices
#   bash record_llama1b_finewebedu_vanilla_s43_50b.sh 4 0 2
#

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MAXTEXT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Configuration matching training script
export MODEL_NAME='llama3.1-1b'
export NUM_STEPS=25000
export SEQ_LEN=8192
export BATCH_SIZE=4
export GRAD_ACCUM=1
export DATA_FILES='/home/terry/gcs-bucket/datasets/fineweb-edu/*.array_record'
export DATA_SHUFFLE_SEED=43

# Command line arguments with defaults
NUM_DEVICES=${1:-8}  # Default to 8 devices (common for v4-8)
HOST_INDEX=${2:-0}
HOST_COUNT=${3:-1}

# Sampling configuration
SAMPLE_PROB=${SAMPLE_PROB:-0.001}  # 0.1% by default
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"
OUTPUT_FILE="$OUTPUT_DIR/train_data_samples_host${HOST_INDEX}.pkl"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Recording training data samples"
echo "========================================"
echo "Configuration:"
echo "  Data files: $DATA_FILES"
echo "  Shuffle seed: $DATA_SHUFFLE_SEED"
echo "  Sequence length: $SEQ_LEN"
echo "  Batch size per device: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Number of steps: $NUM_STEPS"
echo "  Number of devices: $NUM_DEVICES"
echo "  Host index: $HOST_INDEX"
echo "  Host count: $HOST_COUNT"
echo "  Sample probability: $SAMPLE_PROB"
echo "  Output file: $OUTPUT_FILE"
echo "========================================"

cd "$MAXTEXT_DIR"

python3 "$SCRIPT_DIR/record_train_data.py" \
    --grain_train_files="$DATA_FILES" \
    --data_shuffle_seed=$DATA_SHUFFLE_SEED \
    --max_target_length=$SEQ_LEN \
    --per_device_batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=$GRAD_ACCUM \
    --num_devices=$NUM_DEVICES \
    --packing=True \
    --enable_data_shuffling=True \
    --num_steps=$NUM_STEPS \
    --sample_prob=$SAMPLE_PROB \
    --output_file="$OUTPUT_FILE" \
    --save_interval=1000 \
    --dataloading_host_index=$HOST_INDEX \
    --dataloading_host_count=$HOST_COUNT

echo ""
echo "Done! Samples saved to: $OUTPUT_FILE"
echo ""
echo "To analyze the samples, use Python:"
echo "  import pickle"
echo "  with open('$OUTPUT_FILE', 'rb') as f:"
echo "      data = pickle.load(f)"
echo "  print(f'Recorded {len(data[\"samples\"])} samples')"
