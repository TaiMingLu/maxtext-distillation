#!/bin/bash
# Convert a full training checkpoint to param-only for evaluation
# Usage: bash train/rebuttal/convert_to_param_only.sh <run_path> <checkpoint_step> <model_name>
#
# Example:
#   bash train/rebuttal/convert_to_param_only.sh exp3/exp3_baseline_s43 4999 llama3.1-1b
#
# Input:  gs://BUCKET/rebuttal/<run_path>/checkpoints/<step>/items
# Output: gs://BUCKET/rebuttal/param_only/<run_name>/checkpoint_<step>/

set -euo pipefail
cd ~/maxtext
source ~/maxtext_env/bin/activate

RUN_PATH="${1:?Usage: $0 <run_path> <checkpoint_step> <model_name>}"
CHECKPOINT_STEP="${2:?Usage: $0 <run_path> <checkpoint_step> <model_name>}"
MODEL_NAME="${3:-llama3.1-1b}"
BUCKET_NAME="${BUCKET_NAME:?BUCKET_NAME not set}"

# Extract run_name from path (last component)
RUN_NAME=$(basename "$RUN_PATH")

LOAD_PATH="gs://${BUCKET_NAME}/rebuttal/${RUN_PATH}/checkpoints/${CHECKPOINT_STEP}/items"
OUT_DIR="gs://${BUCKET_NAME}/rebuttal/param_only/${RUN_NAME}/checkpoint_${CHECKPOINT_STEP}"

echo "========================================"
echo "Convert to param-only"
echo "  Run: ${RUN_NAME}"
echo "  Model: ${MODEL_NAME}"
echo "  From: ${LOAD_PATH}"
echo "  To:   ${OUT_DIR}"
echo "========================================"

python3.10 -u -m MaxText.generate_param_only_checkpoint MaxText/configs/base.yml \
    load_full_state_path="$LOAD_PATH" \
    checkpoint_dir="$OUT_DIR" \
    enable_checkpointing=True \
    async_checkpointing=False \
    model_name="$MODEL_NAME"

echo "Done. Output: ${OUT_DIR}/0/items"
