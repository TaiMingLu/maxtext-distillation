#!/bin/bash
set -euo pipefail

# Convert official Llama 3.1 8B from HuggingFace to MaxText param-only checkpoint
# Run on a single TPU host (v6e-8 is fine)

cd ~/maxtext
source ~/maxtext_env/bin/activate

BUCKET_NAME="${BUCKET_NAME:?BUCKET_NAME not set}"
HF_PATH="/home/terry/gcs-bucket/rebuttal/hf_models/Llama-3.1-8B"
MAXTEXT_PATH="gs://${BUCKET_NAME}/rebuttal/converted/llama3.1-8b-official"
MODEL_SIZE="llama3.1-8b"

echo "========================================"
echo "Converting official Llama 3.1 8B"
echo "  HF path: $HF_PATH"
echo "  MaxText path: $MAXTEXT_PATH"
echo "========================================"

python3.10 -m MaxText.llama_or_mistral_ckpt \
  --base-model-path "$HF_PATH" \
  --maxtext-model-path "$MAXTEXT_PATH" \
  --model-size "$MODEL_SIZE"

echo "Done. Checkpoint at: $MAXTEXT_PATH"
