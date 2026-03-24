#!/bin/bash
set -euo pipefail

# Convert official Llama 3.2 1B from HuggingFace to MaxText param-only checkpoint
# NOTE: This model has head_dim=64 and tie_word_embeddings=true
#       Uses llama3.2-1b-official MaxText config (not llama3.1-1b)
# Run on a single TPU host (v6e-8 is fine)
# If this fails due to Flash Attention not supporting head_dim=64, skip Exp 2.

cd ~/maxtext
source ~/maxtext_env/bin/activate

BUCKET_NAME="${BUCKET_NAME:?BUCKET_NAME not set}"
HF_PATH="/home/terry/gcs-bucket/rebuttal/hf_models/Llama-3.2-1B"
MAXTEXT_PATH="gs://${BUCKET_NAME}/rebuttal/converted/llama3.2-1b-official"
MODEL_SIZE="llama3.2-1b-official"

echo "========================================"
echo "Converting official Llama 3.2 1B"
echo "  HF path: $HF_PATH"
echo "  MaxText path: $MAXTEXT_PATH"
echo "  NOTE: head_dim=64, may fail if Flash Attention requires 128"
echo "========================================"

python3.10 -m MaxText.llama_or_mistral_ckpt \
  --base-model-path "$HF_PATH" \
  --maxtext-model-path "$MAXTEXT_PATH" \
  --model-size "$MODEL_SIZE"

echo "Done. Checkpoint at: $MAXTEXT_PATH"
