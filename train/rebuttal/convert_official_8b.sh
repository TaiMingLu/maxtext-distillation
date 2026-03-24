#!/bin/bash
set -euo pipefail

# Convert official Llama 3.1 8B from HuggingFace to MaxText param-only checkpoint
# Copies model to local disk first to avoid gcsfuse safetensors issues.

cd ~/maxtext
source ~/maxtext_env/bin/activate

BUCKET_NAME="${BUCKET_NAME:?BUCKET_NAME not set}"
GCS_HF_PATH="/home/terry/gcs-bucket/rebuttal/hf_models/Llama-3.1-8B"
LOCAL_HF_PATH="/tmp/Llama-3.1-8B"
MAXTEXT_PATH="gs://${BUCKET_NAME}/rebuttal/converted/llama3.1-8b-official"
MODEL_SIZE="llama3.1-8b"

echo "========================================"
echo "Converting official Llama 3.1 8B"
echo "  Step 1: Copy from GCS to local disk"
echo "  Step 2: Convert to MaxText"
echo "========================================"

# Copy to local disk to avoid gcsfuse safetensors HeaderTooLarge error
rm -rf "$LOCAL_HF_PATH"
echo "Copying to $LOCAL_HF_PATH ..."
cp -r "$GCS_HF_PATH" "$LOCAL_HF_PATH"
echo "Copy done. $(du -sh $LOCAL_HF_PATH | cut -f1)"

python3.10 -m MaxText.llama_or_mistral_ckpt \
  --base-model-path "$LOCAL_HF_PATH" \
  --maxtext-model-path "$MAXTEXT_PATH" \
  --model-size "$MODEL_SIZE" \
  --huggingface-checkpoint=true

rm -rf "$LOCAL_HF_PATH"
echo "Done. Checkpoint at: $MAXTEXT_PATH"
