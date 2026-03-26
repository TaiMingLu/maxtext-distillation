#!/bin/bash
set -euo pipefail

# Convert official Llama 3.2 3B to MaxText format.
# Uses Meta's native .pth format (same proven path as 8B conversion).
# head_dim=128 (standard), 24 query heads, tie_word_embeddings=true

cd ~/maxtext
source ~/maxtext_env/bin/activate

BUCKET_NAME="${BUCKET_NAME:?BUCKET_NAME not set}"
LOCAL_META_PATH="/dev/shm/meta-ckpt-3b"
rm -rf ~/2026-* /tmp/2026-* /dev/shm/meta-ckpt-* 2>/dev/null

MAXTEXT_PATH="gs://${BUCKET_NAME}/rebuttal/converted/llama3.2-3b-official"
MODEL_SIZE="llama3.2-3b-official"

echo "========================================"
echo "Converting official Llama 3.2 3B (META .pth format)"
echo "========================================"

# Step 1: Copy Meta native checkpoint
rm -rf "$LOCAL_META_PATH"
mkdir -p "$LOCAL_META_PATH"
echo "Copying Meta checkpoint from GCS..."
gcloud storage cp "gs://${BUCKET_NAME}/rebuttal/hf_models/Llama-3.2-3B/original/consolidated.00.pth" "$LOCAL_META_PATH/" || \
  gcloud storage cp "gs://${BUCKET_NAME}/rebuttal/hf_models/Llama-3.2-3B/original/consolidated.pth" "$LOCAL_META_PATH/"
gcloud storage cp "gs://${BUCKET_NAME}/rebuttal/hf_models/Llama-3.2-3B/original/params.json" "$LOCAL_META_PATH/"
gcloud storage cp "gs://${BUCKET_NAME}/rebuttal/hf_models/Llama-3.2-3B/original/tokenizer.model" "$LOCAL_META_PATH/"
echo "Copy done. $(du -sh $LOCAL_META_PATH | cut -f1)"

# Step 2: Convert Meta .pth to MaxText scanned checkpoint
JAX_PLATFORMS=cpu python3.10 -m MaxText.llama_or_mistral_ckpt \
  --base-model-path "$LOCAL_META_PATH" \
  --maxtext-model-path "$MAXTEXT_PATH" \
  --model-size "$MODEL_SIZE"

echo "Wrote scanned checkpoint to $MAXTEXT_PATH"

rm -rf "$LOCAL_META_PATH"
echo "Done. Scanned checkpoint at: ${MAXTEXT_PATH}/0/items"
