#!/bin/bash
set -euo pipefail

# Convert official Llama 3.2 1B to MaxText format.
# Uses Meta's native .pth format — same proven path as the 8B conversion.
# NOTE: head_dim=64 and tie_word_embeddings=true

cd ~/maxtext
source ~/maxtext_env/bin/activate

BUCKET_NAME="${BUCKET_NAME:?BUCKET_NAME not set}"
# Use /dev/shm (RAM-backed tmpfs) to avoid disk full issues
LOCAL_META_PATH="/dev/shm/meta-ckpt-1b"
rm -rf ~/2026-* /tmp/2026-* /tmp/Llama-* /dev/shm/Llama-* /dev/shm/meta-ckpt-* 2>/dev/null

MAXTEXT_PATH="gs://${BUCKET_NAME}/rebuttal/converted/llama3.2-1b-official"
MODEL_SIZE="llama3.2-1b-official"

echo "========================================"
echo "Converting official Llama 3.2 1B (META .pth format)"
echo "========================================"

# Step 1: Copy only the Meta native checkpoint files
rm -rf "$LOCAL_META_PATH"
mkdir -p "$LOCAL_META_PATH"
echo "Copying Meta checkpoint from GCS..."
gcloud storage cp "gs://${BUCKET_NAME}/rebuttal/hf_models/Llama-3.2-1B/original/consolidated.00.pth" "$LOCAL_META_PATH/"
gcloud storage cp "gs://${BUCKET_NAME}/rebuttal/hf_models/Llama-3.2-1B/original/params.json" "$LOCAL_META_PATH/"
gcloud storage cp "gs://${BUCKET_NAME}/rebuttal/hf_models/Llama-3.2-1B/original/tokenizer.model" "$LOCAL_META_PATH/"
echo "Copy done. $(du -sh $LOCAL_META_PATH | cut -f1)"

# Step 2: Convert Meta .pth to MaxText scanned checkpoint (on CPU)
# NOTE: Do NOT use --huggingface-checkpoint — use the META .pth path which is proven to work
JAX_PLATFORMS=cpu python3.10 -m MaxText.llama_or_mistral_ckpt \
  --base-model-path "$LOCAL_META_PATH" \
  --maxtext-model-path "$MAXTEXT_PATH" \
  --model-size "$MODEL_SIZE"

echo "Wrote scanned checkpoint to $MAXTEXT_PATH"

# Step 3: Generate param-only checkpoint
export CONVERTED_CHECKPOINT="${MAXTEXT_PATH}/0/items"
JAX_PLATFORMS=cpu python3.10 -m MaxText.generate_param_only_checkpoint MaxText/configs/base.yml \
  async_checkpointing=false \
  base_output_directory="gs://${BUCKET_NAME}/rebuttal/converted" \
  load_full_state_path="${CONVERTED_CHECKPOINT}" \
  run_name=llama3.2-1b-official-param-only \
  model_name="${MODEL_SIZE}" \
  force_unroll=true

rm -rf "$LOCAL_META_PATH"
echo "Done. Param-only checkpoint at: gs://${BUCKET_NAME}/rebuttal/converted/llama3.2-1b-official-param-only/checkpoints/0/items"
