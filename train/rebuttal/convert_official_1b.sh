#!/bin/bash
set -euo pipefail

# Convert official Llama 3.2 1B to MaxText format.
# NOTE: head_dim=64 and tie_word_embeddings=true (different from our llama3.1-1b)
# Uses HF safetensors format since the 1B model is small (2.4GB single file).

cd ~/maxtext
source ~/maxtext_env/bin/activate

BUCKET_NAME="${BUCKET_NAME:?BUCKET_NAME not set}"
# Use /dev/shm (RAM-backed tmpfs) to avoid disk full issues
LOCAL_HF_PATH="/dev/shm/Llama-3.2-1B"
# Clean up old files first
rm -rf ~/2026-* /tmp/2026-* /tmp/Llama-* /dev/shm/Llama-* 2>/dev/null
MAXTEXT_PATH="gs://${BUCKET_NAME}/rebuttal/converted/llama3.2-1b-official"
MODEL_SIZE="llama3.2-1b-official"

echo "========================================"
echo "Converting official Llama 3.2 1B"
echo "========================================"

# Step 1: Copy from GCS to local
rm -rf "$LOCAL_HF_PATH"
echo "Copying from GCS to $LOCAL_HF_PATH ..."
gcloud storage cp -r "gs://${BUCKET_NAME}/rebuttal/hf_models/Llama-3.2-1B" /dev/shm/
echo "Copy done. $(du -sh $LOCAL_HF_PATH | cut -f1)"

# Step 2: Convert HF checkpoint to MaxText scanned checkpoint (on CPU)
JAX_PLATFORMS=cpu python3.10 -m MaxText.llama_or_mistral_ckpt \
  --base-model-path "$LOCAL_HF_PATH" \
  --maxtext-model-path "$MAXTEXT_PATH" \
  --model-size "$MODEL_SIZE" \
  --huggingface-checkpoint=true

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

rm -rf "$LOCAL_HF_PATH"
echo "Done. Param-only checkpoint at: gs://${BUCKET_NAME}/rebuttal/converted/llama3.2-1b-official-param-only/checkpoints/0/items"
