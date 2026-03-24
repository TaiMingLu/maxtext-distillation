#!/bin/bash
set -euo pipefail

# Convert official Llama 3.1 8B HuggingFace checkpoint to MaxText format.
# Follows the official MaxText conversion pattern from end_to_end/tpu/llama3.1/8b/1_test_llama3.1_8b.sh

cd ~/maxtext
source ~/maxtext_env/bin/activate

BUCKET_NAME="${BUCKET_NAME:?BUCKET_NAME not set}"
LOCAL_HF_PATH="/tmp/Llama-3.1-8B"
MAXTEXT_PATH="gs://${BUCKET_NAME}/rebuttal/converted/llama3.1-8b-official"
MODEL_SIZE="llama3.1-8b"

echo "========================================"
echo "Converting official Llama 3.1 8B"
echo "========================================"

# Step 1: Copy from GCS to local (use gcloud storage cp, not gcsfuse cp)
rm -rf "$LOCAL_HF_PATH"
echo "Copying from GCS to $LOCAL_HF_PATH ..."
gcloud storage cp -r "gs://${BUCKET_NAME}/rebuttal/hf_models/Llama-3.1-8B" /tmp/
echo "Copy done. $(du -sh $LOCAL_HF_PATH | cut -f1)"

# Step 2: Convert (on CPU, no TPU needed)
JAX_PLATFORMS=cpu python3.10 -m MaxText.llama_or_mistral_ckpt \
  --base-model-path "$LOCAL_HF_PATH" \
  --maxtext-model-path "$MAXTEXT_PATH" \
  --model-size "$MODEL_SIZE" \
  --huggingface-checkpoint=true

echo "Wrote scanned checkpoint to $MAXTEXT_PATH"

# Step 3: Generate param-only unscanned checkpoint for KD teacher loading
export CONVERTED_CHECKPOINT="${MAXTEXT_PATH}/0/items"
JAX_PLATFORMS=cpu python3.10 -m MaxText.generate_param_only_checkpoint MaxText/configs/base.yml \
  async_checkpointing=false \
  base_output_directory="gs://${BUCKET_NAME}/rebuttal/converted" \
  load_parameters_path="${CONVERTED_CHECKPOINT}" \
  run_name=llama3.1-8b-official-param-only \
  model_name="${MODEL_SIZE}" \
  force_unroll=true

rm -rf "$LOCAL_HF_PATH"
echo "Done. Param-only checkpoint at: gs://${BUCKET_NAME}/rebuttal/converted/llama3.1-8b-official-param-only/checkpoints/0/items"
