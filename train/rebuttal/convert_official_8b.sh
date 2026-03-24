#!/bin/bash
set -euo pipefail

# Convert official Llama 3.1 8B to MaxText format.
# Uses Meta's native .pth format — same as official MaxText e2e test.
# See: end_to_end/tpu/llama3.1/8b/1_test_llama3.1_8b.sh

cd ~/maxtext
source ~/maxtext_env/bin/activate

BUCKET_NAME="${BUCKET_NAME:?BUCKET_NAME not set}"
LOCAL_META_PATH="/tmp/meta-ckpt-8b"
MAXTEXT_PATH="gs://${BUCKET_NAME}/rebuttal/converted/llama3.1-8b-official"
MODEL_SIZE="llama3.1-8b"

echo "========================================"
echo "Converting official Llama 3.1 8B"
echo "========================================"

# Step 1: Copy only the Meta native checkpoint files (not the full HF repo)
rm -rf "$LOCAL_META_PATH"
mkdir -p "$LOCAL_META_PATH"
echo "Copying Meta checkpoint from GCS..."
gcloud storage cp "gs://${BUCKET_NAME}/rebuttal/hf_models/Llama-3.1-8B/original/consolidated.00.pth" "$LOCAL_META_PATH/"
gcloud storage cp "gs://${BUCKET_NAME}/rebuttal/hf_models/Llama-3.1-8B/original/params.json" "$LOCAL_META_PATH/"
gcloud storage cp "gs://${BUCKET_NAME}/rebuttal/hf_models/Llama-3.1-8B/original/tokenizer.model" "$LOCAL_META_PATH/"
echo "Copy done. $(du -sh $LOCAL_META_PATH | cut -f1)"

# Step 2: Convert Meta .pth to MaxText scanned checkpoint (on CPU)
JAX_PLATFORMS=cpu python3.10 -m MaxText.llama_or_mistral_ckpt \
  --base-model-path "$LOCAL_META_PATH" \
  --maxtext-model-path "$MAXTEXT_PATH" \
  --model-size "$MODEL_SIZE"

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

rm -rf "$LOCAL_META_PATH"
echo "Done. Param-only checkpoint at: gs://${BUCKET_NAME}/rebuttal/converted/llama3.1-8b-official-param-only/checkpoints/0/items"
