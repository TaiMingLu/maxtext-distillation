#!/bin/bash
# Exp 5: Single-model mechanism study evaluation
# Usage: bash train/rebuttal/exp5_single_model.sh <label> <checkpoint_path> <model_name>
# Runs PPL evaluation on one model. Saves results to GCS.

LABEL="${1:?Usage: $0 <label> <checkpoint_path> <model_name>}"
CKPT="${2:?Missing checkpoint path}"
MODEL="${3:?Missing model name}"

BUCKET_NAME="${BUCKET_NAME:?BUCKET_NAME not set}"

# Stay in the repo root (where the queue runner cloned us)
cd ~/maxtext || exit 1
source ~/maxtext_env/bin/activate
export PYTHONPATH="$(pwd):$(pwd)/lm-evaluation-harness:${PYTHONPATH:-}"

HF_TOKENIZER="/home/terry/gcs-bucket/rebuttal/hf_models/Llama-3.1-8B"
SAVE_DIR="/home/terry/gcs-bucket/rebuttal/exp5_mechanism/$LABEL"

# Install lm-eval-harness
cd lm-evaluation-harness && pip install -e . -q 2>&1 | tail -1 || true
cd ~/maxtext

echo "========================================"
echo "Exp 5: $LABEL (model=$MODEL)"
echo "  Checkpoint: $CKPT"
echo "  Save dir: $SAVE_DIR"
echo "========================================"

# Run from repo root so relative paths work
cd ~/maxtext/lm-evaluation-harness

python3.10 -u scripts/test_orbax_eval.py ../MaxText/configs/base.yml \
    load_parameters_path="$CKPT" \
    run_name="exp5_${LABEL}" \
    model_name="$MODEL" \
    max_target_length=4096 \
    dtype=bfloat16 \
    scan_layers=true \
    attention=dot_product \
    --hf_model_path="$HF_TOKENIZER" \
    --eval_mode=ppl \
    --eval_save_dir="$SAVE_DIR" \
    --ppl_batch_size=8 \
    --ppl_seq_length=4096

EXIT_CODE=$?
echo "=== $LABEL done (exit $EXIT_CODE) ==="
exit $EXIT_CODE
