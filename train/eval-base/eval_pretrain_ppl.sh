#!/bin/bash
#
# PPL evaluation script for pretrain models (before SFT)
# Runs lm-evaluation-harness PPL tasks on Orbax checkpoints
#
# Usage:
#   ./eval_pretrain_ppl.sh <run_name> [checkpoint_step] [checkpoint_type]
#
# Examples:
#   ./eval_pretrain_ppl.sh exp1_llama3.1-1b-A1BT50BS42-a1-s43
#   ./eval_pretrain_ppl.sh exp1_llama3.1-1b-A1BT50BS42-a1-s43 24999
#   ./eval_pretrain_ppl.sh exp1_llama3.1-1b-A1BT50BS42-a1-s43 24999 distill
#   ./eval_pretrain_ppl.sh llama3.1-1b-finewebedu-vanilla-s42-50b 24999 pretrain
#
# checkpoint_type:
#   distill  - gs://BUCKET/ckpts/distill_pretrain/... (default)
#   pretrain - gs://BUCKET/ckpts/pretrain/...
#

set +x
set -eo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_name> [checkpoint_step] [checkpoint_type]"
  echo "  run_name: Name of the pretrain run"
  echo "  checkpoint_step: Checkpoint step to evaluate (default: 24999)"
  echo "  checkpoint_type: 'distill' or 'pretrain' (default: distill)"
  exit 1
fi

RUN_NAME="$1"
CHECKPOINT_STEP="${2:-24999}"
CHECKPOINT_TYPE="${3:-distill}"

echo "========================"
echo "environment variables:"
echo "TPU_PREFIX: $TPU_PREFIX"
echo "BUCKET_NAME: $BUCKET_NAME"
echo "========================"

required_vars=(
    "BUCKET_NAME"
    "TPU_PREFIX"
)
for var in "${required_vars[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "[ERROR] $var is not set"
    exit 1
  fi
done

# Extract model name from run name (e.g., exp1_llama3.1-1b-... -> llama3.1-1b)
MODEL_NAME=$(echo "${RUN_NAME}" | sed -E 's/.*_(llama[0-9.]+-(0?[0-9]+b)).*/\1/')
if [[ -z "${MODEL_NAME}" || "${MODEL_NAME}" == "${RUN_NAME}" ]]; then
  # Try another pattern for vanilla runs (e.g., llama3.1-1b-finewebedu-vanilla-...)
  MODEL_NAME=$(echo "${RUN_NAME}" | sed -E 's/^(llama[0-9.]+-(0?[0-9]+b)).*/\1/')
fi
if [[ -z "${MODEL_NAME}" || "${MODEL_NAME}" == "${RUN_NAME}" ]]; then
  echo "[ERROR] Could not extract model name from run name: ${RUN_NAME}"
  echo "Expected format: *_llama3.1-1b-* or llama3.1-1b-* or similar"
  exit 1
fi

# Checkpoint path based on type
if [[ "${CHECKPOINT_TYPE}" == "distill" ]]; then
  CHECKPOINT_PATH="gs://${BUCKET_NAME}/ckpts/distill_pretrain/${RUN_NAME}/checkpoints/${CHECKPOINT_STEP}/items"
elif [[ "${CHECKPOINT_TYPE}" == "pretrain" ]]; then
  CHECKPOINT_PATH="gs://${BUCKET_NAME}/ckpts/pretrain/${RUN_NAME}/checkpoints/${CHECKPOINT_STEP}/items"
else
  echo "[ERROR] Invalid checkpoint_type: ${CHECKPOINT_TYPE}. Use 'distill' or 'pretrain'"
  exit 1
fi

# Configuration
HF_MODEL_PATH="/home/terry/gcs-bucket/HF_HOME/Llama-3.2-1B-Instruct"
EVAL_RESULTS_DIR="/home/terry/gcs-bucket/eval_new/ppl_results"
RESULT_JSON_PATH="${EVAL_RESULTS_DIR}/${RUN_NAME}_step${CHECKPOINT_STEP}.json"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_DISABLE_MOST_OPTIMIZATIONS=False

echo "========================"
echo "PPL Evaluation Configuration:"
echo "  RUN_NAME: ${RUN_NAME}"
echo "  MODEL_NAME: ${MODEL_NAME}"
echo "  CHECKPOINT_STEP: ${CHECKPOINT_STEP}"
echo "  CHECKPOINT_TYPE: ${CHECKPOINT_TYPE}"
echo "  CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
echo "  HF_MODEL_PATH: ${HF_MODEL_PATH}"
echo "  EVAL_RESULTS_DIR: ${EVAL_RESULTS_DIR}"
echo "========================"

if [[ -f "${RESULT_JSON_PATH}" ]]; then
  echo "Results already exist at ${RESULT_JSON_PATH}; skipping."
  exit 0
fi

cd ~/maxtext
export PYTHONPATH="$(pwd):${PYTHONPATH}"

echo "------------------------------------------------------------------"
echo "Evaluating PPL for ${RUN_NAME} at step ${CHECKPOINT_STEP}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "------------------------------------------------------------------"

# Detect if single-host TPU (v*-8 or smaller) or multi-host
# Extract TPU size from name (e.g., taiming-qw-v6e-64_8 -> 64, my-tpu-v6e-8 -> 8)
TPU_SIZE=$(echo "${TPU_PREFIX}" | grep -oP 'v[0-9]+e?-\K[0-9]+' || echo "0")
if [[ "${TPU_SIZE}" -le 8 ]]; then
  echo "Single-host TPU detected (size=${TPU_SIZE}), running directly..."
  cd lm-evaluation-harness
  pip install -e . -q
  python3.10 -u scripts/test_orbax_eval.py ../MaxText/configs/base.yml \
      load_parameters_path=${CHECKPOINT_PATH} \
      run_name=${RUN_NAME}_step${CHECKPOINT_STEP} \
      model_name=${MODEL_NAME} \
      max_target_length=4096 \
      dtype=bfloat16 \
      scan_layers=true \
      attention=dot_product \
      --hf_model_path=${HF_MODEL_PATH} \
      --eval_mode=ppl \
      --eval_save_dir=${EVAL_RESULTS_DIR} \
      --ppl_batch_size=8 \
      --ppl_seq_length=4096
else
  echo "Multi-host TPU detected (size=${TPU_SIZE}), using multihost_runner..."
  python -u multihost_runner_orig.py \
    --TPU_PREFIX=${TPU_PREFIX} \
    --INTERNAL_IP=true \
    --RUN_NAME=maxtext_eval_ppl \
    --COMMAND="
ROOT=\$(pwd)
cd lm-evaluation-harness
export TPU_LOG_DIR=/home/terry/tpu_logs
source ~/maxtext_env/bin/activate
export PYTHONPATH=\${ROOT}:\$(pwd):\$PYTHONPATH
pip install -e . -q
python3.10 -u scripts/test_orbax_eval.py ../MaxText/configs/base.yml \
    load_parameters_path=${CHECKPOINT_PATH} \
    run_name=${RUN_NAME}_step${CHECKPOINT_STEP} \
    model_name=${MODEL_NAME} \
    max_target_length=4096 \
    dtype=bfloat16 \
    scan_layers=true \
    attention=dot_product \
    --hf_model_path=${HF_MODEL_PATH} \
    --eval_mode=ppl \
    --eval_save_dir=${EVAL_RESULTS_DIR} \
    --ppl_batch_size=8 \
    --ppl_seq_length=4096
"
fi

echo "Done evaluating PPL for ${RUN_NAME}"
echo "Results saved to: ${RESULT_JSON_PATH}"
