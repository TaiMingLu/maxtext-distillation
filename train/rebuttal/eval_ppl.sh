#!/bin/bash
#
# PPL evaluation for rebuttal checkpoints
# Same eval logic as train/eval/eval_pretrain_ppl.sh, adapted for rebuttal paths
#
# Usage:
#   ./eval_ppl.sh <run_name> [checkpoint_step] [checkpoint_type]
#
# checkpoint_type:
#   rebuttal - gs://BUCKET/rebuttal/<exp>/... (default)
#   path     - run_name is treated as a full checkpoint path directly
#
# Examples:
#   ./eval_ppl.sh exp3_baseline_s43 4999 rebuttal
#   ./eval_ppl.sh exp1_official-8b-teacher_a1-s43 24999 rebuttal
#   ./eval_ppl.sh gs://taiming_us_central1/rebuttal/baselines/llama3.1-1b-s43/24999/items 0 path
#

set +x
set -eo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_name> [checkpoint_step] [checkpoint_type]"
  echo "  run_name: Name of the rebuttal run, or full checkpoint path if type=path"
  echo "  checkpoint_step: Checkpoint step to evaluate (default: 4999)"
  echo "  checkpoint_type: 'rebuttal' (default) or 'path'"
  exit 1
fi

RUN_NAME="$1"
CHECKPOINT_STEP="${2:-4999}"
CHECKPOINT_TYPE="${3:-rebuttal}"

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

# All rebuttal students are 1.7B (llama3.1-1b)
MODEL_NAME="llama3.1-1b"

# Checkpoint path based on type
if [[ "${CHECKPOINT_TYPE}" == "rebuttal" ]]; then
  # Points to param-only converted checkpoint
  CHECKPOINT_PATH="gs://${BUCKET_NAME}/rebuttal/param_only/${RUN_NAME}/checkpoint_${CHECKPOINT_STEP}/0/items"
elif [[ "${CHECKPOINT_TYPE}" == "path" ]]; then
  CHECKPOINT_PATH="${RUN_NAME}"
  RUN_NAME=$(basename "$(dirname "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")")
else
  echo "[ERROR] Invalid checkpoint_type: ${CHECKPOINT_TYPE}. Use 'rebuttal' or 'path'"
  exit 1
fi

# Configuration — same as train/eval/eval_pretrain_ppl.sh
HF_MODEL_PATH="/home/terry/gcs-bucket/rebuttal/hf_models/Llama-3.1-8B"
EVAL_RESULTS_DIR="/home/terry/gcs-bucket/rebuttal/eval_results/ppl"
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
