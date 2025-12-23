#!/bin/bash
#
# Accuracy evaluation script for BASE models (pretrained, before SFT)
# Runs lm-evaluation-harness accuracy tasks WITHOUT chat template
#
# Usage:
#   ./eval_base_acc.sh <run_name> <model_name> <checkpoint_step> <ckpt_dir> [--resume] [--force]
#
# Examples:
#   ./eval_base_acc.sh llama3.1-1b-finewebedu-vanilla-s42_v6 llama3.1-1b 24999 pretrain
#   ./eval_base_acc.sh llama3.1-1b-finewebedu-vanilla-s42_v6 llama3.1-1b 24999 pretrain --resume
#   ./eval_base_acc.sh exp1_llama3.1-1b-A1BT50BS42-a1-s43 llama3.1-1b 24999 distill --force
#
# --resume: Continue from incomplete results (saves progress after each task)
# --force: Re-run even if results exist
#

set +x
set -eo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <run_name> <model_name> <checkpoint_step> <ckpt_dir> [--resume] [--force]"
  echo "  run_name: Name of the run (e.g., llama3.1-1b-finewebedu-vanilla-s42_v6)"
  echo "  model_name: Model architecture (e.g., llama3.1-1b)"
  echo "  checkpoint_step: Checkpoint step to evaluate"
  echo "  ckpt_dir: Checkpoint directory type (e.g., pretrain, distill)"
  echo "  --resume: Continue from incomplete results"
  echo "  --force: Re-run even if results exist"
  exit 1
fi

RUN_NAME="$1"
MODEL_NAME="$2"
CHECKPOINT_STEP="$3"
CKPT_DIR="$4"
# Check for --force, --resume, --param_only flags in remaining args
FORCE_EVAL=false
RESUME_FLAG="false"
PARAM_ONLY="false"
for arg in "$@"; do
  if [[ "$arg" == "--force" ]]; then
    FORCE_EVAL=true
  fi
  if [[ "$arg" == "--resume" ]]; then
    RESUME_FLAG="true"
  fi
  if [[ "$arg" == "--param_only" ]]; then
    PARAM_ONLY="true"
  fi
done

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

# Configuration
if [[ "${PARAM_ONLY}" == "true" ]]; then
  CHECKPOINT_PATH="gs://${BUCKET_NAME}/ckpts/${CKPT_DIR}/${RUN_NAME}/checkpoint_${CHECKPOINT_STEP}/0/items"
else
  CHECKPOINT_PATH="gs://${BUCKET_NAME}/ckpts/${CKPT_DIR}/${RUN_NAME}/checkpoints/${CHECKPOINT_STEP}/items"
fi

# Use base model tokenizer (NOT Instruct) since we're evaluating base model
HF_MODEL_PATH="/home/terry/gcs-bucket/HF_HOME/Llama-3.1-8B"
EVAL_RESULTS_DIR="/home/terry/gcs-bucket/eval_1222/base_acc_results"
RESULT_JSON_PATH="${EVAL_RESULTS_DIR}/${RUN_NAME}_step${CHECKPOINT_STEP}.json"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_DISABLE_MOST_OPTIMIZATIONS=False

echo "========================"
echo "Base Model ACC Evaluation Configuration:"
echo "  RUN_NAME: ${RUN_NAME}"
echo "  MODEL_NAME: ${MODEL_NAME}"
echo "  CHECKPOINT_STEP: ${CHECKPOINT_STEP}"
echo "  CKPT_DIR: ${CKPT_DIR}"
echo "  CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
echo "  HF_MODEL_PATH: ${HF_MODEL_PATH} (base model, no chat template)"
echo "  EVAL_RESULTS_DIR: ${EVAL_RESULTS_DIR}"
echo "  FORCE_EVAL: ${FORCE_EVAL}"
echo "  RESUME_FLAG: ${RESUME_FLAG}"
echo "========================"

# Check if results exist
if [[ -f "${RESULT_JSON_PATH}" ]] && [[ "${FORCE_EVAL}" == "false" ]]; then
  if [[ "${RESUME_FLAG}" == "true" ]]; then
    echo "Results exist at ${RESULT_JSON_PATH}, --resume flag set, will check for missing benchmarks..."
  else
    # Check if results are complete (no _incomplete flag)
    if grep -q '"_incomplete": true' "${RESULT_JSON_PATH}" 2>/dev/null; then
      echo "Found incomplete results at ${RESULT_JSON_PATH}"
      echo "  -> Use --resume to continue from where it left off, or delete the file to start fresh."
      exit 0
    else
      echo "Results already exist and are complete at ${RESULT_JSON_PATH}; skipping."
      echo "  -> Use --resume to re-check for any new benchmarks."
      exit 0
    fi
  fi
fi

cd ~/maxtext
export PYTHONPATH="$(pwd):${PYTHONPATH}"

echo "------------------------------------------------------------------"
echo "Evaluating BASE model ACC (no chat template): ${RUN_NAME}"
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
      dtype=bfloat16 \
      max_target_length=4096 \
      scan_layers=true \
      attention=dot_product \
      --hf_model_path=${HF_MODEL_PATH} \
      --eval_mode=acc \
      --eval_save_dir=${EVAL_RESULTS_DIR} \
      --acc_batch_size=16 \
      --acc_seq_length=4096 \
      --apply_chat_template=false \
      --resume=${RESUME_FLAG}
else
  echo "Multi-host TPU detected (size=${TPU_SIZE}), using multihost_runner..."
  python -u multihost_runner_orig.py \
    --TPU_PREFIX=${TPU_PREFIX} \
    --INTERNAL_IP=true \
    --RUN_NAME=maxtext_eval \
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
    dtype=bfloat16 \
    max_target_length=4096 \
    scan_layers=true \
    attention=dot_product \
    --hf_model_path=${HF_MODEL_PATH} \
    --eval_mode=acc \
    --eval_save_dir=${EVAL_RESULTS_DIR} \
    --acc_batch_size=16 \
    --acc_seq_length=4096 \
    --apply_chat_template=false \
    --resume=${RESUME_FLAG}
"
fi

echo "Done evaluating BASE model ACC: ${RUN_NAME}"
echo "Results saved to: ${RESULT_JSON_PATH}"
