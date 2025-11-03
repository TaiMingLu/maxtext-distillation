#!/bin/bash

set +x
set -eo pipefail

BUCKET_NAME='taiming_us_central2_b'
TPU_PREFIX='taiming-v4-64'
HF_MODEL_PATH='/home/terry/gcs-bucket/HF_HOME/Llama-3.1-8B'
EVAL_RESULTS_DIR='/home/terry/gcs-bucket/eval/results'
WANDB_API_KEY='01126ae90da25bae0d86704140ac978cb9fd9c73'
WANDB_PROJECT='maxtext_1b'
BASE_OUTPUT_DIRECTORY="gs://${BUCKET_NAME}/eval_param_only"
BASE_OUTPUT_DIRECTORY_DISK='/home/terry/gsc-bucket/eval_param_only'
XLA_PYTHON_CLIENT_MEM_FRACTION='0.9'
XLA_PYTHON_CLIENT_ALLOCATOR='platform'
JAX_DISABLE_MOST_OPTIMIZATIONS='False'
TPU_LOG_DIR='/home/terry/tpu_logs'

cd "${HOME}/maxtext"
export PYTHONPATH="$(pwd):${PYTHONPATH}"

MULTIHOST_COMMAND=$(cat <<EOF
export BUCKET_NAME='${BUCKET_NAME}'
export EVAL_RESULTS_DIR='${EVAL_RESULTS_DIR}'
export BASE_OUTPUT_DIRECTORY='${BASE_OUTPUT_DIRECTORY}'
export BASE_OUTPUT_DIRECTORY_DISK='${BASE_OUTPUT_DIRECTORY_DISK}'
export HF_MODEL_PATH='${HF_MODEL_PATH}'
export WANDB_API_KEY='${WANDB_API_KEY}'
export WANDB_PROJECT='${WANDB_PROJECT}'
export XLA_PYTHON_CLIENT_MEM_FRACTION='${XLA_PYTHON_CLIENT_MEM_FRACTION}'
export XLA_PYTHON_CLIENT_ALLOCATOR='${XLA_PYTHON_CLIENT_ALLOCATOR}'
export JAX_DISABLE_MOST_OPTIMIZATIONS='${JAX_DISABLE_MOST_OPTIMIZATIONS}'
export TPU_LOG_DIR='${TPU_LOG_DIR}'
bash ~/maxtext/lm-evaluation-harness/scripts/test_orbax_eval_all_inner.sh
EOF
)

python -u multihost_runner_orig.py \
  --TPU_PREFIX="${TPU_PREFIX}" \
  --INTERNAL_IP=true \
  --RUN_NAME=maxtext \
  --COMMAND="${MULTIHOST_COMMAND}"
