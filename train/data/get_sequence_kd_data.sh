#!/bin/bash
set -euo pipefail

cd ~/maxtext

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[ERROR] Environment variable '$name' is required." >&2
    exit 1
  fi
}

require_var "TPU_PREFIX"
require_var "BUCKET_NAME"
require_var "HF_ACCESS_TOKEN"

RUN_NAME="sequence-kd"
DATASET_PATH="HuggingFaceFW/fineweb-edu"
DATA_SPLIT="sample-350BT"
TEXT_COLUMN="text"
TEACHER_MODEL_NAME="llama3.1-1b"
TEACHER_PARAMETERS_PATH="/home/terry/gcs-bucket/ckpts/pretrain_param_only/llama3.1-1b_finewebedu_pretrain_shuffled_lr_3e-4_seed_42/checkpoint_24999/0/items"
TOKENIZER_PATH="/home/terry/gcs-bucket/HF_HOME/Llama-3.1-8B"
MAX_PREFILL_LENGTH=256
MAX_TARGET_LENGTH=4096
GEN_BATCH_SIZE=512
SERVER_PER_DEVICE_BATCH=8
JETSTREAM_SERVER_PORT=9000
SERVER_READY_TIMEOUT_SEC=900
ENGINE_PARALLEL_FLAGS="ici_data_parallelism=4 ici_tensor_parallelism=4 ici_fsdp_parallelism=4 ici_autoregressive_parallelism=1"
DECODE_SAMPLING_STRATEGY="greedy"
PROGRESS_PATH="/home/terry/gcs-bucket/sequence_kd_progress/${RUN_NAME}.json"
SERVER_LOG="/tmp/sequence_kd/server.log"
GCS_DATA_PATH="sequence_kd/${TEACHER_MODEL_NAME}"
PROGRESS_DIR="/home/terry/gcs-bucket/sequence_kd_progress"

printf '\n=== Sequence KD Config ===\n'
printf 'Run name: %s\n' "$RUN_NAME"
printf 'Dataset: %s (%s)\n' "$DATASET_PATH" "$DATA_SPLIT"
printf 'Teacher model: %s\n' "$TEACHER_MODEL_NAME"
printf 'Teacher checkpoint: %s\n' "$TEACHER_PARAMETERS_PATH"
printf 'Tokenizer: %s\n' "$TOKENIZER_PATH"
printf 'Progress file: %s\n' "$PROGRESS_PATH"
printf 'Server log: %s\n' "$SERVER_LOG"
printf 'Output bucket path: gs://%s/%s\n' "$BUCKET_NAME" "$GCS_DATA_PATH"
printf '==========================\n\n'

python -u multihost_runner_orig.py \
  --TPU_PREFIX=${TPU_PREFIX} \
  --RUN_NAME=${RUN_NAME} \
  --SCRIPT_DIR=$(pwd) \
  --INTERNAL_IP=true \
  --COMMAND="HF_ACCESS_TOKEN=${HF_ACCESS_TOKEN} \\
BUCKET_NAME=${BUCKET_NAME} \\
PROGRESS_PATH=${PROGRESS_PATH} \\
SERVER_LOG=${SERVER_LOG} \\
GCS_DATA_PATH=${GCS_DATA_PATH} \\
JETSTREAM_SERVER_PORT=${JETSTREAM_SERVER_PORT} \\
SERVER_READY_TIMEOUT_SEC=${SERVER_READY_TIMEOUT_SEC} \\
bash train/data/remote_sequence_kd_runner.sh"
