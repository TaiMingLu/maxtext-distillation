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

RUN_NAME="sequence-kd-$(date +%Y%m%d-%H%M%S)"
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
SERVER_READY_TIMEOUT_SEC=180
ENGINE_PARALLEL_FLAGS="ici_tensor_parallelism=8 ici_fsdp_parallelism=1 ici_autoregressive_parallelism=-1"
DECODE_SAMPLING_STRATEGY="greedy"
PROGRESS_FILE="/home/terry/gcs-bucket/sequence_kd_progress/${RUN_NAME}.json"
SERVER_LOG="$HOME/sequence_kd_logs/${RUN_NAME}_server.log"
GCS_DATA_PATH="sequence_kd/${TEACHER_MODEL_NAME}"

printf '\n=== Sequence KD Config ===\n'
printf 'Run name: %s\n' "$RUN_NAME"
printf 'Dataset: %s (%s)\n' "$DATASET_PATH" "$DATA_SPLIT"
printf 'Teacher model: %s\n' "$TEACHER_MODEL_NAME"
printf 'Teacher checkpoint: %s\n' "$TEACHER_PARAMETERS_PATH"
printf 'Tokenizer: %s\n' "$TOKENIZER_PATH"
printf 'Progress file: %s\n' "$PROGRESS_FILE"
printf 'Server log: %s\n' "$SERVER_LOG"
printf 'Output bucket path: gs://%s/%s\n' "$BUCKET_NAME" "$GCS_DATA_PATH"
printf '==========================\n\n'

python -u multihost_runner_orig.py \
  --TPU_PREFIX="${TPU_PREFIX}" \
  --RUN_NAME="${RUN_NAME}" \
  --SCRIPT_DIR="$(pwd)" \
  --INTERNAL_IP=true \
  --COMMAND "$(cat <<EOF_REMOTE
set -euo pipefail

WORKER_ID="\${TPU_WORKER_ID:-0}"
if [[ "\${WORKER_ID}" != "0" ]]; then
  echo "[INFO] Skipping TPU worker \${WORKER_ID}"
  exit 0
fi

mkdir -p "$(dirname "$SERVER_LOG")" "$(dirname "$PROGRESS_FILE")"

python3 -u -m MaxText.maxengine_server MaxText/configs/base.yml \\
  model_name=${TEACHER_MODEL_NAME} \\
  tokenizer_path=${TOKENIZER_PATH} \\
  tokenizer_type=huggingface \\
  load_parameters_path=${TEACHER_PARAMETERS_PATH} \\
  max_target_length=${MAX_TARGET_LENGTH} \\
  max_prefill_predict_length=${MAX_PREFILL_LENGTH} \\
  per_device_batch_size=${SERVER_PER_DEVICE_BATCH} \\
  decode_sampling_strategy=${DECODE_SAMPLING_STRATEGY} \\
  multi_sampling=False \\
  ${ENGINE_PARALLEL_FLAGS} \\
  > ${SERVER_LOG} 2>&1 &
SERVER_PID=$!

ready=0
for ((elapsed=0; elapsed<${SERVER_READY_TIMEOUT_SEC}; elapsed+=5)); do
  if ss -ltn | grep -q ":${JETSTREAM_SERVER_PORT} "; then
    ready=1
    break
  fi
  if ! kill -0 ${SERVER_PID} >/dev/null 2>&1; then
    echo "[ERROR] maxengine_server exited early" >&2
    exit 1
  fi
  echo "[INFO] Waiting for maxengine_server..."
  sleep 5

done
if [[ ${ready} -ne 1 ]]; then
  echo "[ERROR] maxengine_server did not start within ${SERVER_READY_TIMEOUT_SEC}s" >&2
  kill ${SERVER_PID} >/dev/null 2>&1 || true
  exit 1
fi

python3 -u -m MaxText.sequence_KD_data \\
  --jetstream-server-port ${JETSTREAM_SERVER_PORT} \\
  --dataset-path ${DATASET_PATH} \\
  --data-split ${DATA_SPLIT} \\
  --text-column ${TEXT_COLUMN} \\
  --tokenizer-path ${TOKENIZER_PATH} \\
  --hf-access-token ${HF_ACCESS_TOKEN} \\
  --batch-size ${GEN_BATCH_SIZE} \\
  --max-prefill-length ${MAX_PREFILL_LENGTH} \\
  --max-target-length ${MAX_TARGET_LENGTH} \\
  --progress-path ${PROGRESS_FILE} \\
  upload-to-gcs \\
  --gcs-bucket ${BUCKET_NAME} \\
  --gcs-data-path ${GCS_DATA_PATH}

kill ${SERVER_PID} >/dev/null 2>&1 || true
wait ${SERVER_PID} >/dev/null 2>&1 || true
EOF_REMOTE
)"
