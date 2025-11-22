#!/bin/bash
set -euo pipefail

ROOT=$(pwd)
WORKER_ID=${TPU_WORKER_ID:-0}
if [[ "${WORKER_ID}" != "0" ]]; then
  echo "[INFO] Skipping TPU worker ${WORKER_ID}"
  exit 0
fi

HF_ACCESS_TOKEN=${HF_ACCESS_TOKEN:?HF_ACCESS_TOKEN is required}
BUCKET_NAME=${BUCKET_NAME:?BUCKET_NAME is required}
PROGRESS_PATH=${PROGRESS_PATH:?PROGRESS_PATH is required}
SERVER_LOG=${SERVER_LOG:?SERVER_LOG is required}
GCS_DATA_PATH=${GCS_DATA_PATH:?GCS_DATA_PATH is required}

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
JETSTREAM_SERVER_PORT=${JETSTREAM_SERVER_PORT:-9000}
SERVER_READY_TIMEOUT_SEC=${SERVER_READY_TIMEOUT_SEC:-900}
ENGINE_PARALLEL_FLAGS=(ici_data_parallelism=2 ici_tensor_parallelism=4 ici_fsdp_parallelism=4 ici_autoregressive_parallelism=1)
DECODE_SAMPLING_STRATEGY="greedy"

printf '\n=== Sequence KD Config ===\n'
printf 'Dataset: %s (%s)\n' "$DATASET_PATH" "$DATA_SPLIT"
printf 'Teacher model: %s\n' "$TEACHER_MODEL_NAME"
printf 'Teacher checkpoint: %s\n' "$TEACHER_PARAMETERS_PATH"
printf 'Tokenizer: %s\n' "$TOKENIZER_PATH"
printf 'Progress file: %s\n' "$PROGRESS_PATH"
printf 'Server log: %s\n' "$SERVER_LOG"
printf 'Output bucket path: gs://%s/%s\n' "$BUCKET_NAME" "$GCS_DATA_PATH"
printf '==========================\n\n'

source ~/maxtext_env/bin/activate
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

mkdir -p /tmp/sequence-kd
mkdir -p "$(dirname "${PROGRESS_PATH}")"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

python3 -u -m MaxText.maxengine_server MaxText/configs/base.yml \
  model_name=${TEACHER_MODEL_NAME} \
  tokenizer_path=${TOKENIZER_PATH} \
  tokenizer_type=huggingface \
  load_parameters_path=${TEACHER_PARAMETERS_PATH} \
  max_target_length=${MAX_TARGET_LENGTH} \
  max_prefill_predict_length=${MAX_PREFILL_LENGTH} \
  per_device_batch_size=${SERVER_PER_DEVICE_BATCH} \
  decode_sampling_strategy=${DECODE_SAMPLING_STRATEGY} \
  multi_sampling=False \
  checkpoint_storage_use_ocdbt=False \
  checkpoint_storage_use_zarr3=False \
  "${ENGINE_PARALLEL_FLAGS[@]}" \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

ready=0
for ((elapsed=0; elapsed<${SERVER_READY_TIMEOUT_SEC}; elapsed+=5)); do
  if ss -ltn | grep -q ":${JETSTREAM_SERVER_PORT} " ; then
    echo "[INFO] maxengine_server is listening on port ${JETSTREAM_SERVER_PORT}"
    ready=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[ERROR] maxengine_server exited early"
    exit 1
  fi
  echo "[INFO] Waiting for maxengine_server..."
  sleep 5
done

if [[ ${ready} -ne 1 ]]; then
  echo "[ERROR] maxengine_server did not start within ${SERVER_READY_TIMEOUT_SEC}s"
  exit 1
fi

python3 -u -m MaxText.sequence_KD_data \
  --jetstream-server-port ${JETSTREAM_SERVER_PORT} \
  --dataset-path ${DATASET_PATH} \
  --data-split ${DATA_SPLIT} \
  --text-column ${TEXT_COLUMN} \
  --tokenizer-path ${TOKENIZER_PATH} \
  --hf-access-token "${HF_ACCESS_TOKEN}" \
  --batch-size ${GEN_BATCH_SIZE} \
  --max-prefill-length ${MAX_PREFILL_LENGTH} \
  --max-target-length ${MAX_TARGET_LENGTH} \
  --progress-path "${PROGRESS_PATH}" \
  upload-to-gcs \
  --gcs-bucket "${BUCKET_NAME}" \
  --gcs-data-path "${GCS_DATA_PATH}"
