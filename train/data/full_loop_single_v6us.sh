#!/bin/bash
set -euo pipefail

cd ~/maxtext
ROOT=$(pwd)

# TPU v6e environment variables - single host mode
export PJRT_DEVICE=TPU
unset JAX_COORDINATOR_ADDRESS
export JAX_PROCESS_COUNT=1
export JAX_LOCAL_DEVICE_COUNT=8

# Required environment variables
HF_ACCESS_TOKEN=${HF_ACCESS_TOKEN:?HF_ACCESS_TOKEN is required}
BUCKET_NAME=${BUCKET_NAME:-taiming_us_east1_d}

# Configuration
RUN_NAME="sequence-kd-v6us"
DATASET_PATH="/home/terry/gcs-bucket/HF_HOME/finewebedu/sample/100BT"
DATA_SPLIT="train"
TEXT_COLUMN="text"
TEACHER_MODEL_NAME="llama3.1-1b"
TEACHER_PARAMETERS_PATH="gs://${BUCKET_NAME}/ckpts/pretrain_param_only/llama3.1-1b_finewebedu_pretrain_shuffled_lr_3e-4_seed_42/checkpoint_24999/0/items"
TOKENIZER_PATH="/home/terry/gcs-bucket/HF_HOME/Llama-3.1-8B"
MAX_PREFILL_LENGTH=1024
MAX_TARGET_LENGTH=4096
GEN_BATCH_SIZE=128
SERVER_PER_DEVICE_BATCH=16
JETSTREAM_SERVER_PORT=9000
SERVER_READY_TIMEOUT_SEC=900
ENGINE_PARALLEL_FLAGS=(ici_data_parallelism=1 ici_tensor_parallelism=4 ici_fsdp_parallelism=2 ici_autoregressive_parallelism=1)
# Product: 1 × 4 × 2 × 1 = 8 (for v6e-8)
DECODE_SAMPLING_STRATEGY="weighted"
MAX_EXAMPLES=20000000

SERVER_LOG="/tmp/sequence-kd/server.log"
OUTPUT_DIR="/tmp/sequence-kd/output"
GCS_BUCKET_PATH="/home/terry/gcs-bucket/sequence_kd_data/finewebedu/sample-100BT/T50BS42"

printf '\n=== Sequence KD Config ===\n'
printf 'Run name: %s\n' "$RUN_NAME"
printf 'Bucket: %s\n' "$BUCKET_NAME"
printf 'Dataset: %s (%s)\n' "$DATASET_PATH" "$DATA_SPLIT"
printf 'Teacher model: %s\n' "$TEACHER_MODEL_NAME"
printf 'Teacher checkpoint: %s\n' "$TEACHER_PARAMETERS_PATH"
printf 'Tokenizer: %s\n' "$TOKENIZER_PATH"
printf 'Server log: %s\n' "$SERVER_LOG"
printf 'Output dir: %s\n' "$OUTPUT_DIR"
printf 'GCS bucket path: %s\n' "$GCS_BUCKET_PATH"
printf '==========================\n\n'

source ~/maxtext_env/bin/activate
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# pip install 'huggingface-hub>=0.34.0,<1.0'

mkdir -p /tmp/sequence-kd
mkdir -p "${OUTPUT_DIR}"

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
  max_prefill_predict_length=1024 \
  per_device_batch_size=${SERVER_PER_DEVICE_BATCH} \
  decode_sampling_strategy=${DECODE_SAMPLING_STRATEGY} \
  decode_sampling_temperature=0.8 \
  multi_sampling=False \
  checkpoint_storage_use_ocdbt=True \
  checkpoint_storage_use_zarr3=True \
  attention=dot_product \
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

# Give server time to warm up after listening starts
echo "[INFO] Waiting 60s for server warmup..."
sleep 60

mkdir -p "${GCS_BUCKET_PATH}"

python3 -u -m MaxText.sequence_kd_parquet \
  --input-dir ${DATASET_PATH} \
  --output-dir ${OUTPUT_DIR} \
  --tokenizer-path ${TOKENIZER_PATH} \
  --hf-access-token "${HF_ACCESS_TOKEN}" \
  --text-column ${TEXT_COLUMN} \
  --batch-size ${GEN_BATCH_SIZE} \
  --max-prefill-length ${MAX_PREFILL_LENGTH} \
  --max-target-length ${MAX_TARGET_LENGTH} \
  --jetstream-server-port ${JETSTREAM_SERVER_PORT} \
  --gcs-bucket-path ${GCS_BUCKET_PATH}
