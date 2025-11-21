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
SERVER_READY_TIMEOUT_SEC=180
ENGINE_PARALLEL_FLAGS="ici_tensor_parallelism=8 ici_fsdp_parallelism=1 ici_autoregressive_parallelism=-1"
DECODE_SAMPLING_STRATEGY="greedy"
PROGRESS_DIR="/home/terry/gcs-bucket/sequence_kd_progress"
PROGRESS_FILE="${PROGRESS_DIR}/${RUN_NAME}.json"
LOG_ROOT="/tmp/sequence_kd"
LOG_DIR="${LOG_ROOT}/logs"
SERVER_LOG="${LOG_DIR}/server.log"
SCRIPT_PATH="${LOG_ROOT}/run.sh"
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

REMOTE_B64=$(python - <<'PY'
import base64, textwrap, os
values = dict(
    RUN_NAME=os.environ['RUN_NAME'],
    LOG_ROOT=os.environ['LOG_ROOT'],
    LOG_DIR=os.environ['LOG_DIR'],
    SERVER_LOG=os.environ['SERVER_LOG'],
    PROGRESS_DIR=os.environ['PROGRESS_DIR'],
    PROGRESS_FILE=os.environ['PROGRESS_FILE'],
    TEACHER_MODEL_NAME=os.environ['TEACHER_MODEL_NAME'],
    TEACHER_PARAMETERS_PATH=os.environ['TEACHER_PARAMETERS_PATH'],
    TOKENIZER_PATH=os.environ['TOKENIZER_PATH'],
    MAX_PREFILL_LENGTH=os.environ['MAX_PREFILL_LENGTH'],
    MAX_TARGET_LENGTH=os.environ['MAX_TARGET_LENGTH'],
    SERVER_PER_DEVICE_BATCH=os.environ['SERVER_PER_DEVICE_BATCH'],
    DECODE_SAMPLING_STRATEGY=os.environ['DECODE_SAMPLING_STRATEGY'],
    ENGINE_PARALLEL_FLAGS=os.environ['ENGINE_PARALLEL_FLAGS'],
    SERVER_READY_TIMEOUT_SEC=os.environ['SERVER_READY_TIMEOUT_SEC'],
    JETSTREAM_SERVER_PORT=os.environ['JETSTREAM_SERVER_PORT'],
    DATASET_PATH=os.environ['DATASET_PATH'],
    DATA_SPLIT=os.environ['DATA_SPLIT'],
    TEXT_COLUMN=os.environ['TEXT_COLUMN'],
    TOKENIZER_PATH_ARG=os.environ['TOKENIZER_PATH'],
    HF_ACCESS_TOKEN=os.environ['HF_ACCESS_TOKEN'],
    GEN_BATCH_SIZE=os.environ['GEN_BATCH_SIZE'],
    MAX_PREFILL_LENGTH_ARG=os.environ['MAX_PREFILL_LENGTH'],
    MAX_TARGET_LENGTH_ARG=os.environ['MAX_TARGET_LENGTH'],
    BUCKET_NAME=os.environ['BUCKET_NAME'],
    GCS_DATA_PATH=os.environ['GCS_DATA_PATH'],
)
template = """set -euo pipefail
WORKER_ID="${{TPU_WORKER_ID:-0}}"
if [[ "${{WORKER_ID}}" != "0" ]]; then
  echo "[INFO] Skipping TPU worker ${{WORKER_ID}}"
  exit 0
fi

RUN_NAME="{RUN_NAME}"
LOG_ROOT="{LOG_ROOT}"
LOG_DIR="{LOG_DIR}"
SERVER_LOG="{SERVER_LOG}"
PROGRESS_DIR="{PROGRESS_DIR}"
PROGRESS_FILE="{PROGRESS_FILE}"

rm -rf "${{LOG_ROOT}}"
mkdir -p "${{LOG_DIR}}" "${{PROGRESS_DIR}}"

cleanup() {{
  if [[ -n "${{SERVER_PID:-}}" ]]; then
    kill "${{SERVER_PID}}" >/dev/null 2>&1 || true
    wait "${{SERVER_PID}}" >/dev/null 2>&1 || true
  fi
}}
trap cleanup EXIT

python3 -u -m MaxText.maxengine_server MaxText/configs/base.yml \
  model_name={TEACHER_MODEL_NAME} \
  tokenizer_path={TOKENIZER_PATH} \
  tokenizer_type=huggingface \
  load_parameters_path={TEACHER_PARAMETERS_PATH} \
  max_target_length={MAX_TARGET_LENGTH} \
  max_prefill_predict_length={MAX_PREFILL_LENGTH} \
  per_device_batch_size={SERVER_PER_DEVICE_BATCH} \
  decode_sampling_strategy={DECODE_SAMPLING_STRATEGY} \
  multi_sampling=False \
  {ENGINE_PARALLEL_FLAGS} \
  > "${{SERVER_LOG}}" 2>&1 &
SERVER_PID=$!

ready=0
for ((elapsed=0; elapsed<{SERVER_READY_TIMEOUT_SEC}; elapsed+=5)); do
  if ss -ltn | grep -q ":{JETSTREAM_SERVER_PORT} "; then
    ready=1
    break
  fi
  if ! kill -0 "${{SERVER_PID}}" >/dev/null 2>&1; then
    echo "[ERROR] maxengine_server exited early" >&2
    exit 1
  fi
  echo "[INFO] Waiting for maxengine_server..."
  sleep 5
done
if [[ $ready -ne 1 ]]; then
  echo "[ERROR] maxengine_server did not start within {SERVER_READY_TIMEOUT_SEC}s" >&2
  exit 1
fi

python3 -u -m MaxText.sequence_KD_data \
  --jetstream-server-port {JETSTREAM_SERVER_PORT} \
  --dataset-path {DATASET_PATH} \
  --data-split {DATA_SPLIT} \
  --text-column {TEXT_COLUMN} \
  --tokenizer-path {TOKENIZER_PATH_ARG} \
  --hf-access-token {HF_ACCESS_TOKEN} \
  --batch-size {GEN_BATCH_SIZE} \
  --max-prefill-length {MAX_PREFILL_LENGTH_ARG} \
  --max-target-length {MAX_TARGET_LENGTH_ARG} \
  --progress-path {PROGRESS_FILE} \
  upload-to-gcs \
  --gcs-bucket {BUCKET_NAME} \
  --gcs-data-path {GCS_DATA_PATH}
"""
script = template.format(**values)
print(base64.b64encode(script.encode()).decode())
PY)

python -u multihost_runner_orig.py \
  --TPU_PREFIX="${TPU_PREFIX}" \
  --RUN_NAME="${RUN_NAME}" \
  --SCRIPT_DIR="$(pwd)" \
  --INTERNAL_IP=true \
  --COMMAND "mkdir -p ${LOG_ROOT} && echo ${REMOTE_B64} | base64 -d > ${SCRIPT_PATH} && chmod +x ${SCRIPT_PATH} && bash ${SCRIPT_PATH}"
