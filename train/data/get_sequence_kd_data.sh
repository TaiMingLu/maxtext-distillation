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
JETSTREAM_SERVER_PORT=9000
SERVER_READY_TIMEOUT_SEC=900
PROGRESS_PATH="/home/terry/gcs-bucket/sequence_kd_progress/${RUN_NAME}.json"
SERVER_LOG="/tmp/sequence_kd/server.log"
PROGRESS_DIR="/home/terry/gcs-bucket/sequence_kd_progress"

source ~/maxtext_env/bin/activate

printf '\n=== Multihost Runner Config ===\n'
printf 'Run name: %s\n' "$RUN_NAME"
printf 'Progress file: %s\n' "$PROGRESS_PATH"
printf 'Server log: %s\n' "$SERVER_LOG"
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
JETSTREAM_SERVER_PORT=${JETSTREAM_SERVER_PORT} \\
SERVER_READY_TIMEOUT_SEC=${SERVER_READY_TIMEOUT_SEC} \\
source ~/maxtext_env/bin/activate \\
bash train/data/remote_sequence_kd_runner.sh"
