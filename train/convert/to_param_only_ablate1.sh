#!/bin/bash

cd ~/maxtext

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


for run_dir in $(gsutil ls -d gs://${BUCKET_NAME}/ckpts/ablate1/*/ | grep '_llama'); do
  run_name=$(basename "$run_dir")
  echo "========================================"
  echo "Processing: $run_name"
  echo "========================================"

  # Extract model size from run_name
  # Format 1: ablate1_llama3.1-1b-A3BT300BS42-a1-s43 -> 1b
  # Format 2: ablate1_llama1b-finewebedu-vanilla-s43-300b -> 1b
  model_size=$(echo "$run_name" | sed -E 's/^ablate[0-9]*_llama(3\.1-)?//' | cut -d'-' -f1)
  model_name="llama3.1-${model_size}"
  echo "Detected model_name: $model_name"

  python -u multihost_runner_orig.py \
      --TPU_PREFIX=$TPU_PREFIX \
      --INTERNAL_IP=true \
      --COMMAND="
      export TPU_LOG_DIR=/home/terry/tpu_logs
      source ~/maxtext_env/bin/activate
      python3.10 -u -m MaxText.generate_param_only_checkpoint MaxText/configs/base.yml \
        load_full_state_path=gs://${BUCKET_NAME}/ckpts/ablate1/${run_name}/checkpoints/149999/items \
        checkpoint_dir=gs://${BUCKET_NAME}/ckpts/ablate1_param_only/${run_name}/checkpoints/149999/items \
        enable_checkpointing=True async_checkpointing=False \
        model_name=${model_name}
      "

  if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to convert: $run_name. Continuing..."
  else
    echo "[SUCCESS] Converted: $run_name"
  fi
done
