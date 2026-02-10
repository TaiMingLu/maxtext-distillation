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


for run_dir in $(gsutil ls -d gs://${BUCKET_NAME}/ckpts/exp2/*/); do
  run_name=$(basename "$run_dir")
  echo "========================================"
  echo "Processing: $run_name"
  echo "========================================"

  python -u multihost_runner_orig.py \
      --TPU_PREFIX=$TPU_PREFIX \
      --INTERNAL_IP=true \
      --COMMAND="
      export TPU_LOG_DIR=/home/terry/tpu_logs
      source ~/maxtext_env/bin/activate
      python3.10 -u -m MaxText.generate_param_only_checkpoint MaxText/configs/base.yml \
        load_full_state_path=gs://${BUCKET_NAME}/ckpts/exp2/${run_name}/checkpoints/24999/items \
        checkpoint_dir=gs://${BUCKET_NAME}/ckpts/exp2_param_only/${run_name}/checkpoints/24999/items \
        enable_checkpointing=True async_checkpointing=False \
        model_name=llama3.1-1b
      "

  if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to convert: $run_name. Continuing..."
  else
    echo "[SUCCESS] Converted: $run_name"
  fi
done