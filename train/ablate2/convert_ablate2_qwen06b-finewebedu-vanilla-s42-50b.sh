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


python -u multihost_runner_orig.py \
    --TPU_PREFIX=$TPU_PREFIX \
    --INTERNAL_IP=true \
    --COMMAND="
    export TPU_LOG_DIR=/home/terry/tpu_logs
    source ~/maxtext_env/bin/activate
    python3.10 -u -m MaxText.generate_param_only_checkpoint MaxText/configs/base.yml \
      load_full_state_path=gs://${BUCKET_NAME}/ckpts/ablate2/qwen06b_finewebedu_vanilla_s42_50b/checkpoints/12499/items \
      checkpoint_dir=gs://${BUCKET_NAME}/ckpts/ablate2_param_only/qwen06b-vanilla-50B-s42/checkpoint_12499 \
      enable_checkpointing=True async_checkpointing=False \
      model_name=qwen3-06b
    "
