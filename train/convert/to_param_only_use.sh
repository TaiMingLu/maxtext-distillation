#!/bin/bash

cd ~/maxtext

source ~/maxtext_env/bin/activate

export BUCKET_NAME=taiming_us_central2_b
export TPU_PREFIX=taiming-v4-128_000210
gcloud config set project vision-mix
gcloud config set compute/zone us-central2-b

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


# Loop through checkpoints: 0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000
for checkpoint in 24999; do
    echo "Converting checkpoint $checkpoint..."
    
    python -u multihost_runner_orig.py \
        --TPU_PREFIX=$TPU_PREFIX \
        --INTERNAL_IP=true \
        --COMMAND="
        export TPU_LOG_DIR=/home/terry/tpu_logs
        source ~/maxtext_env/bin/activate
        python3.10 -u -m MaxText.generate_param_only_checkpoint MaxText/configs/base.yml \
          load_full_state_path=gs://taiming_us_central2_b/ckpts/pretrain/llama3.1-3b-finewebedu-vanilla-s42/checkpoints/$checkpoint/items \
          checkpoint_dir=gs://taiming_us_central2_b/ckpts/pretrain_param_only/llama3.1-3b-finewebedu-vanilla-s42/checkpoint_$checkpoint \
          enable_checkpointing=True async_checkpointing=False \
          model_name=llama3.1-3b
        "
done

echo "All checkpoints converted successfully!"
