#!/bin/bash

cd ~/maxtext

source ~/maxtext_env/bin/activate

export BUCKET_NAME=taiming_us_central1_b
export TPU_PREFIX=taiming-v6e-64_020006
gcloud config set project vision-mix
gcloud config set compute/zone us-central1-b

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



python3.10 -u -m MaxText.generate_param_only_checkpoint MaxText/configs/base.yml \
  load_full_state_path=gs://taiming_us_central1_b/ckpts/pretrain/llama3.1-05b-finewebedu-vanilla-s42_v4/checkpoints/24999/items \
  checkpoint_dir=gs://taiming_us_central1_b/ckpts/pretrain_param_only_v6/llama05b-vanilla-50B-s42/checkpoint_24999 \
  enable_checkpointing=True async_checkpointing=False \
  model_name=llama3.1-05b
python3.10 -u -m MaxText.generate_param_only_checkpoint MaxText/configs/base.yml \
  load_full_state_path=gs://taiming_us_central1_b/ckpts/pretrain/llama3.1-1b-finewebedu-vanilla-s42_v4/checkpoints/24999/items \
  checkpoint_dir=gs://taiming_us_central1_b/ckpts/pretrain_param_only_v6/llama1b-vanilla-50B-s42/checkpoint_24999 \
  enable_checkpointing=True async_checkpointing=False \
  model_name=llama3.1-1b
python3.10 -u -m MaxText.generate_param_only_checkpoint MaxText/configs/base.yml \
  load_full_state_path=gs://taiming_us_central1_b/ckpts/pretrain/llama3.1-3b-finewebedu-vanilla-s42_v4/checkpoints/24999/items \
  checkpoint_dir=gs://taiming_us_central1_b/ckpts/pretrain_param_only_v6/llama3b-vanilla-50B-s42/checkpoint_24999 \
  enable_checkpointing=True async_checkpointing=False \
  model_name=llama3.1-3b
python3.10 -u -m MaxText.generate_param_only_checkpoint MaxText/configs/base.yml \
  load_full_state_path=gs://taiming_us_central1_b/ckpts/pretrain/llama3.1-05b-finewebedu-vanilla-100B-s42_v6/checkpoints/49999/items \
  checkpoint_dir=gs://taiming_us_central1_b/ckpts/pretrain_param_only_v6/llama05b-vanilla-100B-s42/checkpoint_49999 \
  enable_checkpointing=True async_checkpointing=False \
  model_name=llama3.1-05b
python3.10 -u -m MaxText.generate_param_only_checkpoint MaxText/configs/base.yml \
  load_full_state_path=gs://taiming_us_central1_b/ckpts/pretrain/llama3.1-1b-finewebedu-vanilla-100B-s42_v6/checkpoints/49999/items \
  checkpoint_dir=gs://taiming_us_central1_b/ckpts/pretrain_param_only_v6/llama1b-vanilla-100B-s42/checkpoint_49999 \
  enable_checkpointing=True async_checkpointing=False \
  model_name=llama3.1-1b
python3.10 -u -m MaxText.generate_param_only_checkpoint MaxText/configs/base.yml \
  load_full_state_path=gs://taiming_us_central1_b/ckpts/pretrain/llama3.1-3b-finewebedu-vanilla-100B-s42_v6/checkpoints/49999/items \
  checkpoint_dir=gs://taiming_us_central1_b/ckpts/pretrain_param_only_v6/llama3b-vanilla-100B-s42/checkpoint_49999 \
  enable_checkpointing=True async_checkpointing=False \
  model_name=llama3.1-3b
python3.10 -u -m MaxText.generate_param_only_checkpoint MaxText/configs/base.yml \
  load_full_state_path=gs://taiming_us_central1_b/ckpts/pretrain/llama3.1-3b-finewebedu-vanilla-30B-s42_v6/checkpoints/14999/items \
  checkpoint_dir=gs://taiming_us_central1_b/ckpts/pretrain_param_only_v6/llama3b-vanilla-30B-s42/checkpoint_14999 \
  enable_checkpointing=True async_checkpointing=False \
  model_name=llama3.1-3b




echo "All checkpoints converted successfully!"
