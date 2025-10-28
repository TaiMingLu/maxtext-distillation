#!/bin/bash

set +x
set -eo pipefail

# export MODEL='llama3.1-8b'
export MODEL='llama3.1-1b'
export MODEL_RUN_NAME="llama3.1-1b_commoncrawl_distill_pretrain_A_1B_T_50B_S42_alpha_05_seed43"
export bucket_name='taiming_us_central2_b'
export HF_MODEL_PATH='/home/terry/gcs-bucket/HF_HOME/Llama-3.1-8B'

export BASE_OUTPUT_DIRECTORY="gs://$bucket_name/ckpts/distill_pretrain_param_only" # Output directory for the checkpoint
export CHECKPOINT_TO_CONVERT="gs://$bucket_name/ckpts/distill_pretrain/${MODEL_RUN_NAME}/checkpoints/24999/items" # Input checkpoint to convert
export DIRECT_PARAMETER_CHECKPOINT_RUN="direct_generate_param_only_checkpoint_${MODEL_RUN_NAME}" # as run name 

export UNSCANNED_CKPT_PATH="${BASE_OUTPUT_DIRECTORY}/${DIRECT_PARAMETER_CHECKPOINT_RUN}/checkpoints/0/items"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_DISABLE_MOST_OPTIMIZATIONS=False

cd ~/maxtext

export TPU_PREFIX="taiming-v4-128"

export PYTHONPATH=$(pwd):$PYTHONPATH
python -u multihost_runner_orig.py \
    --TPU_PREFIX=$TPU_PREFIX \
    --INTERNAL_IP=true \
    --COMMAND="
    cd ~/maxtext
    export TPU_LOG_DIR=/home/terry/tpu_logs
    source ~/maxtext_env/bin/activate
    export WANDB_API_KEY='01126ae90da25bae0d86704140ac978cb9fd9c73'
    export WANDB_PROJECT=maxtext_1b
    export WANDB_NAME=${RUN_NAME}
    python3.10 -u -m MaxText.generate_param_only_checkpoint MaxText/configs/base.yml \
        checkpoint_dir=${BASE_OUTPUT_DIRECTORY} \
        base_output_directory=${BASE_OUTPUT_DIRECTORY} \
        load_parameters_path=${CHECKPOINT_TO_CONVERT} \
        run_name=${DIRECT_PARAMETER_CHECKPOINT_RUN} \
        model_name=$MODEL \
    force_unroll=true"

cd ~/maxtext
export PYTHONPATH=$(pwd):$PYTHONPATH
python -u multihost_runner_orig.py \
    --TPU_PREFIX=$TPU_PREFIX \
    --INTERNAL_IP=true \
    --COMMAND="
    cd ~/maxtext/lm-evaluation-harness
    export TPU_LOG_DIR=/home/terry/tpu_logs
    source ~/maxtext_env/bin/activate
    export WANDB_API_KEY='01126ae90da25bae0d86704140ac978cb9fd9c73'
    export WANDB_PROJECT=maxtext_1b
    export WANDB_NAME=${RUN_NAME}
    python3.10 -u -m scripts/test_orbax_eval.py ../MaxText/configs/base.yml \
        load_parameters_path=${UNSCANNED_CKPT_PATH} \
        run_name=forward_pass_test \
        per_device_batch_size=1 \
        model_name=${MODEL} \
        max_prefill_predict_length=4 \
        max_target_length=8192 \
        dataset_type=synthetic \
        dtype=bfloat16 \
        scan_layers=false \
        attention="dot_product" \
        --hf_model_path=${HF_MODEL_PATH} \
        --add_special_tokens=False
    "