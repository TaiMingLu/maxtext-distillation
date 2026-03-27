#!/bin/bash
# Exp 5 supplement: Label smoothing comparison
# Usage: bash train/rebuttal/exp5_label_smoothing.sh <epsilon>
#   epsilon: 0.0 | 0.1 | 0.3
#
# Short runs: 2B tokens (1000 steps) — enough for token-difficulty pattern.
# Same student (1.7B), same data (FineWeb-Edu), same hyperparameters as main experiments.

set -euo pipefail
cd ~/maxtext

EPSILON="${1:?Usage: $0 <0.0|0.1|0.3>}"

echo "========================"
echo "Exp 5: Label Smoothing ε=$EPSILON"
echo "TPU_PREFIX: $TPU_PREFIX"
echo "========================"

export MODEL_NAME='llama3.1-1b'
export NUM_STEPS=1000  # 2B tokens
export SEQ_LEN=8192
export BATCH_SIZE=4
export GRAD_ACCUM=1
export LR=3.e-4
export MIN_LR_RATIO=0.1
export WARMUP_RATIO=0.05
export ASYNC_CHECKPOINTING=false

# No distillation — pure pretraining with/without label smoothing
export USE_KD=false

export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/rebuttal/exp5_label_smoothing"
export RUN_NAME="ls_eps${EPSILON}_s43"
export RUN_ID="ls_eps${EPSILON//./_}_s43"

wandb login --relogin 01126ae90da25bae0d86704140ac978cb9fd9c73 2>/dev/null || true

python3.10 -u multihost_runner_orig.py \
    --TPU_PREFIX=$TPU_PREFIX \
    --INTERNAL_IP=true \
    --COMMAND="
    export TPU_LOG_DIR=/home/terry/tpu_logs
    source ~/maxtext_env/bin/activate
    export WANDB_API_KEY='01126ae90da25bae0d86704140ac978cb9fd9c73'
    export WANDB_PROJECT=maxtext_1b
    export WANDB_NAME=${RUN_NAME}
    python3.10 -u -m MaxText.train MaxText/configs/base.yml \
        run_name=${RUN_NAME} \
        base_output_directory=${BASE_OUTPUT_DIRECTORY} \
        dataset_type=grain \
        grain_train_files='/home/terry/gcs-bucket/rebuttal/data/fineweb-edu/*.array_record' \
        grain_file_type='arrayrecord' \
        grain_worker_count=1 \
        tokenize_train_data=False \
        tokenize_eval_data=False \
        max_target_length=${SEQ_LEN} \
        async_checkpointing=${ASYNC_CHECKPOINTING} \
        original_max_position_embeddings=${SEQ_LEN} \
        model_name=${MODEL_NAME} \
        steps=${NUM_STEPS} \
        per_device_batch_size=${BATCH_SIZE} \
        gradient_accumulation_steps=${GRAD_ACCUM} \
        learning_rate=${LR} \
        cosine_learning_rate_final_fraction=${MIN_LR_RATIO} \
        warmup_steps_fraction=${WARMUP_RATIO} \
        checkpoint_period=500 \
        checkpoint_max_to_keep=2 \
        gcs_metrics=True \
        use_wandb=True \
        wandb_project=maxtext_1b \
        wandb_run_name=${RUN_NAME} \
        wandb_run_id=${RUN_ID} \
        packing=true \
        enable_data_shuffling=true \
        data_shuffle_seed=43 \
        init_weights_seed=43 \
        wandb_resume=relog \
        wandb_relog_source=auto \
        label_smoothing=${EPSILON}
    "
