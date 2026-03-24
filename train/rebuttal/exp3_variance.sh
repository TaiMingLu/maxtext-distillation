#!/bin/bash
# Exp 3: Variance estimation — single run
# Usage: bash train/rebuttal/exp3_variance.sh <config> <seed>
#   config: baseline | weak | same | strong
#   seed:   43, 44, 45, 46, or 47
#
# Example: bash train/rebuttal/exp3_variance.sh same 44
#
# Trains 1.7B student for 10B tokens (5000 steps)

set -euo pipefail
cd ~/maxtext

CONFIG="${1:?Usage: $0 <baseline|weak|same|strong> <seed>}"
SEED="${2:?Usage: $0 <config> <seed>}"

echo "========================"
echo "Exp 3: Variance — config=$CONFIG, seed=$SEED"
echo "========================"

export MODEL_NAME='llama3.1-1b'
export NUM_STEPS=5000  # 10B tokens
export SEQ_LEN=8192
export BATCH_SIZE=4
export GRAD_ACCUM=1
export LR=3.e-4
export MIN_LR_RATIO=0.1
export WARMUP_RATIO=0.05
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/rebuttal/exp3"
export DATA_FILES='/home/terry/gcs-bucket/rebuttal/data/fineweb-edu/*.array_record'

# Config-specific settings
case "$CONFIG" in
  baseline)
    export USE_KD=false
    export KD_ALPHA=0.0
    export RUN_NAME="exp3_baseline_s${SEED}"
    ;;
  weak)
    # 0.7B teacher, 50B tokens, alpha=0.2
    export USE_KD=true
    export KD_ALPHA=0.2
    export KD_TEACHER_PARAMETERS_PATH="/home/terry/gcs-bucket/rebuttal/teachers/llama05b-50B-s42/checkpoint_24999/0/items"
    export TEACHER_MODEL_NAME="llama3.1-05b"
    export RUN_NAME="exp3_weak-A05B-a02_s${SEED}"
    ;;
  same)
    # 1.7B teacher, 50B tokens, alpha=0.4
    export USE_KD=true
    export KD_ALPHA=0.4
    export KD_TEACHER_PARAMETERS_PATH="/home/terry/gcs-bucket/rebuttal/teachers/llama1b-50B-s42/checkpoint_24999/0/items"
    export RUN_NAME="exp3_same-A1B-a04_s${SEED}"
    ;;
  strong)
    # 3.8B teacher, 50B tokens, alpha=0.6
    export USE_KD=true
    export KD_ALPHA=0.6
    export KD_TEACHER_PARAMETERS_PATH="/home/terry/gcs-bucket/rebuttal/teachers/llama3b-50B-s42/checkpoint_24999/0/items"
    export TEACHER_MODEL_NAME="llama3.1-3b"
    export RUN_NAME="exp3_strong-A3B-a06_s${SEED}"
    ;;
  *)
    echo "ERROR: config must be baseline|weak|same|strong"
    exit 1
    ;;
esac

export KD_TEMPERATURE=1.0

export RUN_ID=$(echo "$RUN_NAME" | tr '-' '_')

echo "RUN_NAME: $RUN_NAME"
echo "USE_KD: $USE_KD"
echo "KD_ALPHA: ${KD_ALPHA:-N/A}"
echo "SEED: $SEED"

source ~/maxtext_env/bin/activate
wandb login --relogin 01126ae90da25bae0d86704140ac978cb9fd9c73

# Build KD args
KD_ARGS=""
if [ "$USE_KD" = "true" ]; then
  KD_ARGS="use_kd=true kd_alpha=${KD_ALPHA} kd_temperature=${KD_TEMPERATURE} kd_teacher_parameters_path=${KD_TEACHER_PARAMETERS_PATH}"
  if [ -n "${TEACHER_MODEL_NAME:-}" ]; then
    KD_ARGS="$KD_ARGS kd_teacher_model_name=${TEACHER_MODEL_NAME}"
  fi
fi

python3.10 -u -m MaxText.train MaxText/configs/base.yml \
    run_name=${RUN_NAME} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    dataset_type=grain \
    grain_train_files=${DATA_FILES} \
    grain_file_type='arrayrecord' \
    grain_worker_count=1 \
    tokenize_train_data=False \
    tokenize_eval_data=False \
    max_target_length=${SEQ_LEN} \
    async_checkpointing=false \
    original_max_position_embeddings=${SEQ_LEN} \
    model_name=${MODEL_NAME} \
    steps=${NUM_STEPS} \
    per_device_batch_size=${BATCH_SIZE} \
    gradient_accumulation_steps=${GRAD_ACCUM} \
    learning_rate=${LR} \
    cosine_learning_rate_final_fraction=${MIN_LR_RATIO} \
    warmup_steps_fraction=${WARMUP_RATIO} \
    checkpoint_period=5000 \
    checkpoint_max_to_keep=10 \
    gcs_metrics=True \
    use_wandb=True \
    wandb_project=maxtext_1b \
    wandb_run_name=${RUN_NAME} \
    wandb_run_id=${RUN_ID} \
    wandb_resume=relog \
    wandb_relog_source=auto \
    packing=true \
    enable_data_shuffling=true \
    data_shuffle_seed=${SEED} \
    init_weights_seed=${SEED} \
    $KD_ARGS
