#!/bin/bash
# Exp 4: Cross-corpus — student on DCLM, teacher on FineWeb-Edu
# Usage: bash train/rebuttal/exp4_dclm.sh <config>
#   config: baseline | weak | same | strong | strongest
#
# Trains 1.7B student for 50B tokens (25000 steps) on DCLM data

set -euo pipefail
cd ~/maxtext

CONFIG="${1:?Usage: $0 <baseline|weak|same|strong|strongest>}"

echo "========================"
echo "Exp 4: DCLM cross-corpus — config=$CONFIG"
echo "========================"

export MODEL_NAME='llama3.1-1b'
export NUM_STEPS=25000  # 50B tokens
export SEQ_LEN=8192
export BATCH_SIZE=4
export GRAD_ACCUM=1
export LR=3.e-4
export MIN_LR_RATIO=0.1
export WARMUP_RATIO=0.05
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/rebuttal/exp4"
export DATA_FILES='/home/terry/gcs-bucket/rebuttal/data/dclm/llama3_64_array_record/*.array_record'

case "$CONFIG" in
  baseline)
    export USE_KD=false
    export KD_ALPHA=0.0
    export RUN_NAME="exp4_dclm_baseline_s43"
    ;;
  weak)
    export USE_KD=true
    export KD_ALPHA=0.2
    export KD_TEACHER_PARAMETERS_PATH="/home/terry/gcs-bucket/rebuttal/teachers/llama05b-50B-s42/checkpoint_24999/0/items"
    export TEACHER_MODEL_NAME="llama3.1-05b"
    export RUN_NAME="exp4_dclm_A05B-a02_s43"
    ;;
  same)
    export USE_KD=true
    export KD_ALPHA=0.2
    export KD_TEACHER_PARAMETERS_PATH="/home/terry/gcs-bucket/rebuttal/teachers/llama1b-50B-s42/checkpoint_24999/0/items"
    export RUN_NAME="exp4_dclm_A1B-a02_s43"
    ;;
  strong)
    export USE_KD=true
    export KD_ALPHA=0.2
    export KD_TEACHER_PARAMETERS_PATH="/home/terry/gcs-bucket/rebuttal/teachers/llama3b-50B-s42/checkpoint_24999/0/items"
    export TEACHER_MODEL_NAME="llama3.1-3b"
    export RUN_NAME="exp4_dclm_A3B-a02_s43"
    ;;
  strongest)
    export USE_KD=true
    export KD_ALPHA=0.2
    export KD_TEACHER_PARAMETERS_PATH="/home/terry/gcs-bucket/rebuttal/teachers/llama8b-50B-s42/checkpoint_24999/0/items"
    export TEACHER_MODEL_NAME="llama3.1-8b"
    export RUN_NAME="exp4_dclm_A8B-a02_s43"
    ;;
  *)
    echo "ERROR: config must be baseline|weak|same|strong|strongest"
    exit 1
    ;;
esac

export KD_TEMPERATURE=1.0

echo "RUN_NAME: $RUN_NAME"
echo "USE_KD: $USE_KD"
echo "DATA: DCLM"

source ~/maxtext_env/bin/activate

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
    checkpoint_period=2500 \
    checkpoint_max_to_keep=5 \
    gcs_metrics=True \
    packing=true \
    enable_data_shuffling=true \
    data_shuffle_seed=43 \
    init_weights_seed=43 \
    $KD_ARGS
