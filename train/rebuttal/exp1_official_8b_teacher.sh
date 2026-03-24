#!/bin/bash
# Exp 1: Pure KD with official Llama 3.1 8B teacher (trained on 15T tokens)
# Student: 1.7B, 50B tokens, pure KD (alpha=1.0)

cd ~/maxtext

echo "========================"
echo "Exp 1: Official 8B teacher, pure KD"
echo "BUCKET_NAME: $BUCKET_NAME"
echo "TPU_PREFIX: $TPU_PREFIX"
echo "========================"

export MODEL_NAME='llama3.1-1b'
export NUM_STEPS=25000
export SEQ_LEN=8192
export BATCH_SIZE=4
export GRAD_ACCUM=1
export LR=3.e-4
export MIN_LR_RATIO=0.1
export WARMUP_RATIO=0.05

export USE_KD=true
export KD_ALPHA=1.0
export KD_TEMPERATURE=1.0
export TEACHER_MODEL_NAME='llama3.1-8b'
export KD_TEACHER_PARAMETERS_PATH="/home/terry/gcs-bucket/rebuttal/converted/llama3.1-8b-official/0/items"

export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/rebuttal/exp1"
export DATA_FILES='/home/terry/gcs-bucket/rebuttal/data/fineweb-edu/*.array_record'
export RUN_NAME="exp1_official-8b-teacher_a1-s43"
export RUN_ID="exp1_official_8b_teacher_a1_s43"

source ~/maxtext_env/bin/activate
wandb login --relogin 01126ae90da25bae0d86704140ac978cb9fd9c73

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
    checkpoint_max_to_keep=2 \
    gcs_metrics=True \
    use_wandb=True \
    wandb_project=maxtext_1b \
    wandb_run_name=${RUN_NAME} \
    wandb_run_id=${RUN_ID} \
    wandb_resume=relog \
    wandb_relog_source=auto \
    packing=true \
    enable_data_shuffling=true \
    data_shuffle_seed=43 \
    init_weights_seed=43 \
    use_kd=${USE_KD} \
    kd_alpha=${KD_ALPHA} \
    kd_temperature=${KD_TEMPERATURE} \
    kd_teacher_parameters_path=${KD_TEACHER_PARAMETERS_PATH} \
    kd_teacher_model_name=${TEACHER_MODEL_NAME}
