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

export MODEL_NAME='llama3.1-1b'
export NUM_STEPS=50000
export SEQ_LEN=8192
export BATCH_SIZE=2
export GRAD_ACCUM=2
export LR=2.e-4
export MIN_LR_RATIO=0.1
export WARMUP_RATIO=0.05
export ASYNC_CHECKPOINTING=false
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/model_ckpts_v6e/maxtext"
export DATA_FILES='/home/terry/gcs-bucket/datasets/fineweb-edu/*.array_record'

export RUN_NAME="${MODEL_NAME}_distill_P_1B_T_1B_v6e"

cp -r /home/terry/gcs-bucket/model_ckpts/maxtext/teacher_params_only/checkpoint_500/0/items /home/terry/teacher_checkpoint


# Distillation parameters
export USE_KD=true
export KD_ALPHA=0.5
export KD_TEMPERATURE=2.0
# export KD_TEACHER_PARAMETERS_PATH="/home/terry/gcs-bucket/model_ckpts/maxtext/llama3.1-1b_seqlen_8192_bs_4_grad_accum_1_lr_2.e-4_min_lr_ratio_0.1_warmup_ratio_0.05_quadratic_warmup/checkpoints/500"
export KD_TEACHER_PARAMETERS_PATH="/home/terry/gcs-bucket/model_ckpts_v6e/maxtext/teacher_params_only/checkpoint_500/0/items"


echo "========================"
echo "running llama3_8b_1b_1e-4.sh"
echo "parameters:"
echo "MODEL_NAME: $MODEL_NAME"
echo "SEQ_LEN: $SEQ_LEN"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GRAD_ACCUM: $GRAD_ACCUM"
echo "LR: $LR"
echo "MIN_LR_RATIO: $MIN_LR_RATIO"
echo "WARMUP_RATIO: $WARMUP_RATIO"
echo "ASYNC_CHECKPOINTING: $ASYNC_CHECKPOINTING"
echo "BASE_OUTPUT_DIRECTORY: $BASE_OUTPUT_DIRECTORY"
echo "DATA_FILES: $DATA_FILES"
echo "RUN_NAME: $RUN_NAME"
echo "TPU_PREFIX: $TPU_PREFIX"
echo "BUCKET_NAME: $BUCKET_NAME"
echo "USE_KD: $USE_KD"
echo "KD_ALPHA: $KD_ALPHA"
echo "KD_TEMPERATURE: $KD_TEMPERATURE"
echo "KD_TEACHER_PARAMETERS_PATH: $KD_TEACHER_PARAMETERS_PATH"
echo "========================"

wandb login --relogin 01126ae90da25bae0d86704140ac978cb9fd9c73

python -u multihost_runner_orig.py \
    --TPU_PREFIX=$TPU_PREFIX \
    --INTERNAL_IP=true \
    --COMMAND="
    export TPU_LOG_DIR=/home/terry/tpu_logs
    source ~/maxtext_env/bin/activate
    export WANDB_API_KEY='01126ae90da25bae0d86704140ac978cb9fd9c73'
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
        checkpoint_max_to_keep=100 \
        use_wandb=False \
        wandb_project=maxtext \
        wandb_run_name=${RUN_NAME} \
        packing=false \
        enable_data_shuffling=false \
        data_shuffle_seed=0 \
        init_weights_seed=0 \
        use_wandb=False \
        use_kd=${USE_KD} \
        kd_alpha=${KD_ALPHA} \
        kd_temperature=${KD_TEMPERATURE} \
        kd_teacher_parameters_path=${KD_TEACHER_PARAMETERS_PATH}
    "