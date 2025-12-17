#!/bin/bash
#
# SFT (Supervised Fine-Tuning) script for Llama 1B
# Loads vanilla pretrained checkpoint (no distillation) and fine-tunes on Dolci dataset
#
# Recipe: cosine
#   - Warmup: 0.01
#   - LR decay: 0.1 (1.0 = constant, <1.0 = cosine decay)
#

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

# Model configuration
export MODEL_NAME='llama3.1-1b'
export SEQ_LEN=4096
export BATCH_SIZE=2  # per-device batch size
export GRAD_ACCUM=1

# SFT training hyperparameters
export NUM_STEPS=4000
export LR=1e-6
export MIN_LR_RATIO=0.1  # 1.0 = constant, <1.0 = cosine decay
export WARMUP_RATIO=0.01
export ASYNC_CHECKPOINTING=false

# Pretrained checkpoint to load (vanilla, no distillation)
export PRETRAINED_CHECKPOINT="gs://${BUCKET_NAME}/ckpts/pretrain/llama3.1-1b-finewebedu-vanilla-s42-50b/checkpoints/24999/items"
# Output directory for SFT checkpoints
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/ckpts/sft"

# Run naming
export RUN_NAME="sft_t4_vanilla1b_b2_s4000_1e-6_cos"
export RUN_ID="sft_t4_vanilla1b_b2_s4000_1e-6_cos"

# HuggingFace dataset configuration
export HF_PATH='/home/terry/gcs-data/datasets/Dolci-Instruct-SFT-7B'
export TRAIN_SPLIT='train'
export EVAL_SPLIT='train'  # No separate eval split, use subset of train

# Tokenizer - MUST use Instruct version for chat template
export TOKENIZER_PATH='/home/terry/gcs-data/HF_HOME/Llama-3.2-1B-Instruct'
export HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}"

echo "========================"
echo "running SFT training"
echo "parameters:"
echo "MODEL_NAME: $MODEL_NAME"
echo "SEQ_LEN: $SEQ_LEN"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GRAD_ACCUM: $GRAD_ACCUM"
echo "LR: $LR"
echo "NUM_STEPS: $NUM_STEPS"
echo "MIN_LR_RATIO: $MIN_LR_RATIO"
echo "WARMUP_RATIO: $WARMUP_RATIO"
echo "PRETRAINED_CHECKPOINT: $PRETRAINED_CHECKPOINT"
echo "BASE_OUTPUT_DIRECTORY: $BASE_OUTPUT_DIRECTORY"
echo "RUN_NAME: $RUN_NAME"
echo "HF_PATH: $HF_PATH"
echo "TOKENIZER_PATH: $TOKENIZER_PATH"
echo "start time: $(date)"
echo "========================"

wandb login --relogin 01126ae90da25bae0d86704140ac978cb9fd9c73

python -u multihost_runner_orig.py \
    --TPU_PREFIX=$TPU_PREFIX \
    --INTERNAL_IP=true \
    --COMMAND="
    export TPU_LOG_DIR=/home/terry/tpu_logs
    source ~/maxtext_env/bin/activate
    export WANDB_API_KEY='01126ae90da25bae0d86704140ac978cb9fd9c73'
    export WANDB_PROJECT=maxtext_sft_experiment
    export WANDB_NAME=${RUN_NAME}
    python3.10 -u -m MaxText.sft_trainer MaxText/configs/sft.yml \
        run_name=${RUN_NAME} \
        base_output_directory=${BASE_OUTPUT_DIRECTORY} \
        model_name=${MODEL_NAME} \
        load_parameters_path=${PRETRAINED_CHECKPOINT} \
        tokenizer_path=${TOKENIZER_PATH} \
        hf_access_token=${HF_ACCESS_TOKEN} \
        max_target_length=${SEQ_LEN} \
        per_device_batch_size=${BATCH_SIZE} \
        gradient_accumulation_steps=${GRAD_ACCUM} \
        steps=${NUM_STEPS} \
        learning_rate=${LR} \
        cosine_learning_rate_final_fraction=${MIN_LR_RATIO} \
        warmup_steps_fraction=${WARMUP_RATIO} \
        async_checkpointing=${ASYNC_CHECKPOINTING} \
        checkpoint_period=1000 \
        checkpoint_max_to_keep=1 \
        use_sft=true \
        sft_train_on_completion_only=true \
        packing=true \
        dataset_type=hf \
        hf_streaming=False \
        hf_num_proc=64 \
        hf_path=${HF_PATH} \
        train_split=${TRAIN_SPLIT} \
        hf_eval_split=${EVAL_SPLIT} \
        train_data_columns=['messages'] \
        eval_data_columns=['messages'] \
        eval_interval=-1 \
        enable_data_shuffling=true \
        data_shuffle_seed=43 \
        gcs_metrics=True \
        use_wandb=True \
        wandb_project=maxtext_sft_experiment \
        wandb_run_name=${RUN_NAME} \
        wandb_run_id=${RUN_ID}
    "
