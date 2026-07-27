#!/bin/bash
# NeurIPS26 rebuttal training runs — one parameterized entry point.
# Usage: bash train/rebuttal_nips26/run.sh <task>
# Requires env: BUCKET_NAME, TPU_PREFIX (set by jobman queue worker).
# Outputs: gs://$BUCKET_NAME/rebuttal/nips26/<run_name>/checkpoints/<step>/items
#   -> convert with train/rebuttal/convert_to_param_only.sh nips26/<run_name> <step> llama3.1-1b
#   -> eval with train/rebuttal/eval_{ppl,acc}.sh <run_name> <step> rebuttal
#
# Task list (see rebuttal_neurips26/02_experiments.md FINAL LAUNCH SET):
#   l2_baseline_s44      baseline @50B, seed 44                     (new-seed pair)
#   l2_same_s44          same-level distill a0.2 @50B, seed 44      (new-seed pair)
#   l4_inverted_same     same-level distill a0.2 @37.5B, seed 43    (compute-matched, inverted)
#   l5_teacher_identical 1.7B teacher pretrain, seed 43 = IDENTICAL data stream as students
#   l5_distill_identical distill from identical-data teacher a0.4 @50B, seed 43
#   l3_distill_{05b,1b,3b,8b}  distill @10B from 50B teachers       (compute-matched, scaled)
#   l6_official8b_tau{1,2}     official Llama-3.1-8B teacher @10B   (overtrained-teacher temperature test)
#   l3_base_{11p7,13p3,17p4,24p7}B  baselines at matched budgets    (v4/us-central2 pool)

set -euo pipefail
cd ~/maxtext

TASK="${1:?Usage: $0 <task>}"

# Paper-standard hyperparameters (identical to main runs)
export MODEL_NAME='llama3.1-1b'
export SEQ_LEN=8192
export BATCH_SIZE=4          # x64 chips = 256 global, as in the paper
export GRAD_ACCUM=1
export LR=3.e-4
export MIN_LR_RATIO=0.1
export WARMUP_RATIO=0.05
export ASYNC_CHECKPOINTING=false
export KD_TEMPERATURE=1.0
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/rebuttal/nips26"
export DATA_FILES='/home/terry/gcs-bucket/rebuttal/data/fineweb-edu/*.array_record'
T_DIR='/home/terry/gcs-bucket/rebuttal/teachers'
USE_KD=false
TEACHER_MODEL_NAME=""

case "$TASK" in
  # ---- L2: new-seed runs @50B (pair with paper runs at seed 43) ----
  l2_baseline_s44)
    SEED=44; NUM_STEPS=25000 ;;
  l2_baseline_s45)
    SEED=45; NUM_STEPS=25000 ;;
  l2_same_s44)
    SEED=44; NUM_STEPS=25000; USE_KD=true; KD_ALPHA=0.2
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama1b-50B-s42/checkpoint_24999/0/items" ;;
  l2_same_s45)
    SEED=45; NUM_STEPS=25000; USE_KD=true; KD_ALPHA=0.2
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama1b-50B-s42/checkpoint_24999/0/items" ;;
  l2_weak_s44)
    SEED=44; NUM_STEPS=25000; USE_KD=true; KD_ALPHA=0.2; TEACHER_MODEL_NAME='llama3.1-05b'
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama05b-50B-s42/checkpoint_24999/0/items" ;;
  l2_weak_s45)
    SEED=45; NUM_STEPS=25000; USE_KD=true; KD_ALPHA=0.2; TEACHER_MODEL_NAME='llama3.1-05b'
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama05b-50B-s42/checkpoint_24999/0/items" ;;

  # ---- L4: inverted compute-match — distill on fewer tokens, total FLOPs = 50B baseline ----
  l4_inverted_same)
    SEED=43; NUM_STEPS=17880; USE_KD=true; KD_ALPHA=0.2
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama1b-50B-s42/checkpoint_24999/0/items" ;;

  # ---- L5: identical-data control (teacher sees the students' exact 50B stream) ----
  l5_teacher_identical)
    SEED=43; NUM_STEPS=25000 ;;
  l5_distill_identical)
    SEED=43; NUM_STEPS=25000; USE_KD=true; KD_ALPHA=0.4
    KD_TEACHER_PARAMETERS_PATH="/home/terry/gcs-bucket/rebuttal/param_only/nips26_l5_teacher_identical/checkpoint_24999/0/items" ;;

  # ---- L3 distill side: full compute-matched design scaled to 10B reference ----
  l3_distill_05b)
    SEED=43; NUM_STEPS=5000; USE_KD=true; KD_ALPHA=0.2; TEACHER_MODEL_NAME='llama3.1-05b'
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama05b-50B-s42/checkpoint_24999/0/items" ;;
  l3_distill_1b)
    SEED=43; NUM_STEPS=5000; USE_KD=true; KD_ALPHA=0.2
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama1b-50B-s42/checkpoint_24999/0/items" ;;
  l3_distill_3b)
    SEED=43; NUM_STEPS=5000; USE_KD=true; KD_ALPHA=0.6; TEACHER_MODEL_NAME='llama3.1-3b'
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama3b-50B-s42/checkpoint_24999/0/items" ;;
  l3_distill_8b)
    SEED=43; NUM_STEPS=5000; USE_KD=true; KD_ALPHA=0.8; TEACHER_MODEL_NAME='llama3.1-8b'
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama8b-50B-s42/checkpoint_24999/0/items" ;;

  # ---- L6: overtrained-teacher temperature test (official 15T-token 8B) ----
  l6_official8b_tau1)
    SEED=43; NUM_STEPS=5000; USE_KD=true; KD_ALPHA=0.8; TEACHER_MODEL_NAME='llama3.1-8b'
    KD_TEACHER_PARAMETERS_PATH="/home/terry/gcs-bucket/rebuttal/converted/llama3.1-8b-official/0/items" ;;
  l6_official8b_tau2)
    SEED=43; NUM_STEPS=5000; USE_KD=true; KD_ALPHA=0.8; KD_TEMPERATURE=2.0; TEACHER_MODEL_NAME='llama3.1-8b'
    KD_TEACHER_PARAMETERS_PATH="/home/terry/gcs-bucket/rebuttal/converted/llama3.1-8b-official/0/items" ;;

  # ---- Wave 2 (2026-07-27): more seeds @50B ----
  l2_baseline_s46)
    SEED=46; NUM_STEPS=25000 ;;
  l2_same_s46)
    SEED=46; NUM_STEPS=25000; USE_KD=true; KD_ALPHA=0.2
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama1b-50B-s42/checkpoint_24999/0/items" ;;
  l2_weak_s46)
    SEED=46; NUM_STEPS=25000; USE_KD=true; KD_ALPHA=0.2; TEACHER_MODEL_NAME='llama3.1-05b'
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama05b-50B-s42/checkpoint_24999/0/items" ;;
  l2_same_s44_a04)
    SEED=44; NUM_STEPS=25000; USE_KD=true; KD_ALPHA=0.4
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama1b-50B-s42/checkpoint_24999/0/items" ;;

  # ---- Wave 2: identical-data control alpha sweep ----
  l5_distill_identical_a02)
    SEED=43; NUM_STEPS=25000; USE_KD=true; KD_ALPHA=0.2
    KD_TEACHER_PARAMETERS_PATH="/home/terry/gcs-bucket/rebuttal/param_only/nips26_l5_teacher_identical/checkpoint_24999/0/items" ;;
  l5_distill_identical_a06)
    SEED=43; NUM_STEPS=25000; USE_KD=true; KD_ALPHA=0.6
    KD_TEACHER_PARAMETERS_PATH="/home/terry/gcs-bucket/rebuttal/param_only/nips26_l5_teacher_identical/checkpoint_24999/0/items" ;;

  # ---- Wave 2: inverted compute-match, remaining teachers (distill tokens = 50B-baseline FLOPs) ----
  l4_inverted_weak)
    SEED=43; NUM_STEPS=20460; USE_KD=true; KD_ALPHA=0.2; TEACHER_MODEL_NAME='llama3.1-05b'
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama05b-50B-s42/checkpoint_24999/0/items" ;;
  l4_inverted_3b)
    SEED=43; NUM_STEPS=13680; USE_KD=true; KD_ALPHA=0.6; TEACHER_MODEL_NAME='llama3.1-3b'
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama3b-50B-s42/checkpoint_24999/0/items" ;;
  l4_inverted_8b)
    SEED=43; NUM_STEPS=9640; USE_KD=true; KD_ALPHA=0.8; TEACHER_MODEL_NAME='llama3.1-8b'
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama8b-50B-s42/checkpoint_24999/0/items" ;;

  # ---- Wave 2: objective ablations at a second teacher (@10B) ----
  l6_revkl_3b)
    SEED=43; NUM_STEPS=5000; USE_KD=true; KD_ALPHA=0.6; KD_LOSS_TYPE=reverse_kl; TEACHER_MODEL_NAME='llama3.1-3b'
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama3b-50B-s42/checkpoint_24999/0/items" ;;
  l6_tau2_8b50b)
    SEED=43; NUM_STEPS=5000; USE_KD=true; KD_ALPHA=0.8; KD_TEMPERATURE=2.0; TEACHER_MODEL_NAME='llama3.1-8b'
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama8b-50B-s42/checkpoint_24999/0/items" ;;

  # ---- Wave 2: empirical label-smoothing baselines (@10B, no KD; App J.4 control) ----
  l8_labelsmooth_01)
    SEED=43; NUM_STEPS=5000; LABEL_SMOOTHING=0.1 ;;
  l8_labelsmooth_005)
    SEED=43; NUM_STEPS=5000; LABEL_SMOOTHING=0.05 ;;

  # ---- L3 baseline side: standard pretraining at compute-matched token budgets ----
  l3_base_11p7B) SEED=43; NUM_STEPS=5580  ;;
  l3_base_13p3B) SEED=43; NUM_STEPS=6340  ;;
  l3_base_17p4B) SEED=43; NUM_STEPS=8300  ;;
  l3_base_24p7B) SEED=43; NUM_STEPS=11780 ;;

  # ---- L6b: objective ablations @10B (counterparts of l3_distill_{1b,8b}) ----
  l6_revkl_1b)
    SEED=43; NUM_STEPS=5000; USE_KD=true; KD_ALPHA=0.2; KD_LOSS_TYPE=reverse_kl
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama1b-50B-s42/checkpoint_24999/0/items" ;;
  l6_topk_8b)
    SEED=43; NUM_STEPS=5000; USE_KD=true; KD_ALPHA=0.8; KD_TOP_K=100; TEACHER_MODEL_NAME='llama3.1-8b'
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama8b-50B-s42/checkpoint_24999/0/items" ;;

  # ---- L7: piecewise alpha schedules @10B (fTyB Q2). Two phases resume the same run_name;
  #      constant-alpha controls are l3_distill_{05b,1b} (alpha=0.2 = phase average). ----
  l7_rampup_1b_p1)
    SEED=43; NUM_STEPS=2500; CKPT_PERIOD=2500; USE_KD=true; KD_ALPHA=0.1; RUN_BASE=l7_rampup_1b
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama1b-50B-s42/checkpoint_24999/0/items" ;;
  l7_rampup_1b_p2)
    SEED=43; NUM_STEPS=5000; CKPT_PERIOD=2500; USE_KD=true; KD_ALPHA=0.3; RUN_BASE=l7_rampup_1b
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama1b-50B-s42/checkpoint_24999/0/items" ;;
  l7_rampdown_1b_p1)
    SEED=43; NUM_STEPS=2500; CKPT_PERIOD=2500; USE_KD=true; KD_ALPHA=0.3; RUN_BASE=l7_rampdown_1b
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama1b-50B-s42/checkpoint_24999/0/items" ;;
  l7_rampdown_1b_p2)
    SEED=43; NUM_STEPS=5000; CKPT_PERIOD=2500; USE_KD=true; KD_ALPHA=0.1; RUN_BASE=l7_rampdown_1b
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama1b-50B-s42/checkpoint_24999/0/items" ;;
  l7_rampup_05b_p1)
    SEED=43; NUM_STEPS=2500; CKPT_PERIOD=2500; USE_KD=true; KD_ALPHA=0.1; RUN_BASE=l7_rampup_05b; TEACHER_MODEL_NAME='llama3.1-05b'
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama05b-50B-s42/checkpoint_24999/0/items" ;;
  l7_rampup_05b_p2)
    SEED=43; NUM_STEPS=5000; CKPT_PERIOD=2500; USE_KD=true; KD_ALPHA=0.3; RUN_BASE=l7_rampup_05b; TEACHER_MODEL_NAME='llama3.1-05b'
    KD_TEACHER_PARAMETERS_PATH="${T_DIR}/llama05b-50B-s42/checkpoint_24999/0/items" ;;

  *) echo "ERROR: unknown task '$TASK'"; exit 1 ;;
esac

export RUN_NAME="nips26_${RUN_BASE:-$TASK}"
export RUN_ID=$(echo "$RUN_NAME" | tr '-' '_')

echo "========================================"
echo "NIPS26 task=${TASK} steps=${NUM_STEPS} seed=${SEED} use_kd=${USE_KD} alpha=${KD_ALPHA:-N/A} tau=${KD_TEMPERATURE}"
echo "TPU_PREFIX=${TPU_PREFIX} BUCKET_NAME=${BUCKET_NAME}"
echo "========================================"

wandb login --relogin 01126ae90da25bae0d86704140ac978cb9fd9c73 2>/dev/null || true

KD_ARGS=""
if [ "$USE_KD" = "true" ]; then
  KD_ARGS="use_kd=true kd_alpha=${KD_ALPHA} kd_temperature=${KD_TEMPERATURE} kd_teacher_parameters_path=${KD_TEACHER_PARAMETERS_PATH}"
  if [ -n "${TEACHER_MODEL_NAME:-}" ]; then
    KD_ARGS="$KD_ARGS kd_teacher_model_name=${TEACHER_MODEL_NAME}"
  fi
  if [ -n "${KD_LOSS_TYPE:-}" ]; then
    KD_ARGS="$KD_ARGS kd_loss_type=${KD_LOSS_TYPE}"
  fi
  if [ -n "${KD_TOP_K:-}" ]; then
    KD_ARGS="$KD_ARGS kd_top_k=${KD_TOP_K}"
  fi
fi
if [ -n "${LABEL_SMOOTHING:-}" ]; then
  KD_ARGS="$KD_ARGS label_smoothing=${LABEL_SMOOTHING}"
fi

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
        checkpoint_period=${CKPT_PERIOD:-5000} \
        checkpoint_max_to_keep=2 \
        gcs_metrics=True \
        use_wandb=True \
        wandb_project=maxtext_1b \
        wandb_run_name=${RUN_NAME} \
        wandb_run_id=${RUN_ID} \
        packing=true \
        enable_data_shuffling=true \
        data_shuffle_seed=${SEED} \
        init_weights_seed=${SEED} \
        wandb_resume=relog \
        wandb_relog_source=auto \
        $KD_ARGS
    "
