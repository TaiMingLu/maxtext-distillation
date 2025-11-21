#!/bin/bash

set +x
set -eo pipefail

cd "${HOME}/maxtext"
export PYTHONPATH="$(pwd):${PYTHONPATH}"

python -u multihost_runner_orig.py \
  --TPU_PREFIX=taiming-v4-64 \
  --INTERNAL_IP=true \
  --RUN_NAME=maxtext \
  --COMMAND='
ROOT=$(pwd)
export TPU_LOG_DIR=/home/terry/tpu_logs
source ~/maxtext_env/bin/activate
export WANDB_API_KEY=01126ae90da25bae0d86704140ac978cb9fd9c73
export WANDB_PROJECT=maxtext_1b
export WANDB_NAME=${RUN_NAME}
export PYTHONPATH=${ROOT}:${PYTHONPATH}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_DISABLE_MOST_OPTIMIZATIONS=False

DESIRED_MODELS=(
  llama3.1-1b-finewebedu-distill-soft-T50BS42-a02-s43
  llama3.1-1b-finewebedu-distill-soft-T50BS42-a04-s43
  llama3.1-1b-finewebedu-distill-soft-T50BS42-a08-s43
)
DESIRED_STEPS=(0 2500 5000 7500 10000 12500 15000 17500 20000 22500 24999)

cd lm-evaluation-harness
python3.10 -m pip install -e .
export PYTHONPATH=${ROOT}:$(pwd):${PYTHONPATH}
cd ${ROOT}

for parent_dir in distill_pretrain pretrain; do
  if ! model_paths=$(gsutil ls "gs://taiming_us_central2_b/ckpts/${parent_dir}/" 2>/dev/null | shuf); then
    echo "Failed to list models under ${parent_dir}, skipping."
    continue
  fi

  if [[ -z "${model_paths}" ]]; then
    echo "No models found under ${parent_dir}, skipping."
    continue
  fi

  for model_path in ${model_paths}; do
    model_run_name="${model_path%/}"
    model_run_name="${model_run_name##*/}"
    [[ -z "${model_run_name}" ]] && continue

    MODEL="${model_run_name%%_*}"

    if [[ ${#DESIRED_MODELS[@]} -gt 0 ]]; then
      allowed=false
      for desired_model in "${DESIRED_MODELS[@]}"; do
        if [[ "${model_run_name}" == "${desired_model}" ]]; then
          allowed=true
          break
        fi
      done
      if [[ "${allowed}" != true ]]; then
        echo "Model ${model_run_name} not in allow list; skipping."
        continue
      fi
    fi

    if ! step_paths=$(gsutil ls "gs://taiming_us_central2_b/ckpts/${parent_dir}/${model_run_name}/checkpoints/" 2>/dev/null); then
      echo "Failed to list checkpoints for ${model_run_name}, skipping."
      continue
    fi

    if [[ -z "${step_paths}" ]]; then
      echo "No checkpoints found for ${model_run_name}, skipping."
      continue
    fi

    model_ckpt_prefix="gs://taiming_us_central2_b/ckpts/${parent_dir}/${model_run_name}/checkpoints/"

    for step_path in ${step_paths}; do
      [[ "${step_path}" == "${model_ckpt_prefix}" ]] && continue
      STEP="${step_path%/}"
      STEP="${STEP##*/}"
      [[ -z "${STEP}" ]] && continue

      if [[ ${#DESIRED_STEPS[@]} -gt 0 ]]; then
        should_process=false
        for desired_step in "${DESIRED_STEPS[@]}"; do
          if [[ "${STEP}" == "${desired_step}" ]]; then
            should_process=true
            break
          fi
        done
        if [[ "${should_process}" != true ]]; then
          echo "Step ${STEP} not in DESIRED_STEPS; skipping."
          continue
        fi
      fi

      MODEL_RUN_NAME="${model_run_name}"
      DIRECT_PARAMETER_CHECKPOINT_RUN="${MODEL_RUN_NAME}_step_${STEP}"
      CHECKPOINT_TO_CONVERT="gs://taiming_us_central2_b/ckpts/${parent_dir}/${MODEL_RUN_NAME}/checkpoints/${STEP}/items"
      UNSCANNED_CKPT_PATH="gs://taiming_us_central2_b/eval_param_only/${DIRECT_PARAMETER_CHECKPOINT_RUN}/checkpoints/0/items"
      RESULT_JSON_PATH="/home/terry/gcs-bucket/eval/results/${DIRECT_PARAMETER_CHECKPOINT_RUN}.json"

      if [[ -f "${RESULT_JSON_PATH}" ]]; then
        echo "Results already exist at ${RESULT_JSON_PATH}; skipping ${parent_dir}/${MODEL_RUN_NAME} step ${STEP}"
        continue
      fi

      echo "------------------------------------------------------------------"
      echo "Converting ${parent_dir}/${MODEL_RUN_NAME} at step ${STEP}"
      rm -rf "/home/terry/gsc-bucket/eval_param_only/${DIRECT_PARAMETER_CHECKPOINT_RUN}"

      python3.10 -u -m MaxText.generate_param_only_checkpoint MaxText/configs/base.yml \
          checkpoint_dir=gs://taiming_us_central2_b/eval_param_only \
          base_output_directory=gs://taiming_us_central2_b/eval_param_only \
          load_full_state_path=${CHECKPOINT_TO_CONVERT} \
          run_name=${DIRECT_PARAMETER_CHECKPOINT_RUN} \
          model_name=${MODEL} \
          force_unroll=true

      echo "Evaluating ${parent_dir}/${MODEL_RUN_NAME} at step ${STEP}"
      cd lm-evaluation-harness
      python3.10 -u scripts/test_orbax_eval.py ../MaxText/configs/base.yml \
          load_parameters_path=${UNSCANNED_CKPT_PATH} \
          run_name=${DIRECT_PARAMETER_CHECKPOINT_RUN} \
          per_device_batch_size=4 \
          model_name=${MODEL} \
          max_prefill_predict_length=4 \
          max_target_length=8192 \
          dataset_type=synthetic \
          dtype=bfloat16 \
          scan_layers=false \
          attention=dot_product \
          --hf_model_path=/home/terry/gcs-bucket/HF_HOME/Llama-3.1-8B \
          --add_special_tokens=False \
          --eval_save_dir=/home/terry/gcs-bucket/eval/results \
          --ppl_batch_size=1 \
          --acc_batch_size=1024
      cd ${ROOT}

    done
  done
done
'
