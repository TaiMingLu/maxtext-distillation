#!/bin/bash

set +x
set -eo pipefail

bucket_name='taiming_us_central2_b'
export TPU_PREFIX="taiming-v4-32"
export HF_MODEL_PATH='/home/terry/gcs-bucket/HF_HOME/Llama-3.1-8B'
EVAL_RESULTS_DIR="/home/terry/gcs-bucket/eval/results"

BASE_OUTPUT_DIRECTORY="gs://$bucket_name/eval_param_only"
BASE_OUTPUT_DIRECTORY_DISK="${HOME}/gsc-bucket/eval_param_only"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_DISABLE_MOST_OPTIMIZATIONS=False

cd "${HOME}/maxtext"
export PYTHONPATH="$(pwd):${PYTHONPATH}"

for parent_dir in distill_pretrain pretrain; do
  if ! model_paths=$(gsutil ls "gs://${bucket_name}/ckpts/${parent_dir}/" 2>/dev/null); then
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

    if ! step_paths=$(gsutil ls "gs://${bucket_name}/ckpts/${parent_dir}/${model_run_name}/checkpoints/" 2>/dev/null); then
      echo "Failed to list checkpoints for ${model_run_name}, skipping."
      continue
    fi

    if [[ -z "${step_paths}" ]]; then
      echo "No checkpoints found for ${model_run_name}, skipping."
      continue
    fi

    model_ckpt_prefix="gs://${bucket_name}/ckpts/${parent_dir}/${model_run_name}/checkpoints/"

    for step_path in ${step_paths}; do
      [[ "${step_path}" == "${model_ckpt_prefix}" ]] && continue
      STEP="${step_path%/}"
      STEP="${STEP##*/}"
      [[ -z "${STEP}" ]] && continue

      MODEL_RUN_NAME="${model_run_name}"
      DIRECT_PARAMETER_CHECKPOINT_RUN="${MODEL_RUN_NAME}_step_${STEP}"
      CHECKPOINT_TO_CONVERT="gs://${bucket_name}/ckpts/${parent_dir}/${MODEL_RUN_NAME}/checkpoints/${STEP}/items"
      UNSCANNED_CKPT_PATH="${BASE_OUTPUT_DIRECTORY}/${DIRECT_PARAMETER_CHECKPOINT_RUN}/checkpoints/0/items"
      RESULT_JSON_PATH="${EVAL_RESULTS_DIR}/${DIRECT_PARAMETER_CHECKPOINT_RUN}.json"

      if [[ -f "${RESULT_JSON_PATH}" ]]; then
        echo "Results already exist at ${RESULT_JSON_PATH}; skipping ${parent_dir}/${MODEL_RUN_NAME} step ${STEP}"
        continue
      fi

      echo "------------------------------------------------------------------"
      echo "Converting ${parent_dir}/${MODEL_RUN_NAME} at step ${STEP}"
      rm -rf "${BASE_OUTPUT_DIRECTORY_DISK}/${DIRECT_PARAMETER_CHECKPOINT_RUN}"

      python -u multihost_runner_orig.py \
        --TPU_PREFIX=${TPU_PREFIX} \
        --INTERNAL_IP=true \
        --RUN_NAME=maxtext \
        --COMMAND="
    ROOT=\$(pwd)
    export TPU_LOG_DIR=/home/terry/tpu_logs
    source ~/maxtext_env/bin/activate
    export WANDB_API_KEY='01126ae90da25bae0d86704140ac978cb9fd9c73'
    export WANDB_PROJECT=maxtext_1b
    export WANDB_NAME=\${RUN_NAME}
    export PYTHONPATH=\${ROOT}:\$PYTHONPATH
    python3.10 -u -m MaxText.generate_param_only_checkpoint MaxText/configs/base.yml \
        checkpoint_dir=${BASE_OUTPUT_DIRECTORY} \
        base_output_directory=${BASE_OUTPUT_DIRECTORY} \
        load_full_state_path=${CHECKPOINT_TO_CONVERT} \
        run_name=${DIRECT_PARAMETER_CHECKPOINT_RUN} \
        model_name=${MODEL} \
        force_unroll=true"

      cd ~/maxtext/
      echo "Evaluating ${parent_dir}/${MODEL_RUN_NAME} at step ${STEP}"
      python -u multihost_runner_orig.py \
        --TPU_PREFIX=${TPU_PREFIX} \
        --INTERNAL_IP=true \
        --RUN_NAME=maxtext \
        --COMMAND="
    ROOT=\$(pwd)
    cd lm-evaluation-harness
    export TPU_LOG_DIR=/home/terry/tpu_logs
    source ~/maxtext_env/bin/activate
    export WANDB_API_KEY='01126ae90da25bae0d86704140ac978cb9fd9c73'
    export WANDB_PROJECT=maxtext_1b
    export WANDB_NAME=\${RUN_NAME}
    export PYTHONPATH=\${ROOT}:\$(pwd):\$PYTHONPATH
    python3.10 -m pip install -e .
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
        --hf_model_path=${HF_MODEL_PATH} \
        --add_special_tokens=False \
        --eval_save_dir=${EVAL_RESULTS_DIR} \
        --ppl_batch_size=2 \
        --acc_batch_size=256 \
    "

      # echo "Cleaning up converted checkpoint for ${parent_dir}/${MODEL_RUN_NAME} step ${STEP}"
      # rm -rf "${BASE_OUTPUT_DIRECTORY}/${DIRECT_PARAMETER_CHECKPOINT_RUN}"

    done
  done
done
