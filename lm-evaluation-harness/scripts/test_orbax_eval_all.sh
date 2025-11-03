#!/bin/bash

set +x
set -eo pipefail

BUCKET_NAME='taiming_us_central2_b'
TPU_PREFIX='taiming-v4-64'
HF_MODEL_PATH='/home/terry/gcs-bucket/HF_HOME/Llama-3.1-8B'
EVAL_RESULTS_DIR='/home/terry/gcs-bucket/eval/results'
WANDB_API_KEY='01126ae90da25bae0d86704140ac978cb9fd9c73'
WANDB_PROJECT='maxtext_1b'
RUN_NAME='maxtext'
BASE_OUTPUT_DIRECTORY="gs://${BUCKET_NAME}/eval_param_only"
BASE_OUTPUT_DIRECTORY_DISK='/home/terry/gsc-bucket/eval_param_only'
XLA_PYTHON_CLIENT_MEM_FRACTION='0.9'
XLA_PYTHON_CLIENT_ALLOCATOR='platform'
JAX_DISABLE_MOST_OPTIMIZATIONS='False'
TPU_LOG_DIR='/home/terry/tpu_logs'

cd "${HOME}/maxtext"
export PYTHONPATH="$(pwd):${PYTHONPATH}"

MULTIHOST_COMMAND=$(cat <<EOF
set +x
set -eo pipefail

export BUCKET_NAME='${BUCKET_NAME}'
export EVAL_RESULTS_DIR='${EVAL_RESULTS_DIR}'
export BASE_OUTPUT_DIRECTORY='${BASE_OUTPUT_DIRECTORY}'
export BASE_OUTPUT_DIRECTORY_DISK='${BASE_OUTPUT_DIRECTORY_DISK}'
export HF_MODEL_PATH='${HF_MODEL_PATH}'
export WANDB_API_KEY='${WANDB_API_KEY}'
export WANDB_PROJECT='${WANDB_PROJECT}'
export RUN_NAME='${RUN_NAME}'
export XLA_PYTHON_CLIENT_MEM_FRACTION='${XLA_PYTHON_CLIENT_MEM_FRACTION}'
export XLA_PYTHON_CLIENT_ALLOCATOR='${XLA_PYTHON_CLIENT_ALLOCATOR}'
export JAX_DISABLE_MOST_OPTIMIZATIONS='${JAX_DISABLE_MOST_OPTIMIZATIONS}'
export TPU_LOG_DIR='${TPU_LOG_DIR}'

cat <<'SCRIPT' > /tmp/test_orbax_eval_all_inner.sh
#!/bin/bash

set +x
set -eo pipefail

: "${BUCKET_NAME:?Set BUCKET_NAME before running}"
: "${EVAL_RESULTS_DIR:?Set EVAL_RESULTS_DIR before running}"
: "${BASE_OUTPUT_DIRECTORY:?Set BASE_OUTPUT_DIRECTORY before running}"
: "${BASE_OUTPUT_DIRECTORY_DISK:?Set BASE_OUTPUT_DIRECTORY_DISK before running}"
: "${HF_MODEL_PATH:?Set HF_MODEL_PATH before running}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY before running}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT before running}"
RUN_NAME=${RUN_NAME:-maxtext}

DESIRED_STEPS=(0 2500 5000 7500 10000 12500 15000 17500 20000 22500 24999)

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}
export JAX_DISABLE_MOST_OPTIMIZATIONS=${JAX_DISABLE_MOST_OPTIMIZATIONS:-False}
export TPU_LOG_DIR=${TPU_LOG_DIR:-/home/terry/tpu_logs}

cd "${HOME}/maxtext"
ROOT=$(pwd)
ORIG_PYTHONPATH="${PYTHONPATH}"
HARNESS_DIR="${HOME}/maxtext/lm-evaluation-harness"

if [[ -n "${ORIG_PYTHONPATH}" ]]; then
  export PYTHONPATH="${ROOT}:${ORIG_PYTHONPATH}"
else
  export PYTHONPATH="${ROOT}"
fi

source ~/maxtext_env/bin/activate
export WANDB_NAME="${RUN_NAME}"

for parent_dir in distill_pretrain pretrain; do
  if ! model_paths=$(gsutil ls "gs://${BUCKET_NAME}/ckpts/${parent_dir}/" 2>/dev/null | shuf); then
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

    if ! step_paths=$(gsutil ls "gs://${BUCKET_NAME}/ckpts/${parent_dir}/${model_run_name}/checkpoints/" 2>/dev/null); then
      echo "Failed to list checkpoints for ${model_run_name}, skipping."
      continue
    fi

    if [[ -z "${step_paths}" ]]; then
      echo "No checkpoints found for ${model_run_name}, skipping."
      continue
    fi

    model_ckpt_prefix="gs://${BUCKET_NAME}/ckpts/${parent_dir}/${model_run_name}/checkpoints/"

    for step_path in ${step_paths}; do
      [[ "${step_path}" == "${model_ckpt_prefix}" ]] && continue
      STEP="${step_path%/}"
      STEP="${STEP##*/}"
      [[ -z "${STEP}" ]] && continue

      should_process=false
      for desired_step in "${DESIRED_STEPS[@]}"; do
        if [[ "${STEP}" == "${desired_step}" ]]; then
          should_process=true
          break
        fi
      done
      if [[ "${should_process}" != true ]]; then
        continue
      fi

      MODEL_RUN_NAME="${model_run_name}"
      DIRECT_PARAMETER_CHECKPOINT_RUN="${MODEL_RUN_NAME}_step_${STEP}"
      CHECKPOINT_TO_CONVERT="gs://${BUCKET_NAME}/ckpts/${parent_dir}/${MODEL_RUN_NAME}/checkpoints/${STEP}/items"
      UNSCANNED_CKPT_PATH="${BASE_OUTPUT_DIRECTORY}/${DIRECT_PARAMETER_CHECKPOINT_RUN}/checkpoints/0/items"
      RESULT_JSON_PATH="${EVAL_RESULTS_DIR}/${DIRECT_PARAMETER_CHECKPOINT_RUN}.json"

      if [[ -f "${RESULT_JSON_PATH}" ]]; then
        echo "Results already exist at ${RESULT_JSON_PATH}; skipping ${parent_dir}/${MODEL_RUN_NAME} step ${STEP}"
        continue
      fi

      echo "------------------------------------------------------------------"
      echo "Converting ${parent_dir}/${MODEL_RUN_NAME} at step ${STEP}"
      rm -rf "${BASE_OUTPUT_DIRECTORY_DISK}/${DIRECT_PARAMETER_CHECKPOINT_RUN}"

      python3.10 -u -m MaxText.generate_param_only_checkpoint MaxText/configs/base.yml \
        checkpoint_dir="${BASE_OUTPUT_DIRECTORY}" \
        base_output_directory="${BASE_OUTPUT_DIRECTORY}" \
        load_full_state_path="${CHECKPOINT_TO_CONVERT}" \
        run_name="${DIRECT_PARAMETER_CHECKPOINT_RUN}" \
        model_name="${MODEL}" \
        force_unroll=true

      echo "Evaluating ${parent_dir}/${MODEL_RUN_NAME} at step ${STEP}"
      cd "${HARNESS_DIR}"
      if [[ -n "${ORIG_PYTHONPATH}" ]]; then
        export PYTHONPATH="${ROOT}:${HARNESS_DIR}:${ORIG_PYTHONPATH}"
      else
        export PYTHONPATH="${ROOT}:${HARNESS_DIR}"
      fi

      python3.10 -m pip install -e .
      python3.10 -u scripts/test_orbax_eval.py ../MaxText/configs/base.yml \
        load_parameters_path="${UNSCANNED_CKPT_PATH}" \
        run_name="${DIRECT_PARAMETER_CHECKPOINT_RUN}" \
        per_device_batch_size=4 \
        model_name="${MODEL}" \
        max_prefill_predict_length=4 \
        max_target_length=8192 \
        dataset_type=synthetic \
        dtype=bfloat16 \
        scan_layers=false \
        attention=dot_product \
        --hf_model_path="${HF_MODEL_PATH}" \
        --add_special_tokens=False \
        --eval_save_dir="${EVAL_RESULTS_DIR}" \
        --ppl_batch_size=4 \
        --acc_batch_size=1024

      echo "Completed evaluation for ${parent_dir}/${MODEL_RUN_NAME} step ${STEP}"
      cd "${HOME}/maxtext"
      if [[ -n "${ORIG_PYTHONPATH}" ]]; then
        export PYTHONPATH="${ROOT}:${ORIG_PYTHONPATH}"
      else
        export PYTHONPATH="${ROOT}"
      fi

      echo "Cleaning up converted checkpoint for ${parent_dir}/${MODEL_RUN_NAME} step ${STEP}"
      rm -rf "${BASE_OUTPUT_DIRECTORY}/${DIRECT_PARAMETER_CHECKPOINT_RUN}"

    done
  done
done
SCRIPT

chmod +x /tmp/test_orbax_eval_all_inner.sh
bash /tmp/test_orbax_eval_all_inner.sh

EOF
)

python -u multihost_runner_orig.py \
  --TPU_PREFIX="${TPU_PREFIX}" \
  --INTERNAL_IP=true \
  --RUN_NAME=maxtext \
  --COMMAND="${MULTIHOST_COMMAND}"
