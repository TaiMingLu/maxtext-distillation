#!/usr/bin/env python3
"""
Script to generate ablation experiment shell scripts and YAML config.
This script generates training scripts for knowledge distillation experiments
with different teacher models, alpha values, and top-k settings.
"""

import os
from pathlib import Path

# Configuration
EXPERIMENT_NAME = "ablate4"
OUTPUT_DIR = Path(__file__).parent

# Teacher model configurations: {code: (model_name, checkpoint_path_segment)}
TEACHER_CONFIGS = {
    "A05": ("llama3.1-05b", "llama05b-vanilla-50B-s42"),
    "A1": ("llama3.1-1b", "llama1b-vanilla-50B-s42"),
    "A3": ("llama3.1-3b", "llama3b-vanilla-50B-s42"),
    "A8": ("llama3.1-8b", "llama8b-vanilla-50B-s42"),
}

# Alpha configurations: {code: (alpha_value, wandb_project)}
ALPHA_CONFIGS = {
    "a02": (0.2, "maxtext_exp"),
    "a1": (1.0, "maxtext_1b"),
}

# Top-k configurations: {suffix: top_k_value}
# Calculated as round(128256 * percentage)
VOCAB_SIZE = 128256
TOPK_CONFIGS = {
    None: None,                              # base case, no top-k (full vocab)
    "k001": round(VOCAB_SIZE * 0.01),        # 1% = 1283
    "k005": round(VOCAB_SIZE * 0.05),        # 5% = 6413
    "k02": round(VOCAB_SIZE * 0.2),          # 20% = 25651
    "k05": round(VOCAB_SIZE * 0.5),          # 50% = 64128
}

# Fixed parameters
SEED = 43
DATA_SEED = "BT50BS42"  # Base training 50B, base seed 42
KD_TEMPERATURE = 1.0  # Fixed temperature


def generate_shell_script(teacher_code, alpha_code, topk_suffix):
    """Generate a shell script for a specific experiment configuration."""

    teacher_model, teacher_ckpt = TEACHER_CONFIGS[teacher_code]
    alpha_value, wandb_project = ALPHA_CONFIGS[alpha_code]
    top_k_value = TOPK_CONFIGS[topk_suffix]

    # Build run name components
    config_name = f"{teacher_code}{DATA_SEED}-{alpha_code}-s{SEED}"
    if topk_suffix:
        config_name += f"-{topk_suffix}"

    run_name = f"exp1_llama3.1-1b-{config_name}"
    run_id = f"exp1_llama1b_finewebedu_distill_soft_{config_name.replace('-', '_')}"
    script_name = f"{EXPERIMENT_NAME}_llama1b-{config_name}.sh"

    # Build the kd_top_k export and flag
    topk_export = ""
    topk_flag = ""
    if top_k_value is not None:
        topk_export = f"export KD_TOP_K={top_k_value}\n"
        topk_flag = "        kd_top_k=${KD_TOP_K} \\\n"

    script_content = f'''#!/bin/bash

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
for var in "${{required_vars[@]}}"; do
  if [[ -z "${{!var:-}}" ]]; then
    echo "[ERROR] $var is not set"
    exit 1
  fi
done

export MODEL_NAME='llama3.1-1b'
export NUM_STEPS=25000
export SEQ_LEN=8192
export BATCH_SIZE=4
export GRAD_ACCUM=1
export LR=3.e-4
export MIN_LR_RATIO=0.1
export WARMUP_RATIO=0.05
export ASYNC_CHECKPOINTING=false

export USE_KD=true
export KD_ALPHA={alpha_value}  #KD_ALPHA=0.0 -- pure cross-entropy (no KD), KD_ALPHA=1.0 -- makes purely the KD term
export KD_TEMPERATURE={KD_TEMPERATURE}
export KD_TEACHER_PARAMETERS_PATH="/home/terry/gcs-bucket/ckpts/pretrain_param_only_v6/{teacher_ckpt}/checkpoint_24999/0/items"
export TEACHER_MODEL_NAME="{teacher_model}"
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/ckpts/{EXPERIMENT_NAME}"
export DATA_FILES='/home/terry/gcs-data/datasets/fineweb-edu/*.array_record'
{topk_export}
export RUN_NAME="{run_name}"
export RUN_ID="{run_id}"

# Distillation parameters

echo "========================"
echo "running {run_name.replace('exp1_llama3.1-1b', 'exp1_llama1b')}.sh"
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
echo "start time: $(date)"
echo "========================"

wandb login --relogin 01126ae90da25bae0d86704140ac978cb9fd9c73


python -u multihost_runner_orig.py \\
    --TPU_PREFIX=$TPU_PREFIX \\
    --INTERNAL_IP=true \\
    --COMMAND="
    export TPU_LOG_DIR=/home/terry/tpu_logs
    source ~/maxtext_env/bin/activate
    export WANDB_API_KEY='01126ae90da25bae0d86704140ac978cb9fd9c73'
    export WANDB_PROJECT={wandb_project}
    export WANDB_NAME=${{RUN_NAME}}
    python3.10 -u -m MaxText.train MaxText/configs/base.yml \\
        run_name=${{RUN_NAME}} \\
        base_output_directory=${{BASE_OUTPUT_DIRECTORY}} \\
        dataset_type=grain \\
        grain_train_files=${{DATA_FILES}} \\
        grain_file_type='arrayrecord' \\
        grain_worker_count=1 \\
        tokenize_train_data=False \\
        tokenize_eval_data=False \\
        max_target_length=${{SEQ_LEN}} \\
        async_checkpointing=${{ASYNC_CHECKPOINTING}} \\
        original_max_position_embeddings=${{SEQ_LEN}} \\
        model_name=${{MODEL_NAME}} \\
        steps=${{NUM_STEPS}} \\
        per_device_batch_size=${{BATCH_SIZE}} \\
        gradient_accumulation_steps=${{GRAD_ACCUM}} \\
        learning_rate=${{LR}} \\
        cosine_learning_rate_final_fraction=${{MIN_LR_RATIO}} \\
        warmup_steps_fraction=${{WARMUP_RATIO}} \\
        checkpoint_period=2500 \\
        checkpoint_max_to_keep=10 \\
        gcs_metrics=True \\
        use_wandb=True \\
        wandb_project={wandb_project} \\
        wandb_run_name=${{RUN_NAME}} \\
        wandb_run_id=${{RUN_ID}} \\
        packing=true \\
        enable_data_shuffling=true \\
        data_shuffle_seed={SEED} \\
        init_weights_seed={SEED} \\
        wandb_resume=relog \\
        wandb_relog_source=auto  \\
        use_kd=${{USE_KD}} \\
        kd_alpha=${{KD_ALPHA}} \\
        kd_temperature=${{KD_TEMPERATURE}} \\
        kd_teacher_parameters_path=${{KD_TEACHER_PARAMETERS_PATH}} \\
{topk_flag}        kd_teacher_model_name=${{TEACHER_MODEL_NAME}}
    "
'''

    return script_name, script_content


def generate_yaml_config(all_scripts):
    """Generate the run_all.yaml configuration file."""

    yaml_content = "tasks:\n"

    # Group scripts by teacher and alpha combination
    for teacher_code in TEACHER_CONFIGS.keys():
        for alpha_code in ALPHA_CONFIGS.keys():
            section_name = f"{teacher_code}{DATA_SEED}-{alpha_code}"
            yaml_content += f"  # ============== {section_name} ==============\n"

            # Add each top-k variant
            for topk_suffix in TOPK_CONFIGS.keys():
                config_name = f"{teacher_code}{DATA_SEED}-{alpha_code}-s{SEED}"
                if topk_suffix:
                    config_name += f"-{topk_suffix}"

                task_id = f"{EXPERIMENT_NAME}_llama1b_{config_name.replace('-', '_')}"
                script_name = f"{EXPERIMENT_NAME}_llama1b-{config_name}.sh"

                yaml_content += f"  - id: {task_id}\n"
                yaml_content += f"    run: bash train/{EXPERIMENT_NAME}/{script_name}\n"

            yaml_content += "\n"

    return yaml_content


def main():
    """Main function to generate all scripts and YAML config."""

    all_scripts = []

    # Generate all shell scripts
    for teacher_code in TEACHER_CONFIGS.keys():
        for alpha_code in ALPHA_CONFIGS.keys():
            for topk_suffix in TOPK_CONFIGS.keys():
                script_name, script_content = generate_shell_script(
                    teacher_code, alpha_code, topk_suffix
                )

                # Write the shell script
                script_path = OUTPUT_DIR / script_name
                with open(script_path, 'w') as f:
                    f.write(script_content)
                os.chmod(script_path, 0o755)

                all_scripts.append(script_name)
                print(f"Generated: {script_name}")

    # Generate YAML config
    yaml_content = generate_yaml_config(all_scripts)
    yaml_path = OUTPUT_DIR / "run_all.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Generated: run_all.yaml")

    print(f"\nTotal scripts generated: {len(all_scripts)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
