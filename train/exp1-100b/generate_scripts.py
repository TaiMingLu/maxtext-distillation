#!/usr/bin/env python3
"""
Generate distillation training scripts for different teacher configurations.

Usage:
    python generate_scripts.py
    python generate_scripts.py --teacher-archs 1b 3b --tokens 50B 100B --alphas 0.5 1.0
"""

import argparse
import os
from itertools import product

# Default configurations
DEFAULT_TEACHER_ARCHS = ["05b", "1b", "3b", "8b"]
DEFAULT_TOKENS = ["100B"]
DEFAULT_TEACHER_SEED = 42
DEFAULT_ALPHAS = [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
DEFAULT_STUDENT_SEED = 43

# Mapping from arch shorthand to model name
ARCH_TO_MODEL = {
    "05b": "llama3.1-05b",
    "1b": "llama3.1-1b",
    "3b": "llama3.1-3b",
    "8b": "llama3.1-8b",
}

# Mapping from tokens to checkpoint step
TOKENS_TO_STEP = {
    "30B": 14999,
    "50B": 24999,
    "100B": 49999,
}

SCRIPT_TEMPLATE = '''#!/bin/bash

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
export NUM_STEPS=50000
export SEQ_LEN=8192
export BATCH_SIZE=4
export GRAD_ACCUM=1
export LR=3.e-4
export MIN_LR_RATIO=0.1
export WARMUP_RATIO=0.05
export ASYNC_CHECKPOINTING=false

export USE_KD=true
export KD_ALPHA={kd_alpha}  #KD_ALPHA=0.0 -- pure cross-entropy (no KD), KD_ALPHA=1.0 -- makes purely the KD term
export KD_TEMPERATURE=1.0
export KD_TEACHER_PARAMETERS_PATH="/home/terry/gcs-bucket/ckpts/pretrain_param_only_v6/{teacher_ckpt_name}/checkpoint_{checkpoint_step}/0/items"
export TEACHER_MODEL_NAME="{teacher_model_name}"
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/ckpts/distill_pretrain"
export DATA_FILES='/mnt/ramdisk/datasets/fineweb-edu/*.array_record'

export RUN_NAME="{run_name}"
export RUN_ID="{run_id}"

# Distillation parameters

echo "========================"
echo "running {script_name}"
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
    export WANDB_PROJECT=maxtext_1b
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
        wandb_project=maxtext_1b \\
        wandb_run_name=${{RUN_NAME}} \\
        wandb_run_id=${{RUN_ID}} \\
        packing=true \\
        enable_data_shuffling=true \\
        data_shuffle_seed={student_seed} \\
        init_weights_seed={student_seed} \\
        wandb_resume=relog \\
        wandb_relog_source=auto  \\
        use_kd=${{USE_KD}} \\
        kd_alpha=${{KD_ALPHA}} \\
        kd_temperature=${{KD_TEMPERATURE}} \\
        kd_teacher_parameters_path=${{KD_TEACHER_PARAMETERS_PATH}} \\
        kd_teacher_model_name=${{TEACHER_MODEL_NAME}}
    "
'''


def alpha_to_str(alpha: float) -> str:
    """Convert alpha float to string for naming (e.g., 0.5 -> 'a05', 1.0 -> 'a1', 0.2 -> 'a02')."""
    val = int(alpha * 10)
    if val >= 10:
        # For 1.0 -> 'a1', 2.0 -> 'a2', etc.
        return f"a{val // 10}"
    else:
        # For 0.5 -> 'a05', 0.2 -> 'a02', etc.
        return f"a{val:02d}"


def get_teacher_naming(arch: str, tokens: str, seed: int) -> str:
    """Generate teacher naming like A3BT50BS42."""
    # Remove 'B' suffix from tokens for naming
    tokens_num = tokens.replace("B", "")
    # Uppercase the arch (05b -> 05B, 1b -> 1B, etc.)
    arch_upper = arch.upper()
    return f"A{arch_upper}T{tokens_num}BS{seed}"


def get_teacher_ckpt_name(arch: str, tokens: str, seed: int) -> str:
    """Generate teacher checkpoint directory name like llama3b-vanilla-50B-s42."""
    # Map arch to size string for checkpoint path
    arch_to_size = {
        "05b": "05b",
        "1b": "1b",
        "3b": "3b",
        "8b": "8b",
    }
    size = arch_to_size[arch]
    return f"llama{size}-vanilla-{tokens}-s{seed}"


def generate_script(
    teacher_arch: str,
    tokens: str,
    teacher_seed: int,
    alpha: float,
    student_seed: int,
    output_dir: str,
) -> dict:
    """Generate a single training script. Returns dict with script info."""
    teacher_naming = get_teacher_naming(teacher_arch, tokens, teacher_seed)
    alpha_str = alpha_to_str(alpha)

    # Script filename
    script_name = f"exp1-100b_llama1b-{teacher_naming}-{alpha_str}-s{student_seed}.sh"

    # Run name and ID
    run_name = f"exp1-100b_llama3.1-1b-{teacher_naming}-{alpha_str}-s{student_seed}"
    run_id = f"exp1_100b_llama1b_finewebedu_distill_soft_{teacher_naming}_{alpha_str}_s{student_seed}"

    # Teacher model name
    teacher_model_name = ARCH_TO_MODEL[teacher_arch]

    # Teacher checkpoint info
    teacher_ckpt_name = get_teacher_ckpt_name(teacher_arch, tokens, teacher_seed)
    checkpoint_step = TOKENS_TO_STEP[tokens]

    content = SCRIPT_TEMPLATE.format(
        kd_alpha=alpha,
        teacher_ckpt_name=teacher_ckpt_name,
        checkpoint_step=checkpoint_step,
        teacher_model_name=teacher_model_name,
        run_name=run_name,
        run_id=run_id,
        script_name=script_name,
        student_seed=student_seed,
    )

    output_path = os.path.join(output_dir, script_name)
    with open(output_path, "w") as f:
        f.write(content)
    os.chmod(output_path, 0o755)

    return {
        "path": output_path,
        "script_name": script_name,
        "run_id": run_id,
        "run_name": run_name,
    }


def generate_run_all_yaml(script_infos: list, output_dir: str) -> str:
    """Generate a run_all.yaml file that lists all generated tasks."""
    lines = ["tasks:"]
    for info in script_infos:
        lines.append(f"  - id: {info['run_id']}")
        lines.append(f"    run: bash train/exp1-100b/{info['script_name']}")
        lines.append(f"    hide: true")

    content = '\n'.join(lines) + '\n'

    output_path = os.path.join(output_dir, "run_all.yaml")
    with open(output_path, "w") as f:
        f.write(content)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate distillation training scripts")
    parser.add_argument(
        "--teacher-archs",
        nargs="+",
        default=DEFAULT_TEACHER_ARCHS,
        help=f"Teacher architectures (default: {DEFAULT_TEACHER_ARCHS})",
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        default=DEFAULT_TOKENS,
        help=f"Number of tokens for teacher training (default: {DEFAULT_TOKENS})",
    )
    parser.add_argument(
        "--teacher-seed",
        type=int,
        default=DEFAULT_TEACHER_SEED,
        help=f"Teacher seed (default: {DEFAULT_TEACHER_SEED})",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=DEFAULT_ALPHAS,
        help=f"KD alpha values (default: {DEFAULT_ALPHAS})",
    )
    parser.add_argument(
        "--student-seed",
        type=int,
        default=DEFAULT_STUDENT_SEED,
        help=f"Student seed (default: {DEFAULT_STUDENT_SEED})",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for generated scripts (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    generated = []
    for teacher_arch, tokens, alpha in product(args.teacher_archs, args.tokens, args.alphas):
        if args.dry_run:
            teacher_naming = get_teacher_naming(teacher_arch, tokens, args.teacher_seed)
            alpha_str = alpha_to_str(alpha)
            script_name = f"exp1_llama1b-{teacher_naming}-{alpha_str}-s{args.student_seed}.sh"
            print(f"Would generate: {script_name}")
        else:
            info = generate_script(
                teacher_arch=teacher_arch,
                tokens=tokens,
                teacher_seed=args.teacher_seed,
                alpha=alpha,
                student_seed=args.student_seed,
                output_dir=args.output_dir,
            )
            generated.append(info)
            print(f"Generated: {info['path']}")

    if not args.dry_run:
        print(f"\nTotal scripts generated: {len(generated)}")

        # Generate run_all.yaml
        run_all_path = generate_run_all_yaml(generated, args.output_dir)
        print(f"Generated: {run_all_path}")


if __name__ == "__main__":
    main()
