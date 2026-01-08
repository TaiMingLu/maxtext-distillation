#!/usr/bin/env python3
"""
Generate distillation training scripts for ablate1 experiment.

Usage:
    python generate_scripts.py
    python generate_scripts.py --teacher-archs 1b 3b 8b --alphas 0.2 1.0
    python generate_scripts.py --include-vanilla  # Also generate vanilla training script
"""

import argparse
import os
from itertools import product

# Default configurations
DEFAULT_STUDENT_MODEL = "llama3.1-1b"
DEFAULT_TEACHER_ARCHS = ["05b", "1b", "3b", "8b"]
DEFAULT_TOKENS = "300B"
DEFAULT_TEACHER_SEED = 42
# DEFAULT_ALPHAS = [0.2, 1.0]
DEFAULT_ALPHAS = [0.4, 0.5, 0.6, 0.8]
DEFAULT_STUDENT_SEED = 43

# Mapping from arch shorthand to model name
ARCH_TO_MODEL = {
    "05b": "llama3.1-05b",
    "1b": "llama3.1-1b",
    "3b": "llama3.1-3b",
    "8b": "llama3.1-8b",
}

# Mapping from arch shorthand to checkpoint directory name
ARCH_TO_CKPT_DIR = {
    "05b": "llama05b",
    "1b": "llama1b",
    "3b": "llama3b",
    "8b": "llama8b",
}

# Fixed checkpoint step for 300B tokens
CHECKPOINT_STEP = 149999


DISTILL_SCRIPT_TEMPLATE = '''#!/bin/bash

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

export MODEL_NAME='{student_model}'
export NUM_STEPS=150000
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
export KD_TEACHER_PARAMETERS_PATH="/home/terry/gcs-bucket/ckpts/pretrain_param_only_v6/{teacher_ckpt_dir}-vanilla-{tokens}-s{teacher_seed}/checkpoint_{checkpoint_step}/0/items"
export TEACHER_MODEL_NAME="{teacher_model}"
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/ckpts/ablate1"
export DATA_FILES='/home/terry/gcs-data/datasets/fineweb-edu/*.array_record'

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
        checkpoint_max_to_keep=1 \\
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

VANILLA_SCRIPT_TEMPLATE = '''#!/bin/bash

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

export MODEL_NAME='{student_model}'
export NUM_STEPS=150000
export SEQ_LEN=8192
export BATCH_SIZE=4
export GRAD_ACCUM=1
export LR=3.e-4
export MIN_LR_RATIO=0.1
export WARMUP_RATIO=0.05
export ASYNC_CHECKPOINTING=false

export USE_KD=false
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/ckpts/ablate1"
export DATA_FILES='/home/terry/gcs-data/datasets/fineweb-edu/*.array_record'

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
        checkpoint_max_to_keep=2 \\
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
        use_kd=${{USE_KD}}
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
    """Generate teacher naming like A1BT300BS42."""
    # Remove 'B' suffix from tokens for naming
    tokens_num = tokens.replace("B", "")
    # Uppercase the arch (1b -> 1B, 3b -> 3B, 05b -> 05B)
    arch_upper = arch.upper()
    return f"A{arch_upper}T{tokens_num}BS{seed}"


def generate_distill_script(
    teacher_arch: str,
    tokens: str,
    teacher_seed: int,
    alpha: float,
    student_seed: int,
    student_model: str,
    output_dir: str,
) -> dict:
    """Generate a single distillation training script. Returns dict with script info."""
    teacher_naming = get_teacher_naming(teacher_arch, tokens, teacher_seed)
    alpha_str = alpha_to_str(alpha)

    # Script filename
    script_name = f"ablate1_llama1b-{teacher_naming}-{alpha_str}-s{student_seed}.sh"

    # Run name and ID
    run_name = f"ablate1_llama3.1-1b-{teacher_naming}-{alpha_str}-s{student_seed}"
    run_id = f"ablate1_llama1b_finewebedu_distill_soft_{teacher_naming}_{alpha_str}_s{student_seed}"

    # Model names
    teacher_model = ARCH_TO_MODEL[teacher_arch]
    teacher_ckpt_dir = ARCH_TO_CKPT_DIR[teacher_arch]

    content = DISTILL_SCRIPT_TEMPLATE.format(
        student_model=student_model,
        kd_alpha=alpha,
        teacher_ckpt_dir=teacher_ckpt_dir,
        tokens=tokens,
        teacher_seed=teacher_seed,
        checkpoint_step=CHECKPOINT_STEP,
        teacher_model=teacher_model,
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


def generate_vanilla_script(
    student_seed: int,
    student_model: str,
    tokens: str,
    output_dir: str,
) -> dict:
    """Generate vanilla (no distillation) training script."""
    tokens_lower = tokens.lower()  # 300B -> 300b

    script_name = f"ablate1_llama1b-finewebedu-vanilla-s{student_seed}-{tokens_lower}.sh"
    run_name = f"ablate1_llama1b-finewebedu-vanilla-s{student_seed}-{tokens_lower}"
    run_id = f"ablate1_llama1b_finewebedu_vanilla_s{student_seed}_{tokens_lower}"

    content = VANILLA_SCRIPT_TEMPLATE.format(
        student_model=student_model,
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
    # Sort by teacher arch, then alpha descending
    def sort_key(info):
        parts = info['run_id'].split('_')
        # Find the teacher part (A1BT300BS42) and alpha part
        for i, p in enumerate(parts):
            if p.startswith('A') and 'T' in p:
                teacher_part = p
                alpha_part = parts[i + 1] if i + 1 < len(parts) else 'a0'
                break
        else:
            # Vanilla script
            return ('zzz', 0)  # Sort last

        # Convert alpha for sorting (a1 -> 1.0, a02 -> 0.2)
        if alpha_part == 'a1':
            alpha = 1.0
        elif alpha_part.startswith('a'):
            alpha = int(alpha_part[1:]) / 10
        else:
            alpha = 0
        return (teacher_part, -alpha)  # negative alpha for descending

    sorted_infos = sorted(script_infos, key=sort_key)

    lines = ["tasks:"]
    current_teacher = None

    for info in sorted_infos:
        # Determine teacher from run_id
        parts = info['run_id'].split('_')
        teacher_part = None
        for p in parts:
            if p.startswith('A') and 'T' in p:
                teacher_part = p
                break

        if teacher_part != current_teacher:
            if current_teacher is not None:
                lines.append("")  # blank line between sections
            if teacher_part:
                lines.append(f"  # ============== distillation (teacher: {teacher_part}) ==============")
            else:
                lines.append(f"  # ============== vanilla training ==============")
            current_teacher = teacher_part

        lines.append(f"  - id: {info['run_id']}")
        lines.append(f"    run: bash train/ablate1/{info['script_name']}")

    content = '\n'.join(lines) + '\n'

    output_path = os.path.join(output_dir, "run_all.yaml")
    with open(output_path, "w") as f:
        f.write(content)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate distillation training scripts for ablate1")
    parser.add_argument(
        "--student-model",
        default=DEFAULT_STUDENT_MODEL,
        help=f"Student model name (default: {DEFAULT_STUDENT_MODEL})",
    )
    parser.add_argument(
        "--teacher-archs",
        nargs="+",
        default=DEFAULT_TEACHER_ARCHS,
        help=f"Teacher architectures (default: {DEFAULT_TEACHER_ARCHS})",
    )
    parser.add_argument(
        "--tokens",
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
        "--include-vanilla",
        action="store_true",
        help="Also generate vanilla (no distillation) training script",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    generated = []

    # Generate distillation scripts
    for teacher_arch, alpha in product(args.teacher_archs, args.alphas):
        if args.dry_run:
            teacher_naming = get_teacher_naming(teacher_arch, args.tokens, args.teacher_seed)
            alpha_str = alpha_to_str(alpha)
            script_name = f"ablate1_llama1b-{teacher_naming}-{alpha_str}-s{args.student_seed}.sh"
            print(f"Would generate: {script_name}")
        else:
            info = generate_distill_script(
                teacher_arch=teacher_arch,
                tokens=args.tokens,
                teacher_seed=args.teacher_seed,
                alpha=alpha,
                student_seed=args.student_seed,
                student_model=args.student_model,
                output_dir=args.output_dir,
            )
            generated.append(info)
            print(f"Generated: {info['path']}")

    # Generate vanilla script if requested
    if args.include_vanilla:
        if args.dry_run:
            tokens_lower = args.tokens.lower()
            script_name = f"ablate1_llama1b-finewebedu-vanilla-s{args.student_seed}-{tokens_lower}.sh"
            print(f"Would generate: {script_name}")
        else:
            info = generate_vanilla_script(
                student_seed=args.student_seed,
                student_model=args.student_model,
                tokens=args.tokens,
                output_dir=args.output_dir,
            )
            generated.append(info)
            print(f"Generated: {info['path']}")

    if not args.dry_run and generated:
        print(f"\nTotal scripts generated: {len(generated)}")

        # Generate run_all.yaml
        run_all_path = generate_run_all_yaml(generated, args.output_dir)
        print(f"Generated: {run_all_path}")


if __name__ == "__main__":
    main()
