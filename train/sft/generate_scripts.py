#!/usr/bin/env python3
"""
Generate SFT training scripts for distillation experiment checkpoints.

Usage:
    python generate_scripts.py
    python generate_scripts.py --exp exp1 --teacher-archs 1b 3b --tokens 50B --alphas 0.5 1.0
"""

import argparse
import os
from itertools import product

# Experiment configurations
# exp1: fixed tokens (50B), varying alpha
EXP1_CONFIG = {
    "teacher_archs": ["05b", "1b", "3b", "8b"],
    "tokens": ["50B"],
    "alphas": [0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
}

# exp2: varying tokens, fewer alphas
EXP2_CONFIG = {
    "teacher_archs": ["05b", "1b", "3b", "8b"],
    "tokens": ["30B", "50B", "100B"],
    "alphas": [0.5, 1.0],
}

# Always use step 24999 for SFT (final checkpoint)
SFT_CHECKPOINT_STEP = 24999

# Vanilla baselines (no distillation)
VANILLA_CONFIG = {
    "checkpoints": [
        {
            "name": "llama3.1-1b-finewebedu-vanilla-s42-50b",
            "path": "ckpts/pretrain/llama3.1-1b-finewebedu-vanilla-s42-50b",
        },
    ],
}

DEFAULT_TEACHER_SEED = 42
DEFAULT_STUDENT_SEED = 43

SCRIPT_TEMPLATE = '''#!/bin/bash
#
# SFT (Supervised Fine-Tuning) script for Llama 1B
# Loads pretrained checkpoint from {exp_name} and fine-tunes on Dolci dataset
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
for var in "${{required_vars[@]}}"; do
  if [[ -z "${{!var:-}}" ]]; then
    echo "[ERROR] $var is not set"
    exit 1
  fi
done

# Model configuration
export MODEL_NAME='llama3.1-1b'
export SEQ_LEN=4096
export BATCH_SIZE=8  # per-device batch size; on v6e-32 (32 chips): 8 * 32 * 4096 = ~1M tokens/step
export GRAD_ACCUM=1

# SFT training hyperparameters (typically lower LR than pretraining)
export NUM_STEPS=1000  # ~1B tokens total (1M tokens/step * 1000 steps)
export LR=1.e-5
export MIN_LR_RATIO=1.0  # Constant LR (no decay)
export WARMUP_RATIO=0.1
export ASYNC_CHECKPOINTING=false

# Pretrained checkpoint to load (from {exp_name} distillation run)
export PRETRAINED_CHECKPOINT="gs://${{BUCKET_NAME}}/ckpts/distill_pretrain/{pretrain_run_name}/checkpoints/{checkpoint_step}/items"
# Output directory for SFT checkpoints
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/ckpts/sft"

# Run naming
export RUN_NAME="{run_name}"
export RUN_ID="{run_id}"

# HuggingFace dataset configuration
# Dolci-Instruct-SFT-7B: 2.15M samples with messages format [{{role, content}}, ...]
# Using local copy to avoid re-downloading each run
export HF_PATH='/home/terry/gcs-bucket/datasets/Dolci-Instruct-SFT-7B'
export TRAIN_SPLIT='train'
export EVAL_SPLIT='train'  # No separate eval split, use subset of train

# Tokenizer - MUST use Instruct version for chat template
# The base model (Llama-3.2-1B) does NOT have a chat template
# The Instruct model has the chat template needed for SFT on conversational data
export TOKENIZER_PATH='/home/terry/gcs-bucket/HF_HOME/Llama-3.2-1B-Instruct'
# Set your HuggingFace token (required for gated Llama models)
export HF_ACCESS_TOKEN="${{HF_ACCESS_TOKEN:-}}"

echo "========================"
echo "running SFT training"
echo "parameters:"
echo "MODEL_NAME: $MODEL_NAME"
echo "SEQ_LEN: $SEQ_LEN"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GRAD_ACCUM: $GRAD_ACCUM"
echo "LR: $LR"
echo "NUM_STEPS: $NUM_STEPS"
echo "PRETRAINED_CHECKPOINT: $PRETRAINED_CHECKPOINT"
echo "BASE_OUTPUT_DIRECTORY: $BASE_OUTPUT_DIRECTORY"
echo "RUN_NAME: $RUN_NAME"
echo "HF_PATH: $HF_PATH"
echo "TOKENIZER_PATH: $TOKENIZER_PATH"
echo "start time: $(date)"
echo "========================"

python -u multihost_runner_orig.py \\
    --TPU_PREFIX=$TPU_PREFIX \\
    --INTERNAL_IP=true \\
    --COMMAND="
    export TPU_LOG_DIR=/home/terry/tpu_logs
    source ~/maxtext_env/bin/activate
    python3.10 -u -m MaxText.sft_trainer MaxText/configs/sft.yml \\
        run_name=${{RUN_NAME}} \\
        base_output_directory=${{BASE_OUTPUT_DIRECTORY}} \\
        model_name=${{MODEL_NAME}} \\
        load_parameters_path=${{PRETRAINED_CHECKPOINT}} \\
        tokenizer_path=${{TOKENIZER_PATH}} \\
        hf_access_token=${{HF_ACCESS_TOKEN}} \\
        max_target_length=${{SEQ_LEN}} \\
        per_device_batch_size=${{BATCH_SIZE}} \\
        gradient_accumulation_steps=${{GRAD_ACCUM}} \\
        steps=${{NUM_STEPS}} \\
        learning_rate=${{LR}} \\
        cosine_learning_rate_final_fraction=${{MIN_LR_RATIO}} \\
        warmup_steps_fraction=${{WARMUP_RATIO}} \\
        async_checkpointing=${{ASYNC_CHECKPOINTING}} \\
        checkpoint_period=1000 \\
        checkpoint_max_to_keep=1 \\
        use_sft=true \\
        sft_train_on_completion_only=true \\
        packing=true \\
        dataset_type=hf \\
        hf_path=${{HF_PATH}} \\
        train_split=${{TRAIN_SPLIT}} \\
        hf_eval_split=${{EVAL_SPLIT}} \\
        train_data_columns=['messages'] \\
        eval_data_columns=['messages'] \\
        eval_interval=-1 \\
        enable_data_shuffling=true \\
        data_shuffle_seed=43 \\
        gcs_metrics=True
    "
'''


VANILLA_SCRIPT_TEMPLATE = '''#!/bin/bash
#
# SFT (Supervised Fine-Tuning) script for Llama 1B
# Loads vanilla pretrained checkpoint (no distillation) and fine-tunes on Dolci dataset
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
for var in "${{required_vars[@]}}"; do
  if [[ -z "${{!var:-}}" ]]; then
    echo "[ERROR] $var is not set"
    exit 1
  fi
done

# Model configuration
export MODEL_NAME='llama3.1-1b'
export SEQ_LEN=4096
export BATCH_SIZE=8  # per-device batch size; on v6e-32 (32 chips): 8 * 32 * 4096 = ~1M tokens/step
export GRAD_ACCUM=1

# SFT training hyperparameters (typically lower LR than pretraining)
export NUM_STEPS=1000  # ~1B tokens total (1M tokens/step * 1000 steps)
export LR=1.e-5
export MIN_LR_RATIO=1.0  # Constant LR (no decay)
export WARMUP_RATIO=0.1
export ASYNC_CHECKPOINTING=false

# Pretrained checkpoint to load (vanilla, no distillation)
export PRETRAINED_CHECKPOINT="gs://${{BUCKET_NAME}}/{ckpt_path}/checkpoints/{checkpoint_step}/items"
# Output directory for SFT checkpoints
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/ckpts/sft"

# Run naming
export RUN_NAME="{run_name}"
export RUN_ID="{run_id}"

# HuggingFace dataset configuration
# Dolci-Instruct-SFT-7B: 2.15M samples with messages format [{{role, content}}, ...]
# Using local copy to avoid re-downloading each run
export HF_PATH='/home/terry/gcs-bucket/datasets/Dolci-Instruct-SFT-7B'
export TRAIN_SPLIT='train'
export EVAL_SPLIT='train'  # No separate eval split, use subset of train

# Tokenizer - MUST use Instruct version for chat template
# The base model (Llama-3.2-1B) does NOT have a chat template
# The Instruct model has the chat template needed for SFT on conversational data
export TOKENIZER_PATH='/home/terry/gcs-bucket/HF_HOME/Llama-3.2-1B-Instruct'
# Set your HuggingFace token (required for gated Llama models)
export HF_ACCESS_TOKEN="${{HF_ACCESS_TOKEN:-}}"

echo "========================"
echo "running SFT training"
echo "parameters:"
echo "MODEL_NAME: $MODEL_NAME"
echo "SEQ_LEN: $SEQ_LEN"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GRAD_ACCUM: $GRAD_ACCUM"
echo "LR: $LR"
echo "NUM_STEPS: $NUM_STEPS"
echo "PRETRAINED_CHECKPOINT: $PRETRAINED_CHECKPOINT"
echo "BASE_OUTPUT_DIRECTORY: $BASE_OUTPUT_DIRECTORY"
echo "RUN_NAME: $RUN_NAME"
echo "HF_PATH: $HF_PATH"
echo "TOKENIZER_PATH: $TOKENIZER_PATH"
echo "start time: $(date)"
echo "========================"

python -u multihost_runner_orig.py \\
    --TPU_PREFIX=$TPU_PREFIX \\
    --INTERNAL_IP=true \\
    --COMMAND="
    export TPU_LOG_DIR=/home/terry/tpu_logs
    source ~/maxtext_env/bin/activate
    python3.10 -u -m MaxText.sft_trainer MaxText/configs/sft.yml \\
        run_name=${{RUN_NAME}} \\
        base_output_directory=${{BASE_OUTPUT_DIRECTORY}} \\
        model_name=${{MODEL_NAME}} \\
        load_parameters_path=${{PRETRAINED_CHECKPOINT}} \\
        tokenizer_path=${{TOKENIZER_PATH}} \\
        hf_access_token=${{HF_ACCESS_TOKEN}} \\
        max_target_length=${{SEQ_LEN}} \\
        per_device_batch_size=${{BATCH_SIZE}} \\
        gradient_accumulation_steps=${{GRAD_ACCUM}} \\
        steps=${{NUM_STEPS}} \\
        learning_rate=${{LR}} \\
        cosine_learning_rate_final_fraction=${{MIN_LR_RATIO}} \\
        warmup_steps_fraction=${{WARMUP_RATIO}} \\
        async_checkpointing=${{ASYNC_CHECKPOINTING}} \\
        checkpoint_period=1000 \\
        checkpoint_max_to_keep=1 \\
        use_sft=true \\
        sft_train_on_completion_only=true \\
        packing=true \\
        dataset_type=hf \\
        hf_path=${{HF_PATH}} \\
        train_split=${{TRAIN_SPLIT}} \\
        hf_eval_split=${{EVAL_SPLIT}} \\
        train_data_columns=['messages'] \\
        eval_data_columns=['messages'] \\
        eval_interval=-1 \\
        enable_data_shuffling=true \\
        data_shuffle_seed=43 \\
        gcs_metrics=True
    "
'''


def generate_vanilla_script(
    ckpt_name: str,
    ckpt_path: str,
    checkpoint_step: int,
    output_dir: str,
) -> dict:
    """Generate a single SFT training script for vanilla baseline."""
    # SFT script filename
    script_name = f"sft_vanilla_{ckpt_name}.sh"

    # Run name and ID for SFT
    run_name = f"sft_vanilla_{ckpt_name}"
    run_id = f"sft_vanilla_{ckpt_name.replace('-', '_')}"

    content = VANILLA_SCRIPT_TEMPLATE.format(
        ckpt_path=ckpt_path,
        checkpoint_step=checkpoint_step,
        run_name=run_name,
        run_id=run_id,
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


def alpha_to_str(alpha: float) -> str:
    """Convert alpha float to string for naming (e.g., 0.5 -> 'a05', 1.0 -> 'a1', 0.2 -> 'a02')."""
    val = int(alpha * 10)
    if val >= 10:
        return f"a{val // 10}"
    else:
        return f"a{val:02d}"


def get_teacher_naming(arch: str, tokens: str, seed: int) -> str:
    """Generate teacher naming like A3BT50BS42."""
    tokens_num = tokens.replace("B", "")
    arch_upper = arch.upper()
    return f"A{arch_upper}T{tokens_num}BS{seed}"


def generate_script(
    exp_name: str,
    teacher_arch: str,
    tokens: str,
    teacher_seed: int,
    alpha: float,
    student_seed: int,
    checkpoint_step: int,
    output_dir: str,
) -> dict:
    """Generate a single SFT training script. Returns dict with script info."""
    teacher_naming = get_teacher_naming(teacher_arch, tokens, teacher_seed)
    alpha_str = alpha_to_str(alpha)

    # Pretrain run name (source checkpoint)
    pretrain_run_name = f"{exp_name}_llama3.1-1b-{teacher_naming}-{alpha_str}-s{student_seed}"

    # SFT script filename
    script_name = f"sft_{exp_name}_llama3.1-1b-{teacher_naming}-{alpha_str}-s{student_seed}.sh"

    # Run name and ID for SFT
    run_name = f"sft_{exp_name}_llama3.1-1b-{teacher_naming}-{alpha_str}-s{student_seed}"
    run_id = f"sft_{exp_name}_llama3.1_1b_{teacher_naming}_{alpha_str}_s{student_seed}"

    # Depends on the pretrain task (matches exp1/exp2 run_all.yaml format)
    depends_on = f"{exp_name}_llama1b_finewebedu_distill_soft_{teacher_naming}_{alpha_str}_s{student_seed}"

    content = SCRIPT_TEMPLATE.format(
        exp_name=exp_name,
        pretrain_run_name=pretrain_run_name,
        checkpoint_step=checkpoint_step,
        run_name=run_name,
        run_id=run_id,
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
        "depends_on": depends_on,
    }


def generate_run_all_yaml(script_infos: list, output_dir: str) -> str:
    """Generate a run_all.yaml file that lists all generated tasks."""
    lines = ["tasks:"]
    for info in script_infos:
        lines.append(f"  - id: {info['run_id']}")
        lines.append(f"    run: bash train/sft/{info['script_name']}")
        if info.get('depends_on'):
            lines.append(f"    depends_on: {info['depends_on']}")

    content = '\n'.join(lines) + '\n'

    output_path = os.path.join(output_dir, "run_all.yaml")
    with open(output_path, "w") as f:
        f.write(content)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate SFT training scripts for distillation checkpoints")
    parser.add_argument(
        "--exp",
        nargs="+",
        choices=["exp1", "exp2", "vanilla"],
        default=["exp1", "exp2", "vanilla"],
        help="Which experiments to generate SFT scripts for (default: all)",
    )
    parser.add_argument(
        "--teacher-archs",
        nargs="+",
        default=None,
        help="Override teacher architectures (default: use experiment config)",
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        default=None,
        help="Override tokens (default: use experiment config)",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=None,
        help="Override KD alpha values (default: use experiment config)",
    )
    parser.add_argument(
        "--teacher-seed",
        type=int,
        default=DEFAULT_TEACHER_SEED,
        help=f"Teacher seed (default: {DEFAULT_TEACHER_SEED})",
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

    for exp_name in args.exp:
        if exp_name == "vanilla":
            # Generate vanilla baseline scripts
            for ckpt_info in VANILLA_CONFIG["checkpoints"]:
                script_name = f"sft_vanilla_{ckpt_info['name']}.sh"
                if args.dry_run:
                    print(f"Would generate: {script_name}")
                else:
                    info = generate_vanilla_script(
                        ckpt_name=ckpt_info["name"],
                        ckpt_path=ckpt_info["path"],
                        checkpoint_step=SFT_CHECKPOINT_STEP,
                        output_dir=args.output_dir,
                    )
                    generated.append(info)
                    print(f"Generated: {info['path']}")
            continue

        if exp_name == "exp1":
            config = EXP1_CONFIG
        else:
            config = EXP2_CONFIG

        teacher_archs = args.teacher_archs or config["teacher_archs"]
        tokens_list = args.tokens or config["tokens"]
        alphas = args.alphas or config["alphas"]

        for teacher_arch, tokens, alpha in product(teacher_archs, tokens_list, alphas):
            teacher_naming = get_teacher_naming(teacher_arch, tokens, args.teacher_seed)
            alpha_str = alpha_to_str(alpha)
            script_name = f"sft_{exp_name}_llama3.1-1b-{teacher_naming}-{alpha_str}-s{args.student_seed}.sh"

            if args.dry_run:
                print(f"Would generate: {script_name}")
            else:
                info = generate_script(
                    exp_name=exp_name,
                    teacher_arch=teacher_arch,
                    tokens=tokens,
                    teacher_seed=args.teacher_seed,
                    alpha=alpha,
                    student_seed=args.student_seed,
                    checkpoint_step=SFT_CHECKPOINT_STEP,
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
