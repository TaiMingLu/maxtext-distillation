#!/usr/bin/env python3
"""
Generate SFT training scripts for distillation experiments, teachers, and baseline.

Uses best hyperparameters from sweep:
- LR: 5e-5 with cosine decay (1% warmup, decay to 0.1)
- Batch size: 2, Steps: 4000
- Plain completion format (no special tokens)

Usage:
    python generate_scripts.py
    python generate_scripts.py --exp exp1 exp2 teacher baseline
    python generate_scripts.py --exp exp1 --teacher-archs 1b 3b --tokens 50B --alphas 0.5 1.0
"""

import argparse
import os
from itertools import product

# =============================================================================
# BEST HYPERPARAMETERS (from sweep)
# =============================================================================
SFT_HYPERPARAMS = {
    "lr": "1e-5",
    "min_lr_ratio": 0.1,      # Cosine decay to 1/10
    "warmup_ratio": 0.01,     # 1% warmup
    "batch_size": 4,
    "steps": 4000,
}

# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

# exp1: fixed tokens (50B), varying alpha
EXP1_CONFIG = {
    "teacher_archs": ["05b", "1b", "3b", "8b"],
    "tokens": ["50B"],
    "alphas": [0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
}

# exp2: varying tokens, fewer alphas
EXP2_CONFIG = {
    "teacher_archs": ["05b", "1b", "3b", "8b"],
    "tokens": ["5B", "10B", "30B", "50B", "80B", "100B", "300B"],
    "alphas": [0.5, 1.0],
}

# Teacher models - all size/token combinations
TEACHER_CONFIG = {
    "sizes": ["05b", "1b", "3b", "8b"],
    "tokens": ["5B", "10B", "30B", "50B", "80B", "100B", "300B"],
    "seed": 42,
}

# Baseline - single 1B model trained from scratch
BASELINE_CONFIG = {
    "run_name": "llama3.1-1b-finewebedu-vanilla-s43-50b",
    "model_name": "llama3.1-1b",
    "checkpoint_step": 24999,
    "ckpt_dir": "vanilla",
}

# Token to checkpoint step mapping: step = (tokens_in_billions * 500) - 1
TOKEN_TO_CHECKPOINT_STEP = {
    "5b": 2499,
    "10b": 4999,
    "30b": 14999,
    "50b": 24999,
    "80b": 39999,
    "100b": 49999,
    "300b": 149999,
    "1000b": 499999,
}

DEFAULT_TEACHER_SEED = 42
DEFAULT_STUDENT_SEED = 43


# =============================================================================
# SCRIPT TEMPLATES
# =============================================================================

SCRIPT_TEMPLATE = '''#!/bin/bash
#
# SFT (Supervised Fine-Tuning) script for {model_name}
# Loads pretrained checkpoint and fine-tunes on Dolci dataset
#
# Hyperparameters (from sweep):
#   - LR: {lr} with cosine decay to {min_lr_ratio}
#   - Warmup: {warmup_ratio}
#   - Batch size: {batch_size}, Steps: {steps}
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
export MODEL_NAME='{model_name}'
export SEQ_LEN=4096
export BATCH_SIZE={batch_size}
export GRAD_ACCUM=1

# SFT training hyperparameters (best from sweep)
export NUM_STEPS={steps}
export LR={lr}
export MIN_LR_RATIO={min_lr_ratio}  # Cosine decay to 1/10
export WARMUP_RATIO={warmup_ratio}  # 1% warmup
export ASYNC_CHECKPOINTING=false

# Pretrained checkpoint to load
export PRETRAINED_CHECKPOINT="gs://${{BUCKET_NAME}}/ckpts_copy/{checkpoint_path}"
# Output directory for SFT checkpoints
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/ckpts/sft"

# Run naming
export RUN_NAME="{run_name}"
export RUN_ID="{run_id}"

# HuggingFace dataset configuration
export HF_PATH='/home/terry/gcs-data/datasets/Dolci-Instruct-SFT-7B'
export TRAIN_SPLIT='train'
export EVAL_SPLIT='train'

# Tokenizer
export TOKENIZER_PATH='/home/terry/gcs-data/HF_HOME/Llama-3.2-1B-Instruct'
export HF_ACCESS_TOKEN="${{HF_ACCESS_TOKEN:-}}"

echo "========================"
echo "running SFT training"
echo "parameters:"
echo "MODEL_NAME: $MODEL_NAME"
echo "SEQ_LEN: $SEQ_LEN"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "LR: $LR"
echo "MIN_LR_RATIO: $MIN_LR_RATIO"
echo "WARMUP_RATIO: $WARMUP_RATIO"
echo "NUM_STEPS: $NUM_STEPS"
echo "PRETRAINED_CHECKPOINT: $PRETRAINED_CHECKPOINT"
echo "BASE_OUTPUT_DIRECTORY: $BASE_OUTPUT_DIRECTORY"
echo "RUN_NAME: $RUN_NAME"
echo "HF_PATH: $HF_PATH"
echo "TOKENIZER_PATH: $TOKENIZER_PATH"
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
    export WANDB_PROJECT=maxtext_sft1228
    export WANDB_NAME=${{RUN_NAME}}
    python3.10 -u -m MaxText.sft_trainer MaxText/configs/sft.yml \\
        run_name=${{RUN_NAME}} \\
        base_output_directory=${{BASE_OUTPUT_DIRECTORY}} \\
        model_name=${{MODEL_NAME}} \\
        load_parameters_path=${{PRETRAINED_CHECKPOINT}} \\
        tokenizer_path=${{TOKENIZER_PATH}} \\
        hf_access_token=${{HF_ACCESS_TOKEN}} \\
        max_target_length=${{SEQ_LEN}} \\
        per_device_batch_size=${{BATCH_SIZE}} \\
        gradient_accumulation_steps=1 \\
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
        hf_streaming=False \\
        hf_num_proc=64 \\
        hf_path=${{HF_PATH}} \\
        train_split=${{TRAIN_SPLIT}} \\
        hf_eval_split=${{EVAL_SPLIT}} \\
        train_data_columns=['messages'] \\
        eval_data_columns=['messages'] \\
        eval_interval=-1 \\
        enable_data_shuffling=true \\
        data_shuffle_seed=43 \\
        gcs_metrics=True \\
        use_wandb=True \\
        wandb_project=maxtext_sft1228 \\
        wandb_run_name=${{RUN_NAME}} \\
        wandb_run_id=${{RUN_ID}}
    "
'''


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_checkpoint_step_for_tokens(tokens: str) -> int:
    """Get checkpoint step for a given token count."""
    tokens_lower = tokens.lower()
    if tokens_lower in TOKEN_TO_CHECKPOINT_STEP:
        return TOKEN_TO_CHECKPOINT_STEP[tokens_lower]
    # Fallback: calculate from tokens (step = tokens_in_billions * 500 - 1)
    tokens_num = int(tokens_lower.replace("b", ""))
    return tokens_num * 500 - 1


def alpha_to_str(alpha: float) -> str:
    """Convert alpha float to string for naming (e.g., 0.5 -> 'a05', 1.0 -> 'a1', 0.2 -> 'a02')."""
    val = int(alpha * 10)
    if val >= 10:
        return f"a{val // 10}"
    else:
        return f"a{val:02d}"


def get_teacher_naming(arch: str, tokens: str, seed: int) -> str:
    """Generate teacher naming like A3BT50BS42."""
    tokens_num = tokens.replace("B", "").replace("b", "")
    arch_upper = arch.upper()
    return f"A{arch_upper}T{tokens_num}BS{seed}"


def size_to_model_name(size: str) -> str:
    """Convert size string to model name (e.g., '8b' -> 'llama3.1-8b')."""
    # Valid: llama3.1-05b, llama3.1-1b, llama3.1-3b, llama3.1-8b
    return f"llama3.1-{size}"


# =============================================================================
# SCRIPT GENERATORS
# =============================================================================

def generate_distill_script(
    exp_name: str,
    teacher_arch: str,
    tokens: str,
    teacher_seed: int,
    alpha: float,
    student_seed: int,
    output_dir: str,
) -> dict:
    """Generate a single SFT training script for distillation checkpoint."""
    teacher_naming = get_teacher_naming(teacher_arch, tokens, teacher_seed)
    alpha_str = alpha_to_str(alpha)
    # Student models are ALWAYS trained for 50B tokens, regardless of teacher tokens
    checkpoint_step = 24999

    # Pretrain run name (source checkpoint)
    pretrain_run_name = f"{exp_name}_llama3.1-1b-{teacher_naming}-{alpha_str}-s{student_seed}"

    # Full checkpoint path: exp1 or exp2 directory
    checkpoint_path = f"{exp_name}/{pretrain_run_name}/checkpoints/{checkpoint_step}/items"

    # SFT script filename and run name
    script_name = f"sft_{exp_name}_llama3.1-1b-{teacher_naming}-{alpha_str}-s{student_seed}.sh"
    run_name = f"sft_{exp_name}_llama3.1-1b-{teacher_naming}-{alpha_str}-s{student_seed}"
    run_id = f"sft_{exp_name}_llama3.1_1b_{teacher_naming}_{alpha_str}_s{student_seed}"

    # Depends on the pretrain task
    depends_on = f"{exp_name}_llama1b_finewebedu_distill_soft_{teacher_naming}_{alpha_str}_s{student_seed}"

    content = SCRIPT_TEMPLATE.format(
        model_name="llama3.1-1b",
        checkpoint_path=checkpoint_path,
        run_name=run_name,
        run_id=run_id,
        **SFT_HYPERPARAMS,
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
        "model_name": "llama3.1-1b",
        "depends_on": depends_on,
        "steps": SFT_HYPERPARAMS["steps"],
    }


def generate_teacher_script(
    size: str,
    tokens: str,
    seed: int,
    output_dir: str,
) -> dict:
    """Generate SFT training script for teacher model."""
    # Teacher run name format: llama{size}b-vanilla-{tokens}B-s{seed}
    # e.g., llama05b-vanilla-100B-s42
    tokens_upper = tokens.upper()  # Ensure uppercase B
    pretrain_run_name = f"llama{size}-vanilla-{tokens_upper}-s{seed}"
    checkpoint_step = get_checkpoint_step_for_tokens(tokens)
    model_name = size_to_model_name(size)

    script_name = f"sft_teacher_{pretrain_run_name}.sh"
    run_name = f"sft_{pretrain_run_name}"
    run_id = f"sft_teacher_{pretrain_run_name.replace('-', '_')}"

    # Full checkpoint path: pretrain_param_only_v6 with checkpoint_{step}/0/items format
    checkpoint_path = f"pretrain_param_only_v6/{pretrain_run_name}/checkpoint_{checkpoint_step}/0/items"

    # Depends on teacher training
    depends_on = f"llama{size}_finewebedu_teacher_s{seed}_{tokens}"

    content = SCRIPT_TEMPLATE.format(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        run_name=run_name,
        run_id=run_id,
        **SFT_HYPERPARAMS,
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
        "model_name": model_name,
        "depends_on": depends_on,
        "steps": SFT_HYPERPARAMS["steps"],
    }


def generate_baseline_script(output_dir: str) -> dict:
    """Generate SFT training script for baseline model."""
    pretrain_run_name = BASELINE_CONFIG["run_name"]
    checkpoint_step = BASELINE_CONFIG["checkpoint_step"]
    model_name = BASELINE_CONFIG["model_name"]
    ckpt_dir = BASELINE_CONFIG["ckpt_dir"]

    # Full checkpoint path for baseline
    checkpoint_path = f"{ckpt_dir}/{pretrain_run_name}/checkpoints/{checkpoint_step}/items"

    script_name = f"sft_baseline_{pretrain_run_name}.sh"
    run_name = f"sft_{pretrain_run_name}"
    run_id = f"sft_baseline_{pretrain_run_name.replace('-', '_').replace('.', '_')}"

    content = SCRIPT_TEMPLATE.format(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        run_name=run_name,
        run_id=run_id,
        **SFT_HYPERPARAMS,
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
        "model_name": model_name,
        "depends_on": None,  # Baseline has no dependency
        "steps": SFT_HYPERPARAMS["steps"],
    }


# =============================================================================
# YAML GENERATORS
# =============================================================================

def generate_run_all_yaml(script_infos: list, output_dir: str) -> str:
    """Generate a run_all.yaml file that lists all generated tasks (no depends_on)."""
    lines = ["tasks:"]
    for info in script_infos:
        lines.append(f"  - id: {info['run_id']}")
        lines.append(f"    run: bash train/sft/{info['script_name']}")
        lines.append(f"    hide: true")

    content = '\n'.join(lines) + '\n'

    output_path = os.path.join(output_dir, "run_all.yaml")
    with open(output_path, "w") as f:
        f.write(content)

    return output_path


def generate_eval_all_yaml(script_infos: list, output_dir: str) -> str:
    """Generate an eval_all.yaml file with evaluation tasks that depend on SFT training."""
    lines = ["tasks:"]

    for info in script_infos:
        run_name = info["run_name"]
        sft_task_id = info["run_id"]
        model_name = info.get("model_name", "llama3.1-1b")
        # checkpoint step is steps - 1 (e.g., 4000 steps -> checkpoint 3999)
        ckpt_step = info["steps"] - 1

        lines.append(f"  - id: eval_{run_name.replace('-', '_')}")
        lines.append(f"    run: bash train/eval-base/eval_base_acc.sh {run_name} {model_name} {ckpt_step} sft --resume")
        lines.append(f"    depends_on: {sft_task_id}")
        lines.append(f"    hide: true")

    content = '\n'.join(lines) + '\n'

    output_path = os.path.join(output_dir, "eval_all.yaml")
    with open(output_path, "w") as f:
        f.write(content)

    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate SFT training scripts")
    parser.add_argument(
        "--exp",
        nargs="+",
        choices=["exp1", "exp2", "teacher", "baseline"],
        default=["exp1", "exp2", "teacher", "baseline"],
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

    print("=" * 60)
    print("SFT HYPERPARAMETERS (from sweep)")
    print("=" * 60)
    print(f"  LR: {SFT_HYPERPARAMS['lr']}")
    print(f"  Min LR ratio: {SFT_HYPERPARAMS['min_lr_ratio']} (cosine decay)")
    print(f"  Warmup: {SFT_HYPERPARAMS['warmup_ratio']}")
    print(f"  Batch size: {SFT_HYPERPARAMS['batch_size']}")
    print(f"  Steps: {SFT_HYPERPARAMS['steps']}")
    print("=" * 60)
    print()

    for exp_name in args.exp:
        # =====================================================================
        # BASELINE
        # =====================================================================
        if exp_name == "baseline":
            script_name = f"sft_baseline_{BASELINE_CONFIG['run_name']}.sh"
            if args.dry_run:
                print(f"Would generate: {script_name}")
            else:
                info = generate_baseline_script(output_dir=args.output_dir)
                generated.append(info)
                print(f"Generated: {info['script_name']}")
            continue

        # =====================================================================
        # TEACHER
        # =====================================================================
        if exp_name == "teacher":
            sizes = TEACHER_CONFIG["sizes"]
            tokens_list = TEACHER_CONFIG["tokens"]
            seed = TEACHER_CONFIG["seed"]

            for size, tokens in product(sizes, tokens_list):
                script_name = f"sft_teacher_llama{size}-finewebedu-teacher-s{seed}-{tokens}.sh"
                if args.dry_run:
                    print(f"Would generate: {script_name}")
                else:
                    info = generate_teacher_script(
                        size=size,
                        tokens=tokens,
                        seed=seed,
                        output_dir=args.output_dir,
                    )
                    generated.append(info)
                    print(f"Generated: {info['script_name']}")
            continue

        # =====================================================================
        # EXP1 / EXP2 (distillation)
        # =====================================================================
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
                info = generate_distill_script(
                    exp_name=exp_name,
                    teacher_arch=teacher_arch,
                    tokens=tokens,
                    teacher_seed=args.teacher_seed,
                    alpha=alpha,
                    student_seed=args.student_seed,
                    output_dir=args.output_dir,
                )
                generated.append(info)
                print(f"Generated: {info['script_name']}")

    if not args.dry_run and generated:
        print(f"\nTotal scripts generated: {len(generated)}")

        # Count by type
        baseline_count = sum(1 for i in generated if "baseline" in i["run_id"])
        teacher_count = sum(1 for i in generated if "teacher" in i["run_id"])
        distill_count = len(generated) - baseline_count - teacher_count

        print(f"  Baseline: {baseline_count}")
        print(f"  Teacher: {teacher_count}")
        print(f"  Distillation (exp1/exp2): {distill_count}")

        # Generate run_all.yaml
        run_all_path = generate_run_all_yaml(generated, args.output_dir)
        print(f"\nGenerated: {run_all_path}")

        # Generate eval_all.yaml
        eval_all_path = generate_eval_all_yaml(generated, args.output_dir)
        print(f"Generated: {eval_all_path}")


if __name__ == "__main__":
    main()
