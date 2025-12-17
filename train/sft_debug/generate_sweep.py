#!/usr/bin/env python3
"""
Generate SFT hyperparameter sweep scripts.

Generates shell scripts for different combinations of:
- Learning rates
- LR schedules (constant vs cosine decay)
- Batch sizes
- Step counts

Usage:
    python generate_sweep.py
    python generate_sweep.py --lrs 1e-4 5e-5 --batch-sizes 1 2
    python generate_sweep.py --dry-run
"""

import argparse
import os
from itertools import product


# =============================================================================
# DEFAULT SWEEP CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "learning_rates": ["1e-4", "5e-4", "1e-5", "5e-5", "1e-6", "5e-6"],
    "batch_sizes": [2],
    "steps": [4000],
    "experiment_version": "t4",  # Increment when changing config
}

# LR Schedule Recipes
# Each recipe defines: (suffix, warmup_ratio, min_lr_ratio)
LR_RECIPES = {
    "constant": ("", 0.1, 1.0),           # No suffix: 10% warmup, constant LR
    "cosine": ("_cos", 0.01, 0.1),        # _cos suffix: 1% warmup, decay to 1/10
}

# Fixed parameters (not part of sweep)
FIXED_PARAMS = {
    "model_name": "llama3.1-1b",
    "seq_len": 4096,
    "grad_accum": 1,
    "async_checkpointing": "false",
    "checkpoint_period": 1000,
    "checkpoint_max_to_keep": 1,
    "pretrain_checkpoint_path": "ckpts/pretrain/llama3.1-1b-finewebedu-vanilla-s42-50b/checkpoints/24999/items",
    "hf_path": "/home/terry/gcs-data/datasets/Dolci-Instruct-SFT-7B",
    "tokenizer_path": "/home/terry/gcs-data/HF_HOME/Llama-3.2-1B-Instruct",
    "wandb_project": "maxtext_sft_experiment",
}


# =============================================================================
# SCRIPT TEMPLATE
# =============================================================================

SCRIPT_TEMPLATE = '''#!/bin/bash
#
# SFT (Supervised Fine-Tuning) script for Llama 1B
# Loads vanilla pretrained checkpoint (no distillation) and fine-tunes on Dolci dataset
#
# Recipe: {recipe_name}
#   - Warmup: {warmup_ratio}
#   - LR decay: {min_lr_ratio} (1.0 = constant, <1.0 = cosine decay)
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
export SEQ_LEN={seq_len}
export BATCH_SIZE={batch_size}  # per-device batch size
export GRAD_ACCUM={grad_accum}

# SFT training hyperparameters
export NUM_STEPS={steps}
export LR={lr}
export MIN_LR_RATIO={min_lr_ratio}  # 1.0 = constant, <1.0 = cosine decay
export WARMUP_RATIO={warmup_ratio}
export ASYNC_CHECKPOINTING={async_checkpointing}

# Pretrained checkpoint to load (vanilla, no distillation)
export PRETRAINED_CHECKPOINT="gs://${{BUCKET_NAME}}/{pretrain_checkpoint_path}"
# Output directory for SFT checkpoints
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/ckpts/sft"

# Run naming
export RUN_NAME="{run_name}"
export RUN_ID="{run_id}"

# HuggingFace dataset configuration
export HF_PATH='{hf_path}'
export TRAIN_SPLIT='train'
export EVAL_SPLIT='train'  # No separate eval split, use subset of train

# Tokenizer - MUST use Instruct version for chat template
export TOKENIZER_PATH='{tokenizer_path}'
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

python -u multihost_runner_orig.py \\
    --TPU_PREFIX=$TPU_PREFIX \\
    --INTERNAL_IP=true \\
    --COMMAND="
    export TPU_LOG_DIR=/home/terry/tpu_logs
    source ~/maxtext_env/bin/activate
    export WANDB_API_KEY='01126ae90da25bae0d86704140ac978cb9fd9c73'
    export WANDB_PROJECT={wandb_project}
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
        gradient_accumulation_steps=${{GRAD_ACCUM}} \\
        steps=${{NUM_STEPS}} \\
        learning_rate=${{LR}} \\
        cosine_learning_rate_final_fraction=${{MIN_LR_RATIO}} \\
        warmup_steps_fraction=${{WARMUP_RATIO}} \\
        async_checkpointing=${{ASYNC_CHECKPOINTING}} \\
        checkpoint_period={checkpoint_period} \\
        checkpoint_max_to_keep={checkpoint_max_to_keep} \\
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
        wandb_project={wandb_project} \\
        wandb_run_name=${{RUN_NAME}} \\
        wandb_run_id=${{RUN_ID}}
    "
'''


def generate_run_name(
    exp_version: str,
    batch_size: int,
    steps: int,
    lr: str,
    recipe_suffix: str,
) -> str:
    """Generate run name like 'sft_t4_vanilla1b_b1_s8000_1e-4' or 'sft_t4_vanilla1b_b1_s8000_1e-4_cos'."""
    return f"sft_{exp_version}_vanilla1b_b{batch_size}_s{steps}_{lr}{recipe_suffix}"


def generate_script(
    exp_version: str,
    batch_size: int,
    steps: int,
    lr: str,
    recipe_name: str,
    recipe_suffix: str,
    warmup_ratio: float,
    min_lr_ratio: float,
    output_dir: str,
) -> dict:
    """Generate a single SFT training script. Returns dict with script info."""
    run_name = generate_run_name(exp_version, batch_size, steps, lr, recipe_suffix)
    run_id = run_name  # Same as run_name
    script_name = f"{run_name}.sh"

    content = SCRIPT_TEMPLATE.format(
        run_name=run_name,
        run_id=run_id,
        batch_size=batch_size,
        steps=steps,
        lr=lr,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
        recipe_name=recipe_name,
        **FIXED_PARAMS,
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
        "lr": lr,
        "recipe": recipe_name,
        "warmup": warmup_ratio,
        "min_lr": min_lr_ratio,
        "batch_size": batch_size,
        "steps": steps,
    }


def generate_run_all_yaml(script_infos: list, output_dir: str) -> str:
    """Generate a run_all.yaml file that lists all generated tasks."""
    lines = ["tasks:"]
    for info in script_infos:
        lines.append(f"  - id: {info['run_id']}")
        lines.append(f"    run: bash train/sft_debug/{info['script_name']}")

    content = "\n".join(lines) + "\n"

    output_path = os.path.join(output_dir, "run_all.yaml")
    with open(output_path, "w") as f:
        f.write(content)

    return output_path


def generate_eval_all_yaml(script_infos: list, output_dir: str, exp_version: str) -> str:
    """Generate an eval_all.yaml file with evaluation tasks for all SFT runs + vanilla baseline."""
    lines = ["tasks:"]

    # Add vanilla baseline evaluation first
    lines.append(f"  - id: eval_{exp_version}_vanilla1b_pretrain")
    lines.append("    run: bash train/eval-base/eval_base_acc.sh llama3.1-1b-finewebedu-vanilla-s42_v6 llama3.1-1b 24999 pretrain --resume")

    # Add SFT evaluations
    for info in script_infos:
        run_name = info["run_name"]
        # checkpoint step is steps - 1 (e.g., 8000 steps -> checkpoint 7999)
        ckpt_step = info["steps"] - 1
        lines.append(f"  - id: eval_{run_name}")
        lines.append(f"    run: bash train/eval-base/eval_base_acc.sh {run_name} llama3.1-1b {ckpt_step} sft --resume")

    content = "\n".join(lines) + "\n"

    output_path = os.path.join(output_dir, "eval_all.yaml")
    with open(output_path, "w") as f:
        f.write(content)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT hyperparameter sweep scripts")
    parser.add_argument(
        "--exp-version",
        default=DEFAULT_CONFIG["experiment_version"],
        help=f"Experiment version prefix (default: {DEFAULT_CONFIG['experiment_version']})",
    )
    parser.add_argument(
        "--lrs",
        nargs="+",
        default=DEFAULT_CONFIG["learning_rates"],
        help=f"Learning rates (default: {DEFAULT_CONFIG['learning_rates']})",
    )
    parser.add_argument(
        "--recipes",
        nargs="+",
        choices=list(LR_RECIPES.keys()),
        default=list(LR_RECIPES.keys()),
        help=f"LR recipes to use (default: {list(LR_RECIPES.keys())})",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_CONFIG["batch_sizes"],
        help=f"Batch sizes (default: {DEFAULT_CONFIG['batch_sizes']})",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=DEFAULT_CONFIG["steps"],
        help=f"Training steps (default: {DEFAULT_CONFIG['steps']})",
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

    # Generate all combinations
    combinations = list(
        product(args.batch_sizes, args.steps, args.lrs, args.recipes))

    print(f"Generating {len(combinations)} scripts...")
    print(f"  Experiment version: {args.exp_version}")
    print(f"  Learning rates: {args.lrs}")
    print(f"  Recipes: {args.recipes}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Steps: {args.steps}")
    print()
    print("LR Recipes:")
    for name in args.recipes:
        suffix, warmup, min_lr = LR_RECIPES[name]
        suffix_str = suffix if suffix else "(none)"
        print(f"  {name}: suffix={suffix_str}, warmup={warmup}, min_lr_ratio={min_lr}")
    print()

    for batch_size, steps, lr, recipe_name in combinations:
        recipe_suffix, warmup_ratio, min_lr_ratio = LR_RECIPES[recipe_name]
        run_name = generate_run_name(
            args.exp_version, batch_size, steps, lr, recipe_suffix)
        script_name = f"{run_name}.sh"

        if args.dry_run:
            print(f"Would generate: {script_name}")
        else:
            info = generate_script(
                exp_version=args.exp_version,
                batch_size=batch_size,
                steps=steps,
                lr=lr,
                recipe_name=recipe_name,
                recipe_suffix=recipe_suffix,
                warmup_ratio=warmup_ratio,
                min_lr_ratio=min_lr_ratio,
                output_dir=args.output_dir,
            )
            generated.append(info)
            print(f"Generated: {info['script_name']}")

    if not args.dry_run and generated:
        print(f"\nTotal scripts generated: {len(generated)}")

        # Generate run_all.yaml
        run_all_path = generate_run_all_yaml(generated, args.output_dir)
        print(f"Generated: {run_all_path}")

        # Generate eval_all.yaml
        eval_all_path = generate_eval_all_yaml(generated, args.output_dir, args.exp_version)
        print(f"Generated: {eval_all_path}")

        # Print summary table
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Script':<55} {'LR':<8} {'Recipe':<10} {'Warmup':<8} {'MinLR':<8}")
        print("-" * 80)
        for info in generated:
            print(f"{info['script_name']:<55} {info['lr']:<8} {info['recipe']:<10} {info['warmup']:<8} {info['min_lr']:<8}")


if __name__ == "__main__":
    main()
