#!/usr/bin/env python3
"""
Generate SFT hyperparameter sweep scripts.

Generates shell scripts for different combinations of:
- Learning rates
- Warmup ratios
- Batch sizes
- Step counts

Usage:
    python generate_sweep.py
    python generate_sweep.py --lrs 1e-4 5e-5 --warmups 0.1 0.3 --batch-sizes 1 2
    python generate_sweep.py --dry-run
"""

import argparse
import os
from itertools import product


# =============================================================================
# DEFAULT SWEEP CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "learning_rates": ["1e-4", "2e-4", "5e-4", "1e-5", "2e-5", "5e-5", "1e-6", "2e-6", "5e-6"],
    "warmup_ratios": [0.1, 0.25],
    "batch_sizes": [1],
    "steps": [8000],
    "experiment_version": "t3",  # Increment when changing config
}

# Fixed parameters (not part of sweep)
FIXED_PARAMS = {
    "model_name": "llama3.1-1b",
    "seq_len": 4096,
    "grad_accum": 1,
    "min_lr_ratio": 1.0,  # Constant LR (no decay)
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
export MIN_LR_RATIO={min_lr_ratio}  # Constant LR (no decay)
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


def format_lr(lr: str) -> str:
    """Format learning rate for filename (e.g., '1e-4' -> '1e-4', '5e-5' -> '5e-5')."""
    return lr.replace(".", "")


def format_warmup(warmup: float) -> str:
    """Format warmup ratio for filename suffix (e.g., 0.1 -> '', 0.3 -> '_w03')."""
    if warmup == 0.1:
        return ""  # Default, no suffix
    # Convert 0.3 -> "w03", 0.5 -> "w05", etc.
    return f"_w{int(warmup * 10):02d}"


def generate_run_name(
    exp_version: str,
    batch_size: int,
    steps: int,
    lr: str,
    warmup: float,
) -> str:
    """Generate run name like 'sft_t3_vanilla1b_b1_s8000_1e-4_w03'."""
    warmup_suffix = format_warmup(warmup)
    return f"sft_{exp_version}_vanilla1b_b{batch_size}_s{steps}_{lr}{warmup_suffix}"


def generate_script(
    exp_version: str,
    batch_size: int,
    steps: int,
    lr: str,
    warmup: float,
    output_dir: str,
) -> dict:
    """Generate a single SFT training script. Returns dict with script info."""
    run_name = generate_run_name(exp_version, batch_size, steps, lr, warmup)
    run_id = run_name  # Same as run_name
    script_name = f"{run_name}.sh"

    content = SCRIPT_TEMPLATE.format(
        run_name=run_name,
        run_id=run_id,
        batch_size=batch_size,
        steps=steps,
        lr=lr,
        warmup_ratio=warmup,
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
        "warmup": warmup,
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
        "--warmups",
        nargs="+",
        type=float,
        default=DEFAULT_CONFIG["warmup_ratios"],
        help=f"Warmup ratios (default: {DEFAULT_CONFIG['warmup_ratios']})",
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
        product(args.batch_sizes, args.steps, args.lrs, args.warmups))

    print(f"Generating {len(combinations)} scripts...")
    print(f"  Experiment version: {args.exp_version}")
    print(f"  Learning rates: {args.lrs}")
    print(f"  Warmup ratios: {args.warmups}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Steps: {args.steps}")
    print()

    for batch_size, steps, lr, warmup in combinations:
        run_name = generate_run_name(
            args.exp_version, batch_size, steps, lr, warmup)
        script_name = f"{run_name}.sh"

        if args.dry_run:
            print(f"Would generate: {script_name}")
        else:
            info = generate_script(
                exp_version=args.exp_version,
                batch_size=batch_size,
                steps=steps,
                lr=lr,
                warmup=warmup,
                output_dir=args.output_dir,
            )
            generated.append(info)
            print(f"Generated: {info['script_name']}")

    if not args.dry_run and generated:
        print(f"\nTotal scripts generated: {len(generated)}")

        # Generate run_all.yaml
        run_all_path = generate_run_all_yaml(generated, args.output_dir)
        print(f"Generated: {run_all_path}")

        # Print summary table
        print("\n" + "=" * 70)
        print("SUMMARY TABLE")
        print("=" * 70)
        print(f"{'Script':<50} {'LR':<8} {'Warmup':<8}")
        print("-" * 70)
        for info in generated:
            warmup_str = f"{info['warmup']:.1f}"
            print(f"{info['script_name']:<50} {info['lr']:<8} {warmup_str:<8}")


if __name__ == "__main__":
    main()
