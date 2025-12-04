#!/usr/bin/env python3
"""
Generate top-k KD training scripts for v6.

Usage:
    # Generate with 0.5B teacher, top-k values 1, 2, 10
    python generate_topk_scripts.py --teacher 05b --ks 1 2 10

    # Generate with 1B teacher (self-distill), top-k values 1, 2, 10, 100, 1000, 10000
    python generate_topk_scripts.py --teacher 1b --ks 1 2 10 100 1000 10000

    # Generate with 3B teacher
    python generate_topk_scripts.py --teacher 3b --ks 1 2 10

    # Generate with multiple alpha values
    python generate_topk_scripts.py --teacher 1b --ks 1 2 --alphas 0.5 1.0

    # Dry run (just print what would be generated)
    python generate_topk_scripts.py --teacher 1b --ks 1 2 --dry-run
"""

import argparse
import os
from pathlib import Path


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
export NUM_STEPS=25000
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
export KD_USE_HARD_LABELS={kd_use_hard_labels}
export KD_TEACHER_PARAMETERS_PATH="{teacher_ckpt_path}"
{teacher_model_line}export KD_TOP_K={top_k}
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/ckpts/distill_pretrain"
export DATA_FILES='/home/terry/gcs-bucket/datasets/fineweb-edu/*.array_record'

export RUN_NAME="${{MODEL_NAME}}-finewebedu-distill-top{top_k}-{teacher_tag}-{alpha_tag}-s43_v6"
export RUN_ID="llama1b_finewebedu_distill_top{top_k}_{teacher_tag}_{alpha_tag}_s43_v6"

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
echo "KD_USE_HARD_LABELS: $KD_USE_HARD_LABELS"
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
        data_shuffle_seed=43 \\
        init_weights_seed=43 \\
        wandb_resume=relog \\
        wandb_relog_source=auto  \\
        use_kd=${{USE_KD}} \\
        kd_alpha=${{KD_ALPHA}} \\
        kd_temperature=${{KD_TEMPERATURE}} \\
        kd_teacher_parameters_path=${{KD_TEACHER_PARAMETERS_PATH}} \\
        kd_use_hard_labels=${{KD_USE_HARD_LABELS}} \\
        kd_top_k=${{KD_TOP_K}}{teacher_model_arg}
    "
'''


# Teacher configurations
TEACHER_CONFIGS = {
    "05b": {
        "model_name": "llama3.1-05b",
        "ckpt_path": "/home/terry/gcs-bucket/ckpts/pretrain_param_only/llama3.1-05b-finewebedu-vanilla-s42/checkpoint_24999/0/items",
        "tag": "A05BT50BS42",  # A=Assistant(0.5B), T50=Teacher seed 50, BS42=Base seed 42
    },
    "1b": {
        "model_name": None,  # Self-distill, no separate teacher model name needed
        "ckpt_path": "/home/terry/gcs-bucket/ckpts/pretrain_param_only/llama3.1-1b_finewebedu_pretrain_shuffled_lr_3e-4_seed_42/checkpoint_24999/0/items",
        "tag": "A1BT50BS42",
    },
    "3b": {
        "model_name": "llama3.1-3b",
        "ckpt_path": "/home/terry/gcs-bucket/ckpts/pretrain_param_only/llama3.1-3b-finewebedu-vanilla-s42/checkpoint_24999/0/items",
        "tag": "A3BT50BS42",
    },
}


def get_alpha_tag(alpha: float) -> str:
    """Convert alpha value to tag string."""
    if alpha == 0.5:
        return "a05"
    elif alpha == 1.0:
        return "a1"
    else:
        # Handle other values like 0.25 -> a025
        return f"a{str(alpha).replace('.', '').replace('0', '', 1)}"


def generate_script(top_k: int, teacher: str, alpha: float, output_dir: Path) -> tuple[str, str]:
    """Generate a single script and return (filename, content)."""
    config = TEACHER_CONFIGS[teacher]

    alpha_tag = get_alpha_tag(alpha)
    teacher_tag = config["tag"]

    # Determine if hard labels should be used (only for top-1)
    use_hard_labels = "true" if top_k == 1 else "false"

    # Teacher model name line (only needed for non-self-distill)
    if config["model_name"]:
        teacher_model_line = f'export TEACHER_MODEL_NAME="{config["model_name"]}"\n'
        teacher_model_arg = " \\\n        kd_teacher_model_name=${TEACHER_MODEL_NAME}"
    else:
        teacher_model_line = ""
        teacher_model_arg = ""

    script_name = f"llama1b-finewebedu-distill-top{top_k}-{teacher_tag}-{alpha_tag}-s43.sh"

    content = SCRIPT_TEMPLATE.format(
        kd_alpha=alpha,
        kd_use_hard_labels=use_hard_labels,
        teacher_ckpt_path=config["ckpt_path"],
        teacher_model_line=teacher_model_line,
        top_k=top_k,
        teacher_tag=teacher_tag,
        alpha_tag=alpha_tag,
        script_name=script_name,
        teacher_model_arg=teacher_model_arg,
    )

    return script_name, content


def main():
    parser = argparse.ArgumentParser(
        description="Generate top-k KD training scripts for v6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--teacher",
        type=str,
        nargs="+",
        default=["05b", "1b", "3b"],
        choices=["05b", "1b", "3b"],
        help="Teacher model size(s): 05b, 1b (self-distill), or 3b (default: all three)"
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 2, 10, 100, 1000, 10000],
        help="List of top-k values (default: 1 2 10 100 1000 10000)"
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.5, 1.0],
        help="List of alpha values (default: 0.5 1.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same directory as this script)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent

    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []

    for teacher in args.teacher:
        for top_k in args.ks:
            for alpha in args.alphas:
                filename, content = generate_script(top_k, teacher, alpha, output_dir)
                filepath = output_dir / filename

                if args.dry_run:
                    print(f"Would generate: {filepath}")
                else:
                    with open(filepath, "w") as f:
                        f.write(content)
                    os.chmod(filepath, 0o755)
                    print(f"Generated: {filepath}")

                generated_files.append(filename)

    print(f"\nTotal: {len(generated_files)} scripts")
    if not args.dry_run:
        print(f"Scripts written to: {output_dir}")


if __name__ == "__main__":
    main()
