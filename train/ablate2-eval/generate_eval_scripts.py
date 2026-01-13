#!/usr/bin/env python3
"""
Generate evaluation task YAML for ablate2 distillation experiments (Qwen models).

- Pretrain models -> PPL evaluation (eval_pretrain_ppl.sh)
- Pretrain models -> ACC evaluation without chat template (eval_base_acc.sh)

Usage:
    python generate_eval_scripts.py
    python generate_eval_scripts.py --exp baseline teacher kd  # Default
    python generate_eval_scripts.py --exp kd --eval-type ppl   # KD PPL only
"""

import argparse
import os
from itertools import product

# Ablate2 KD configuration (qwen03b student only)
KD_CONFIG = {
    "student_arch": "03b",  # Only qwen03b student
    "teacher_archs": ["015b", "03b", "06b", "1b"],
    "tokens": "50B",
    "alphas": [0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
    "teacher_seed": 42,
    "student_seed": 43,
}

# Checkpoint steps
PRETRAIN_CHECKPOINT_STEP = 24999  # For 50B tokens

# Teacher baselines - trained vanilla models used as teachers
TEACHER_CONFIG = {
    "sizes": ["015b", "03b", "06b", "1b"],
    "tokens": "50B",
    "seed": 42,
    "ckpt_dir": "ablate2_param_only",
}

# Baseline - models trained from scratch (no distillation)
# Only qwen03b with seed 43
BASELINE_CONFIG = {
    "sizes": ["03b"],
    "tokens": "50B",
    "seeds": [43],
    "ckpt_dir": "ablate2",
}

# Default CLI parameters
DEFAULT_EXPERIMENTS = ["baseline", "teacher", "kd"]
DEFAULT_EVAL_TYPE = "base"  # Options: "base" (ppl+base_acc), "ppl", "base_acc"
DEFAULT_OUTPUT_DIR = "."
DEFAULT_ENABLE_DEPENDS_ON = True  # Enable depends_on for task dependencies
DEFAULT_HIDE = False


def alpha_to_str(alpha: float) -> str:
    """Convert alpha float to string for naming (e.g., 0.5 -> 'a05', 1.0 -> 'a1')."""
    val = int(alpha * 10)
    if val >= 10:
        return f"a{val // 10}"
    else:
        return f"a{val:02d}"


def get_teacher_naming(arch: str, tokens: str, seed: int) -> str:
    """Generate teacher naming like A015BT50BS42."""
    tokens_num = tokens.replace("B", "")
    arch_upper = arch.upper()
    return f"A{arch_upper}T{tokens_num}BS{seed}"


def size_to_model_name(size: str) -> str:
    """Convert size string to model name for maxtext (e.g., '03b' -> 'qwen3-03b')."""
    return f"qwen3-{size}"


def get_kd_run_name(student_arch: str, teacher_arch: str, tokens: str,
                    teacher_seed: int, alpha: float, student_seed: int) -> str:
    """Get KD run name like ablate2_qwen03b_A015BT50BS42_a02_s43."""
    teacher_naming = get_teacher_naming(teacher_arch, tokens, teacher_seed)
    alpha_str = alpha_to_str(alpha)
    return f"ablate2_qwen{student_arch}_{teacher_naming}_{alpha_str}_s{student_seed}"


def get_teacher_run_name(size: str, tokens: str, seed: int) -> str:
    """Get teacher run name like qwen015b-vanilla-50B-s42."""
    return f"qwen{size}-vanilla-{tokens}-s{seed}"


def get_baseline_run_name(size: str, tokens: str, seed: int) -> str:
    """Get baseline run name like qwen03b_finewebedu_vanilla_s42_50b."""
    return f"qwen{size}_finewebedu_vanilla_s{seed}_{tokens.lower()}"


def get_kd_training_task_id(student_arch: str, teacher_arch: str, tokens: str,
                             teacher_seed: int, alpha: float, student_seed: int) -> str:
    """Get the training task ID for KD (same as run_name)."""
    return get_kd_run_name(student_arch, teacher_arch, tokens, teacher_seed, alpha, student_seed)


def get_baseline_training_task_id(size: str, tokens: str, seed: int) -> str:
    """Get the training task ID for baseline like ablate2_qwen03b_finewebedu_vanilla_s42_50b."""
    return f"ablate2_qwen{size}_finewebedu_vanilla_s{seed}_{tokens.lower()}"


def get_teacher_training_task_id(size: str, tokens: str, seed: int) -> str:
    """Get the training task ID for teacher (convert task)."""
    return f"convert_ablate2_qwen{size}_finewebedu_vanilla_s{seed}_{tokens.lower()}"


def generate_kd_tasks() -> list:
    """Generate eval tasks for KD models (qwen03b student with all teachers)."""
    tasks = []

    student_arch = KD_CONFIG["student_arch"]
    teacher_archs = KD_CONFIG["teacher_archs"]
    tokens = KD_CONFIG["tokens"]
    alphas = KD_CONFIG["alphas"]
    teacher_seed = KD_CONFIG["teacher_seed"]
    student_seed = KD_CONFIG["student_seed"]

    for teacher_arch, alpha in product(teacher_archs, alphas):
        run_name = get_kd_run_name(student_arch, teacher_arch, tokens,
                                    teacher_seed, alpha, student_seed)
        model_name = size_to_model_name(student_arch)

        # Get training task ID for depends_on
        training_task_id = get_kd_training_task_id(
            student_arch, teacher_arch, tokens, teacher_seed, alpha, student_seed
        ) if DEFAULT_ENABLE_DEPENDS_ON else None

        # PPL eval
        ppl_task_id = f"eval_ppl_kd_{run_name.replace('-', '_')}"
        tasks.append({
            "task_id": ppl_task_id,
            "run_name": run_name,
            "checkpoint_step": PRETRAIN_CHECKPOINT_STEP,
            "ckpt_dir": "ablate2",
            "model_name": model_name,
            "eval_type": "ppl",
            "param_only": False,
            "depends_on": training_task_id,
        })

        # Base ACC eval
        base_acc_task_id = f"eval_base_acc_kd_{run_name.replace('-', '_')}"
        tasks.append({
            "task_id": base_acc_task_id,
            "run_name": run_name,
            "checkpoint_step": PRETRAIN_CHECKPOINT_STEP,
            "ckpt_dir": "ablate2",
            "model_name": model_name,
            "eval_type": "base_acc",
            "param_only": False,
            "depends_on": training_task_id,
        })

    return tasks


def generate_teacher_tasks() -> list:
    """Generate eval tasks for teacher models (vanilla s42)."""
    tasks = []

    sizes = TEACHER_CONFIG["sizes"]
    tokens = TEACHER_CONFIG["tokens"]
    seed = TEACHER_CONFIG["seed"]
    ckpt_dir = TEACHER_CONFIG["ckpt_dir"]

    for size in sizes:
        run_name = get_teacher_run_name(size, tokens, seed)
        model_name = size_to_model_name(size)

        # Get training task ID for depends_on (convert task)
        training_task_id = get_teacher_training_task_id(size, tokens, seed) if DEFAULT_ENABLE_DEPENDS_ON else None

        # PPL eval
        ppl_task_id = f"eval_ppl_teacher_{run_name.replace('-', '_')}"
        tasks.append({
            "task_id": ppl_task_id,
            "run_name": run_name,
            "checkpoint_step": PRETRAIN_CHECKPOINT_STEP,
            "ckpt_dir": ckpt_dir,
            "model_name": model_name,
            "eval_type": "ppl",
            "param_only": True,
            "depends_on": training_task_id,
        })

        # Base ACC eval
        base_acc_task_id = f"eval_base_acc_teacher_{run_name.replace('-', '_')}"
        tasks.append({
            "task_id": base_acc_task_id,
            "run_name": run_name,
            "checkpoint_step": PRETRAIN_CHECKPOINT_STEP,
            "ckpt_dir": ckpt_dir,
            "model_name": model_name,
            "eval_type": "base_acc",
            "param_only": True,
            "depends_on": training_task_id,
        })

    return tasks


def generate_baseline_tasks() -> list:
    """Generate eval tasks for baseline models (vanilla, no distillation)."""
    tasks = []

    sizes = BASELINE_CONFIG["sizes"]
    tokens = BASELINE_CONFIG["tokens"]
    seeds = BASELINE_CONFIG["seeds"]
    ckpt_dir = BASELINE_CONFIG["ckpt_dir"]

    for size, seed in product(sizes, seeds):
        run_name = get_baseline_run_name(size, tokens, seed)
        model_name = size_to_model_name(size)

        # Get training task ID for depends_on
        training_task_id = get_baseline_training_task_id(size, tokens, seed) if DEFAULT_ENABLE_DEPENDS_ON else None

        # PPL eval
        ppl_task_id = f"eval_ppl_baseline_{run_name.replace('-', '_').replace('.', '_')}"
        tasks.append({
            "task_id": ppl_task_id,
            "run_name": run_name,
            "checkpoint_step": PRETRAIN_CHECKPOINT_STEP,
            "ckpt_dir": ckpt_dir,
            "model_name": model_name,
            "eval_type": "ppl",
            "param_only": False,
            "depends_on": training_task_id,
        })

        # Base ACC eval
        base_acc_task_id = f"eval_base_acc_baseline_{run_name.replace('-', '_').replace('.', '_')}"
        tasks.append({
            "task_id": base_acc_task_id,
            "run_name": run_name,
            "checkpoint_step": PRETRAIN_CHECKPOINT_STEP,
            "ckpt_dir": ckpt_dir,
            "model_name": model_name,
            "eval_type": "base_acc",
            "param_only": False,
            "depends_on": training_task_id,
        })

    return tasks


def generate_run_all_yaml(tasks: list, output_dir: str) -> str:
    """Generate a run_all.yaml file that lists all eval tasks."""
    # Sort tasks: baseline first, then teacher, then kd
    baseline_tasks = [t for t in tasks if "baseline" in t["task_id"]]
    teacher_tasks = [t for t in tasks if "teacher" in t["task_id"]]
    kd_tasks = [t for t in tasks if "kd" in t["task_id"]]
    sorted_tasks = baseline_tasks + teacher_tasks + kd_tasks

    lines = ["tasks:"]

    for task in sorted_tasks:
        task_id = task["task_id"]
        run_name = task["run_name"]
        checkpoint_step = task["checkpoint_step"]
        eval_type = task["eval_type"]
        depends_on = task.get("depends_on")

        lines.append(f"  - id: {task_id}")
        if DEFAULT_HIDE:
            lines.append(f"    hide: true")

        ckpt_dir = task.get("ckpt_dir", "ablate2")
        model_name = task.get("model_name", "qwen3-03b")
        param_only_flag = " --param_only" if task.get("param_only") else ""

        if eval_type == "ppl":
            lines.append(f"    run: bash train/ablate2-eval/eval_pretrain_ppl.sh {run_name} {model_name} {checkpoint_step} {ckpt_dir} --resume{param_only_flag}")
        elif eval_type == "base_acc":
            lines.append(f"    run: bash train/ablate2-eval/eval_base_acc.sh {run_name} {model_name} {checkpoint_step} {ckpt_dir} --resume{param_only_flag}")

        if depends_on:
            lines.append(f"    depends_on: {depends_on}")

    content = '\n'.join(lines) + '\n'

    output_path = os.path.join(output_dir, "run_all.yaml")
    with open(output_path, "w") as f:
        f.write(content)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation tasks for ablate2 experiments")
    parser.add_argument(
        "--exp",
        nargs="+",
        choices=["baseline", "teacher", "kd"],
        default=DEFAULT_EXPERIMENTS,
        help=f"Which experiments to generate eval tasks for (default: {DEFAULT_EXPERIMENTS})",
    )
    parser.add_argument(
        "--eval-type",
        choices=["base", "ppl", "base_acc"],
        default=DEFAULT_EVAL_TYPE,
        help=f"Which eval types to generate: base (PPL + base ACC), ppl, base_acc (default: {DEFAULT_EVAL_TYPE})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for generated YAML (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_tasks = []

    for exp_name in args.exp:
        if exp_name == "baseline":
            baseline_tasks = generate_baseline_tasks()
            if args.eval_type == "ppl":
                baseline_tasks = [t for t in baseline_tasks if t["eval_type"] == "ppl"]
            elif args.eval_type == "base_acc":
                baseline_tasks = [t for t in baseline_tasks if t["eval_type"] == "base_acc"]
            all_tasks.extend(baseline_tasks)

            if args.dry_run:
                for t in baseline_tasks:
                    print(f"Would generate: {t['task_id']}")
            else:
                for t in baseline_tasks:
                    print(f"Added task: {t['task_id']}")

        elif exp_name == "teacher":
            teacher_tasks = generate_teacher_tasks()
            if args.eval_type == "ppl":
                teacher_tasks = [t for t in teacher_tasks if t["eval_type"] == "ppl"]
            elif args.eval_type == "base_acc":
                teacher_tasks = [t for t in teacher_tasks if t["eval_type"] == "base_acc"]
            all_tasks.extend(teacher_tasks)

            if args.dry_run:
                for t in teacher_tasks:
                    print(f"Would generate: {t['task_id']}")
            else:
                for t in teacher_tasks:
                    print(f"Added task: {t['task_id']}")

        elif exp_name == "kd":
            kd_tasks = generate_kd_tasks()
            if args.eval_type == "ppl":
                kd_tasks = [t for t in kd_tasks if t["eval_type"] == "ppl"]
            elif args.eval_type == "base_acc":
                kd_tasks = [t for t in kd_tasks if t["eval_type"] == "base_acc"]
            all_tasks.extend(kd_tasks)

            if args.dry_run:
                for t in kd_tasks:
                    print(f"Would generate: {t['task_id']}")
            else:
                for t in kd_tasks:
                    print(f"Added task: {t['task_id']}")

    if not args.dry_run and all_tasks:
        print(f"\nTotal eval tasks: {len(all_tasks)}")

        # Count by type
        ppl_count = sum(1 for t in all_tasks if t["eval_type"] == "ppl")
        base_acc_count = sum(1 for t in all_tasks if t["eval_type"] == "base_acc")

        print(f"  PPL tasks: {ppl_count}")
        print(f"  Base ACC tasks: {base_acc_count}")

        # Generate run_all.yaml
        yaml_path = generate_run_all_yaml(all_tasks, args.output_dir)
        print(f"\nGenerated: {yaml_path}")


if __name__ == "__main__":
    main()
