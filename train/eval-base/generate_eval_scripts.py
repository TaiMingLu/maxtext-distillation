#!/usr/bin/env python3
"""
Generate evaluation task YAML for distillation experiments.

- Pretrain models -> PPL evaluation (eval_pretrain_ppl.sh)
- Pretrain models -> ACC evaluation without chat template (eval_base_acc.sh)
- SFT models -> ACC evaluation with chat template (eval_sft_acc.sh)

Usage:
    python generate_eval_scripts.py
    python generate_eval_scripts.py --exp exp1 --eval-type base  # Default: base PPL + base ACC
    python generate_eval_scripts.py --exp exp1 --eval-type all   # All eval types
    python generate_eval_scripts.py --exp exp1 exp2 --eval-type ppl  # PPL only
    python generate_eval_scripts.py --exp exp1 --eval-type base_acc  # Base model ACC only (no chat template)
    python generate_eval_scripts.py --exp exp1 --eval-type sft  # SFT ACC only (with chat template)
"""

import argparse
import os
from itertools import product

# Experiment configurations (same as SFT generate_scripts.py)
EXP1_CONFIG = {
    "teacher_archs": ["05b", "1b", "3b", "8b"],
    "tokens": ["50B"],
    "alphas": [0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
}

EXP2_CONFIG = {
    "teacher_archs": ["05b", "1b", "3b", "8b"],
    "tokens": ["5B", "10B", "30B", "50B", "80B", "100B", "300B"],
    # "tokens": ["5B", "10B", "80B"],
    "alphas": [0.5, 1.0],
}

# Checkpoint steps
PRETRAIN_CHECKPOINT_STEP = 24999  # Default for 50B tokens
SFT_CHECKPOINT_STEP = 999

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

# Teacher baselines - all size/token combinations
TEACHER_CONFIG = {
    "sizes": ["05b", "1b", "3b", "8b"],
    "tokens": ["5B", "10B", "30B", "50B", "80B", "100B", "300B"],  # Uppercase for run_name
    # "tokens": ["5B", "10B", "80B"],
    "seed": 42,
    "ckpt_dir": "pretrain_param_only_v6",
}

# Baseline - single 1B model trained from scratch
BASELINE_CONFIG = {
    "run_name": "llama3.1-1b-finewebedu-vanilla-s43-50b",
    "model_name": "llama3.1-1b",
    "checkpoint_step": 24999,
    "ckpt_dir": "vanilla",
}

DEFAULT_TEACHER_SEED = 42
DEFAULT_STUDENT_SEED = 43

# Default CLI parameters (edit these instead of using command-line args)
DEFAULT_EXPERIMENTS = ["exp1", "exp2", "teacher", "baseline"]  # Options: "exp1", "exp2", "teacher", "baseline"
DEFAULT_EVAL_TYPE = "ppl"  # Options: "all", "base", "ppl", "base_acc", "sft"
DEFAULT_OUTPUT_DIR = "."
DEFAULT_TEACHER_ARCHS = None  # None = use experiment config, or e.g. ["05b", "1b"]
DEFAULT_TOKENS = None  # None = use experiment config, or e.g. ["50B", "100B"]
DEFAULT_ALPHAS = None  # None = use experiment config, or e.g. [0.5, 1.0]
DEFAULT_ENABLE_DEPENDS_ON = False  # Set to True to enable depends_on for task dependencies
DEFAULT_HIDE = True  # Set to False to make tasks visible by default

# Models that have NO training dependency (already trained)
# Only used when DEFAULT_ENABLE_DEPENDS_ON is True
NO_DEPENDENCY_TEACHERS = []


def get_checkpoint_step_for_tokens(tokens: str) -> int:
    """Get checkpoint step for a given token count."""
    tokens_lower = tokens.lower()
    if tokens_lower in TOKEN_TO_CHECKPOINT_STEP:
        return TOKEN_TO_CHECKPOINT_STEP[tokens_lower]
    # Fallback: calculate from tokens (step = tokens_in_billions * 500 - 1)
    tokens_num = int(tokens_lower.replace("b", ""))
    return tokens_num * 500 - 1


def alpha_to_str(alpha: float) -> str:
    """Convert alpha float to string for naming (e.g., 0.5 -> 'a05', 1.0 -> 'a1')."""
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


def get_pretrain_run_name(exp_name: str, teacher_arch: str, tokens: str,
                          teacher_seed: int, alpha: float, student_seed: int) -> str:
    """Get pretrain run name like exp1_llama3.1-1b-A1BT50BS42-a1-s43."""
    teacher_naming = get_teacher_naming(teacher_arch, tokens, teacher_seed)
    alpha_str = alpha_to_str(alpha)
    return f"{exp_name}_llama3.1-1b-{teacher_naming}-{alpha_str}-s{student_seed}"


def get_sft_run_name(exp_name: str, teacher_arch: str, tokens: str,
                     teacher_seed: int, alpha: float, student_seed: int) -> str:
    """Get SFT run name like sft_exp1_llama3.1-1b-A1BT50BS42-a1-s43."""
    pretrain_name = get_pretrain_run_name(exp_name, teacher_arch, tokens,
                                           teacher_seed, alpha, student_seed)
    return f"sft_{pretrain_name}"


def get_pretrain_training_task_id(exp_name: str, teacher_arch: str, tokens: str,
                                   teacher_seed: int, alpha: float, student_seed: int) -> str:
    """Get the training task ID for pretrain (for depends_on)."""
    teacher_naming = get_teacher_naming(teacher_arch, tokens, teacher_seed)
    alpha_str = alpha_to_str(alpha)
    return f"{exp_name}_llama1b_finewebedu_distill_soft_{teacher_naming}_{alpha_str}_s{student_seed}"


def get_sft_training_task_id(exp_name: str, teacher_arch: str, tokens: str,
                              teacher_seed: int, alpha: float, student_seed: int) -> str:
    """Get the training task ID for SFT (for depends_on)."""
    teacher_naming = get_teacher_naming(teacher_arch, tokens, teacher_seed)
    alpha_str = alpha_to_str(alpha)
    return f"sft_{exp_name}_llama3.1_1b_{teacher_naming}_{alpha_str}_s{student_seed}"


def generate_pretrain_ppl_task(
    exp_name: str,
    teacher_arch: str,
    tokens: str,
    teacher_seed: int,
    alpha: float,
    student_seed: int,
    checkpoint_step: int,
) -> dict:
    """Generate a pretrain PPL evaluation task."""
    run_name = get_pretrain_run_name(exp_name, teacher_arch, tokens,
                                      teacher_seed, alpha, student_seed)

    # Task ID for eval (use underscores)
    task_id = f"eval_ppl_{run_name.replace('-', '_')}"

    # Depends on training task (only if enabled and not in no-dependency list)
    depends_on = None
    if DEFAULT_ENABLE_DEPENDS_ON and teacher_arch not in NO_DEPENDENCY_TEACHERS:
        depends_on = get_pretrain_training_task_id(exp_name, teacher_arch, tokens,
                                                    teacher_seed, alpha, student_seed)

    return {
        "task_id": task_id,
        "run_name": run_name,
        "checkpoint_step": checkpoint_step,
        "ckpt_dir": exp_name,  # exp1 or exp2
        "eval_type": "ppl",
        "depends_on": depends_on,
    }


def generate_sft_acc_task(
    exp_name: str,
    teacher_arch: str,
    tokens: str,
    teacher_seed: int,
    alpha: float,
    student_seed: int,
    checkpoint_step: int,
) -> dict:
    """Generate an SFT ACC evaluation task."""
    run_name = get_sft_run_name(exp_name, teacher_arch, tokens,
                                 teacher_seed, alpha, student_seed)

    # Task ID for eval (use underscores)
    task_id = f"eval_acc_{run_name.replace('-', '_')}"

    # Depends on SFT training task (only if enabled)
    depends_on = None
    if DEFAULT_ENABLE_DEPENDS_ON:
        depends_on = get_sft_training_task_id(exp_name, teacher_arch, tokens,
                                               teacher_seed, alpha, student_seed)

    return {
        "task_id": task_id,
        "run_name": run_name,
        "checkpoint_step": checkpoint_step,
        "ckpt_dir": "sft",
        "eval_type": "acc",
        "depends_on": depends_on,
    }


def generate_base_acc_task(
    exp_name: str,
    teacher_arch: str,
    tokens: str,
    teacher_seed: int,
    alpha: float,
    student_seed: int,
    checkpoint_step: int,
) -> dict:
    """Generate a BASE model ACC evaluation task (no chat template)."""
    run_name = get_pretrain_run_name(exp_name, teacher_arch, tokens,
                                      teacher_seed, alpha, student_seed)

    # Task ID for eval (use underscores)
    task_id = f"eval_base_acc_{run_name.replace('-', '_')}"

    # Depends on pretrain training task (only if enabled and not in no-dependency list)
    depends_on = None
    if DEFAULT_ENABLE_DEPENDS_ON and teacher_arch not in NO_DEPENDENCY_TEACHERS:
        depends_on = get_pretrain_training_task_id(exp_name, teacher_arch, tokens,
                                                    teacher_seed, alpha, student_seed)

    return {
        "task_id": task_id,
        "run_name": run_name,
        "checkpoint_step": checkpoint_step,
        "ckpt_dir": exp_name,  # exp1 or exp2
        "eval_type": "base_acc",
        "depends_on": depends_on,
    }


def get_teacher_run_name(size: str, tokens: str, seed: int) -> str:
    """Get teacher run name like llama05b-vanilla-100B-s42."""
    return f"llama{size}-vanilla-{tokens}-s{seed}"


def get_teacher_training_task_id(size: str, tokens: str, seed: int) -> str:
    """Get the training task ID for teacher (for depends_on)."""
    return f"llama{size}_finewebedu_teacher_s{seed}_{tokens}"


def size_to_model_name(size: str) -> str:
    """Convert size string to model name (e.g., '8b' -> 'llama3.1-8b')."""
    return f"llama3.1-{size}"


def generate_teacher_tasks(checkpoint_step_sft: int) -> list:
    """Generate eval tasks for teacher models (all size/token combinations)."""
    tasks = []

    sizes = TEACHER_CONFIG["sizes"]
    tokens_list = TEACHER_CONFIG["tokens"]
    seed = TEACHER_CONFIG["seed"]
    ckpt_dir = TEACHER_CONFIG["ckpt_dir"]

    for size, tokens in product(sizes, tokens_list):
        name = get_teacher_run_name(size, tokens, seed)
        training_task_id = get_teacher_training_task_id(size, tokens, seed)
        model_name = size_to_model_name(size)
        # Get checkpoint step based on tokens
        checkpoint_step_pretrain = get_checkpoint_step_for_tokens(tokens)

        # Depends on only if enabled
        pretrain_depends_on = training_task_id if DEFAULT_ENABLE_DEPENDS_ON else None
        sft_depends_on = f"sft_{training_task_id}" if DEFAULT_ENABLE_DEPENDS_ON else None

        # PPL eval for teacher pretrain
        ppl_task_id = f"eval_ppl_teacher_{name.replace('-', '_')}"
        tasks.append({
            "task_id": ppl_task_id,
            "run_name": name,
            "checkpoint_step": checkpoint_step_pretrain,
            "ckpt_dir": ckpt_dir,
            "model_name": model_name,
            "eval_type": "ppl",
            "param_only": True,
            "depends_on": pretrain_depends_on,
        })

        # Base ACC eval for teacher pretrain (no chat template)
        base_acc_task_id = f"eval_base_acc_teacher_{name.replace('-', '_')}"
        tasks.append({
            "task_id": base_acc_task_id,
            "run_name": name,
            "checkpoint_step": checkpoint_step_pretrain,
            "ckpt_dir": ckpt_dir,
            "model_name": model_name,
            "eval_type": "base_acc",
            "param_only": True,
            "depends_on": pretrain_depends_on,
        })

        # ACC eval for teacher SFT (with chat template)
        sft_name = f"sft_{name}"
        acc_task_id = f"eval_acc_sft_teacher_{name.replace('-', '_')}"
        tasks.append({
            "task_id": acc_task_id,
            "run_name": sft_name,
            "checkpoint_step": checkpoint_step_sft,
            "ckpt_dir": "sft",
            "model_name": model_name,
            "eval_type": "acc",
            "depends_on": sft_depends_on,
        })

    return tasks


def generate_baseline_tasks(checkpoint_step_sft: int) -> list:
    """Generate eval tasks for the baseline model (single 1B model)."""
    tasks = []

    name = BASELINE_CONFIG["run_name"]
    model_name = BASELINE_CONFIG["model_name"]
    checkpoint_step = BASELINE_CONFIG["checkpoint_step"]
    ckpt_dir = BASELINE_CONFIG["ckpt_dir"]

    # PPL eval for baseline pretrain
    ppl_task_id = f"eval_ppl_baseline_{name.replace('-', '_').replace('.', '_')}"
    tasks.append({
        "task_id": ppl_task_id,
        "run_name": name,
        "checkpoint_step": checkpoint_step,
        "ckpt_dir": ckpt_dir,
        "model_name": model_name,
        "eval_type": "ppl",
        "depends_on": None,
    })

    # Base ACC eval for baseline pretrain (no chat template)
    base_acc_task_id = f"eval_base_acc_baseline_{name.replace('-', '_').replace('.', '_')}"
    tasks.append({
        "task_id": base_acc_task_id,
        "run_name": name,
        "checkpoint_step": checkpoint_step,
        "ckpt_dir": ckpt_dir,
        "model_name": model_name,
        "eval_type": "base_acc",
        "depends_on": None,
    })

    # ACC eval for baseline SFT (with chat template)
    sft_name = f"sft_{name}"
    acc_task_id = f"eval_acc_sft_baseline_{name.replace('-', '_').replace('.', '_')}"
    tasks.append({
        "task_id": acc_task_id,
        "run_name": sft_name,
        "checkpoint_step": checkpoint_step_sft,
        "ckpt_dir": "sft",
        "model_name": model_name,
        "eval_type": "acc",
        "depends_on": None,
    })

    return tasks


def generate_run_all_yaml(tasks: list, output_dir: str) -> str:
    """Generate a run_all.yaml file that lists all eval tasks.

    - Hide controlled by DEFAULT_HIDE
    - All tasks are resumable by default (--resume flag)
    """
    # Sort tasks: baseline first, then exp1, exp2, then teacher
    baseline_tasks = [t for t in tasks if "baseline" in t["task_id"]]
    exp1_tasks = [t for t in tasks if "exp1" in t["task_id"]]
    exp2_tasks = [t for t in tasks if "exp2" in t["task_id"]]
    teacher_tasks = [t for t in tasks if "teacher" in t["task_id"]]
    sorted_tasks = baseline_tasks + exp1_tasks + exp2_tasks + teacher_tasks

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

        # Get checkpoint directory (exp1, exp2, pretrain, or sft)
        ckpt_dir = task.get("ckpt_dir", "sft")
        # Get model name (default to llama3.1-1b for distill tasks)
        model_name = task.get("model_name", "llama3.1-1b")
        # Check if param_only checkpoint format
        param_only_flag = " --param_only" if task.get("param_only") else ""

        if eval_type == "ppl":
            lines.append(f"    run: bash train/eval-base/eval_pretrain_ppl.sh {run_name} {model_name} {checkpoint_step} {ckpt_dir} --resume{param_only_flag}")
        elif eval_type == "base_acc":
            lines.append(f"    run: bash train/eval-base/eval_base_acc.sh {run_name} {model_name} {checkpoint_step} {ckpt_dir} --resume{param_only_flag}")
        else:  # acc (SFT with chat template)
            lines.append(f"    run: bash train/eval-base/eval_sft_acc.sh {run_name} {model_name} {checkpoint_step} {ckpt_dir} --resume{param_only_flag}")

        if depends_on:
            lines.append(f"    depends_on: {depends_on}")

    content = '\n'.join(lines) + '\n'

    output_path = os.path.join(output_dir, "run_all.yaml")
    with open(output_path, "w") as f:
        f.write(content)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation tasks for distillation experiments")
    parser.add_argument(
        "--exp",
        nargs="+",
        choices=["exp1", "exp2", "teacher", "baseline"],
        default=DEFAULT_EXPERIMENTS,
        help=f"Which experiments to generate eval tasks for (default: {DEFAULT_EXPERIMENTS})",
    )
    parser.add_argument(
        "--eval-type",
        choices=["all", "base", "ppl", "base_acc", "sft"],
        default=DEFAULT_EVAL_TYPE,
        help=f"Which eval types to generate: base (PPL + base ACC), all (everything), ppl (PPL only), base_acc (ACC without chat template), sft (ACC with chat template) (default: {DEFAULT_EVAL_TYPE})",
    )
    parser.add_argument(
        "--teacher-archs",
        nargs="+",
        default=DEFAULT_TEACHER_ARCHS,
        help=f"Override teacher architectures (default: {DEFAULT_TEACHER_ARCHS})",
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        default=DEFAULT_TOKENS,
        help=f"Override tokens (default: {DEFAULT_TOKENS})",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=DEFAULT_ALPHAS,
        help=f"Override KD alpha values (default: {DEFAULT_ALPHAS})",
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
        if exp_name == "teacher":
            teacher_tasks = generate_teacher_tasks(SFT_CHECKPOINT_STEP)
            if args.eval_type == "ppl":
                teacher_tasks = [t for t in teacher_tasks if t["eval_type"] == "ppl"]
            elif args.eval_type == "sft":
                teacher_tasks = [t for t in teacher_tasks if t["eval_type"] == "acc"]
            elif args.eval_type == "base_acc":
                teacher_tasks = [t for t in teacher_tasks if t["eval_type"] == "base_acc"]
            elif args.eval_type == "base":
                # base = PPL + base_acc (no SFT)
                teacher_tasks = [t for t in teacher_tasks if t["eval_type"] in ["ppl", "base_acc"]]
            # else: all - keep all tasks
            all_tasks.extend(teacher_tasks)

            if args.dry_run:
                for t in teacher_tasks:
                    print(f"Would generate: {t['task_id']}")
            else:
                for t in teacher_tasks:
                    print(f"Added task: {t['task_id']}")
            continue

        if exp_name == "baseline":
            baseline_tasks = generate_baseline_tasks(SFT_CHECKPOINT_STEP)
            if args.eval_type == "ppl":
                baseline_tasks = [t for t in baseline_tasks if t["eval_type"] == "ppl"]
            elif args.eval_type == "sft":
                baseline_tasks = [t for t in baseline_tasks if t["eval_type"] == "acc"]
            elif args.eval_type == "base_acc":
                baseline_tasks = [t for t in baseline_tasks if t["eval_type"] == "base_acc"]
            elif args.eval_type == "base":
                baseline_tasks = [t for t in baseline_tasks if t["eval_type"] in ["ppl", "base_acc"]]
            all_tasks.extend(baseline_tasks)

            if args.dry_run:
                for t in baseline_tasks:
                    print(f"Would generate: {t['task_id']}")
            else:
                for t in baseline_tasks:
                    print(f"Added task: {t['task_id']}")
            continue

        if exp_name == "exp1":
            config = EXP1_CONFIG
        else:
            config = EXP2_CONFIG

        teacher_archs = args.teacher_archs or config["teacher_archs"]
        tokens_list = args.tokens or config["tokens"]
        alphas = args.alphas or config["alphas"]

        for teacher_arch, tokens, alpha in product(teacher_archs, tokens_list, alphas):
            # Generate PPL eval for pretrain
            if args.eval_type in ["all", "base", "ppl"]:
                ppl_task = generate_pretrain_ppl_task(
                    exp_name=exp_name,
                    teacher_arch=teacher_arch,
                    tokens=tokens,
                    teacher_seed=args.teacher_seed,
                    alpha=alpha,
                    student_seed=args.student_seed,
                    checkpoint_step=PRETRAIN_CHECKPOINT_STEP,
                )
                all_tasks.append(ppl_task)

                if args.dry_run:
                    dep_str = f" (depends: {ppl_task['depends_on']})" if ppl_task['depends_on'] else " (no dependency)"
                    print(f"Would generate: {ppl_task['task_id']}{dep_str}")
                else:
                    print(f"Added task: {ppl_task['task_id']}")

            # Generate base ACC eval for pretrain (no chat template)
            if args.eval_type in ["all", "base", "base_acc"]:
                base_acc_task = generate_base_acc_task(
                    exp_name=exp_name,
                    teacher_arch=teacher_arch,
                    tokens=tokens,
                    teacher_seed=args.teacher_seed,
                    alpha=alpha,
                    student_seed=args.student_seed,
                    checkpoint_step=PRETRAIN_CHECKPOINT_STEP,
                )
                all_tasks.append(base_acc_task)

                if args.dry_run:
                    dep_str = f" (depends: {base_acc_task['depends_on']})" if base_acc_task['depends_on'] else " (no dependency)"
                    print(f"Would generate: {base_acc_task['task_id']}{dep_str}")
                else:
                    print(f"Added task: {base_acc_task['task_id']}")

            # Generate ACC eval for SFT (with chat template)
            if args.eval_type in ["all", "sft"]:
                acc_task = generate_sft_acc_task(
                    exp_name=exp_name,
                    teacher_arch=teacher_arch,
                    tokens=tokens,
                    teacher_seed=args.teacher_seed,
                    alpha=alpha,
                    student_seed=args.student_seed,
                    checkpoint_step=SFT_CHECKPOINT_STEP,
                )
                all_tasks.append(acc_task)

                if args.dry_run:
                    print(f"Would generate: {acc_task['task_id']} (depends: {acc_task['depends_on']})")
                else:
                    print(f"Added task: {acc_task['task_id']}")

    if not args.dry_run and all_tasks:
        print(f"\nTotal eval tasks: {len(all_tasks)}")

        # Count by type
        ppl_count = sum(1 for t in all_tasks if t["eval_type"] == "ppl")
        base_acc_count = sum(1 for t in all_tasks if t["eval_type"] == "base_acc")
        sft_acc_count = sum(1 for t in all_tasks if t["eval_type"] == "acc")
        no_dep_count = sum(1 for t in all_tasks if t.get("depends_on") is None)

        print(f"  PPL tasks (pretrain): {ppl_count}")
        print(f"  Base ACC tasks (pretrain, no chat template): {base_acc_count}")
        print(f"  SFT ACC tasks (with chat template): {sft_acc_count}")
        print(f"  Tasks with no dependency: {no_dep_count}")

        # Generate run_all.yaml
        yaml_path = generate_run_all_yaml(all_tasks, args.output_dir)
        print(f"\nGenerated: {yaml_path}")


if __name__ == "__main__":
    main()
