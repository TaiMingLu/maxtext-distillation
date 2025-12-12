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
    "tokens": ["30B", "50B", "100B"],
    "alphas": [0.5, 1.0],
}

# Checkpoint steps
PRETRAIN_CHECKPOINT_STEP = 24999
SFT_CHECKPOINT_STEP = 999

# Vanilla baselines - all size/token combinations
VANILLA_CONFIG = {
    "sizes": ["05b", "1b", "3b", "8b"],
    "tokens": ["30b", "50b", "100b", "300b", "1000b"],
    "seed": 42,
}

DEFAULT_TEACHER_SEED = 42
DEFAULT_STUDENT_SEED = 43

# Models that have NO training dependency (already trained)
# These are: vanilla model, and all models with 0.5B teacher (A05B) or 1B teacher (A1B)
NO_DEPENDENCY_TEACHERS = ["05b", "1b"]


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

    # Depends on training task (if not in no-dependency list)
    depends_on = None
    if teacher_arch not in NO_DEPENDENCY_TEACHERS:
        depends_on = get_pretrain_training_task_id(exp_name, teacher_arch, tokens,
                                                    teacher_seed, alpha, student_seed)

    return {
        "task_id": task_id,
        "run_name": run_name,
        "checkpoint_step": checkpoint_step,
        "checkpoint_type": "distill",
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

    # Depends on SFT training task
    depends_on = get_sft_training_task_id(exp_name, teacher_arch, tokens,
                                           teacher_seed, alpha, student_seed)

    return {
        "task_id": task_id,
        "run_name": run_name,
        "checkpoint_step": checkpoint_step,
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

    # Depends on pretrain training task (if not in no-dependency list)
    depends_on = None
    if teacher_arch not in NO_DEPENDENCY_TEACHERS:
        depends_on = get_pretrain_training_task_id(exp_name, teacher_arch, tokens,
                                                    teacher_seed, alpha, student_seed)

    return {
        "task_id": task_id,
        "run_name": run_name,
        "checkpoint_step": checkpoint_step,
        "checkpoint_type": "distill",
        "eval_type": "base_acc",
        "depends_on": depends_on,
    }


def get_vanilla_run_name(size: str, tokens: str, seed: int) -> str:
    """Get vanilla run name like llama8b-finewebedu-vanilla-s42-300b."""
    return f"llama{size}-finewebedu-vanilla-s{seed}-{tokens}"


def get_vanilla_training_task_id(size: str, tokens: str, seed: int) -> str:
    """Get the training task ID for vanilla (for depends_on)."""
    return f"llama{size}_finewebedu_vanilla_s{seed}_{tokens}"


def generate_vanilla_tasks(checkpoint_step_pretrain: int, checkpoint_step_sft: int) -> list:
    """Generate eval tasks for vanilla models (all size/token combinations)."""
    tasks = []

    sizes = VANILLA_CONFIG["sizes"]
    tokens_list = VANILLA_CONFIG["tokens"]
    seed = VANILLA_CONFIG["seed"]

    for size, tokens in product(sizes, tokens_list):
        name = get_vanilla_run_name(size, tokens, seed)
        training_task_id = get_vanilla_training_task_id(size, tokens, seed)

        # PPL eval for vanilla pretrain
        ppl_task_id = f"eval_ppl_vanilla_{name.replace('-', '_')}"
        tasks.append({
            "task_id": ppl_task_id,
            "run_name": name,
            "checkpoint_step": checkpoint_step_pretrain,
            "checkpoint_type": "pretrain",
            "eval_type": "ppl",
            "depends_on": training_task_id,
        })

        # Base ACC eval for vanilla pretrain (no chat template)
        base_acc_task_id = f"eval_base_acc_vanilla_{name.replace('-', '_')}"
        tasks.append({
            "task_id": base_acc_task_id,
            "run_name": name,
            "checkpoint_step": checkpoint_step_pretrain,
            "checkpoint_type": "pretrain",
            "eval_type": "base_acc",
            "depends_on": training_task_id,
        })

        # ACC eval for vanilla SFT (with chat template)
        sft_name = f"sft_{name}"
        acc_task_id = f"eval_acc_sft_vanilla_{name.replace('-', '_')}"
        tasks.append({
            "task_id": acc_task_id,
            "run_name": sft_name,
            "checkpoint_step": checkpoint_step_sft,
            "eval_type": "acc",
            "depends_on": f"sft_{training_task_id}",
        })

    return tasks


def generate_run_all_yaml(tasks: list, output_dir: str) -> str:
    """Generate a run_all.yaml file that lists all eval tasks.

    - Vanilla tasks are NOT hidden (visible by default)
    - Other tasks are hidden by default (hide: true)
    - All tasks are resumable by default (--resume flag)
    """
    # Sort tasks: vanilla first, then others
    vanilla_tasks = [t for t in tasks if "vanilla" in t["task_id"]]
    other_tasks = [t for t in tasks if "vanilla" not in t["task_id"]]
    sorted_tasks = vanilla_tasks + other_tasks

    lines = ["tasks:"]

    for task in sorted_tasks:
        task_id = task["task_id"]
        run_name = task["run_name"]
        checkpoint_step = task["checkpoint_step"]
        eval_type = task["eval_type"]
        depends_on = task.get("depends_on")
        is_vanilla = "vanilla" in task_id

        lines.append(f"  - id: {task_id}")
        # Do NOT hide vanilla tasks
        if not is_vanilla:
            lines.append(f"    hide: true")

        if eval_type == "ppl":
            ckpt_type = task.get("checkpoint_type", "distill")
            lines.append(f"    run: bash train/eval-base/eval_pretrain_ppl.sh {run_name} {checkpoint_step} {ckpt_type} --resume")
        elif eval_type == "base_acc":
            ckpt_type = task.get("checkpoint_type", "distill")
            lines.append(f"    run: bash train/eval-base/eval_base_acc.sh {run_name} {checkpoint_step} {ckpt_type} --resume")
        else:  # acc (SFT with chat template)
            lines.append(f"    run: bash train/eval-base/eval_sft_acc.sh {run_name} {checkpoint_step} --resume")

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
        choices=["exp1", "exp2", "vanilla"],
        default=["exp1", "exp2", "vanilla"],
        help="Which experiments to generate eval tasks for (default: all)",
    )
    parser.add_argument(
        "--eval-type",
        choices=["all", "base", "ppl", "base_acc", "sft"],
        default="base",
        help="Which eval types to generate: base (PPL + base ACC, default), all (everything), ppl (PPL only), base_acc (ACC without chat template), sft (ACC with chat template)",
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
        help="Output directory for generated YAML (default: current directory)",
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
        if exp_name == "vanilla":
            vanilla_tasks = generate_vanilla_tasks(PRETRAIN_CHECKPOINT_STEP, SFT_CHECKPOINT_STEP)
            if args.eval_type == "ppl":
                vanilla_tasks = [t for t in vanilla_tasks if t["eval_type"] == "ppl"]
            elif args.eval_type == "sft":
                vanilla_tasks = [t for t in vanilla_tasks if t["eval_type"] == "acc"]
            elif args.eval_type == "base_acc":
                vanilla_tasks = [t for t in vanilla_tasks if t["eval_type"] == "base_acc"]
            elif args.eval_type == "base":
                # base = PPL + base_acc (no SFT)
                vanilla_tasks = [t for t in vanilla_tasks if t["eval_type"] in ["ppl", "base_acc"]]
            # else: all - keep all tasks
            all_tasks.extend(vanilla_tasks)

            if args.dry_run:
                for t in vanilla_tasks:
                    print(f"Would generate: {t['task_id']}")
            else:
                for t in vanilla_tasks:
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
