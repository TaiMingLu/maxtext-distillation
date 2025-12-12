#!/usr/bin/env python3
"""
Analyze evaluation results from JSON files.

Parses result JSONs from PPL and ACC evaluations, extracts run metadata
from filenames, and organizes results by experiment configuration.

Usage:
    python analyze_results.py /path/to/results_dir
    python analyze_results.py /path/to/results_dir --output results.csv
    python analyze_results.py /path/to/ppl_results /path/to/acc_results --merge
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm import tqdm


# Known PPL tasks
PPL_TASKS = [
    "c4", "wikitext", "cnn_dailymail", "finewebedu-train-0.001",
    "dm_mathematics", "gsm8k", "arxiv", "humaneval", "pg19",
    "codesearchnet", "pubmed_qa", "echr", "xquad"
]

# Known ACC tasks
ACC_TASKS = [
    "hellaswag", "winogrande", "arc_easy", "piqa", "boolq",
    "sciq", "mmlu", "mathqa"
]


def parse_run_name(run_name: str) -> dict:
    """
    Parse run name to extract experiment configuration.

    Examples:
        - exp1_llama3.1-1b-A05BT50BS42-a04-s43 -> KD run
        - llama3.1-1b-finewebedu-vanilla-s42-50b -> vanilla run
        - sft_exp1_llama3.1-1b-A05BT50BS42-a04-s43 -> SFT run

    Returns dict with:
        - is_vanilla: bool
        - is_sft: bool
        - is_kd: bool
        - exp_name: str (exp1, exp2, or None for vanilla)
        - model_arch: str (e.g., llama3.1-1b)
        - teacher_arch: str or None (e.g., 05b, 1b, 3b, 8b)
        - teacher_tokens: str or None (e.g., 50B, 30B, 100B)
        - teacher_seed: int or None
        - alpha: float or None (KD alpha, e.g., 0.4, 0.5, 1.0)
        - student_seed: int or None
    """
    result = {
        "run_name": run_name,
        "is_vanilla": False,
        "is_sft": False,
        "is_kd": False,
        "exp_name": None,
        "model_arch": None,
        "teacher_arch": None,
        "teacher_tokens": None,
        "teacher_seed": None,
        "alpha": None,
        "student_seed": None,
    }

    name = run_name

    # Check for SFT prefix
    if name.startswith("sft_"):
        result["is_sft"] = True
        name = name[4:]  # Remove sft_ prefix

    # Check for vanilla
    if "vanilla" in name.lower():
        result["is_vanilla"] = True
        # Pattern: llama3.1-1b-finewebedu-vanilla-s42-50b
        match = re.match(r"(llama[\d.]+-([\d]+b)).*vanilla.*s(\d+)-(\d+)b", name, re.IGNORECASE)
        if match:
            result["model_arch"] = match.group(1)
            result["student_seed"] = int(match.group(3))
            result["teacher_tokens"] = f"{match.group(4)}B"  # For consistency, tokens trained
        return result

    # Check for KD experiment pattern: exp1_llama3.1-1b-A05BT50BS42-a04-s43
    # Pattern breakdown:
    # - exp_name: exp1, exp2, etc.
    # - model: llama3.1-1b
    # - teacher naming: A{arch}T{tokens}BS{seed} e.g., A05BT50BS42
    # - alpha: a{value} e.g., a04 (0.4), a1 (1.0)
    # - student seed: s{seed} e.g., s43

    kd_pattern = r"(exp\d+)_(llama[\d.]+-[\d]+b)-A(\d+B?)T(\d+)BS(\d+)-a(\d+)-s(\d+)"
    match = re.match(kd_pattern, name, re.IGNORECASE)

    if match:
        result["is_kd"] = True
        result["exp_name"] = match.group(1)
        result["model_arch"] = match.group(2)

        # Parse teacher arch (05B -> 0.5b, 1B -> 1b, 3B -> 3b, 8B -> 8b)
        teacher_arch_raw = match.group(3).lower()
        if teacher_arch_raw.endswith('b'):
            result["teacher_arch"] = teacher_arch_raw
        else:
            # e.g., "05" -> "05b", "1" -> "1b"
            result["teacher_arch"] = f"{teacher_arch_raw}b"

        result["teacher_tokens"] = f"{match.group(4)}B"
        result["teacher_seed"] = int(match.group(5))

        # Parse alpha: a04 -> 0.4, a1 -> 1.0, a02 -> 0.2
        alpha_str = match.group(6)
        if len(alpha_str) == 1:
            result["alpha"] = float(alpha_str)
        else:
            result["alpha"] = int(alpha_str) / 10.0

        result["student_seed"] = int(match.group(7))
        return result

    # Fallback: try to extract what we can
    # Look for model arch
    model_match = re.search(r"(llama[\d.]+-([\d]+b))", name, re.IGNORECASE)
    if model_match:
        result["model_arch"] = model_match.group(1)

    return result


def parse_filename(filename: str) -> dict:
    """
    Parse filename to extract run name and checkpoint step.

    Example: exp1_llama3.1-1b-A05BT50BS42-a04-s43_step24999.json
    """
    # Remove .json extension
    base = filename.replace(".json", "")

    # Extract step
    step_match = re.search(r"_step(\d+)$", base)
    step = int(step_match.group(1)) if step_match else None

    # Get run name (everything before _step)
    if step_match:
        run_name = base[:step_match.start()]
    else:
        run_name = base

    return {
        "filename": filename,
        "run_name": run_name,
        "step": step,
    }


def load_json_results(json_path: Path) -> Optional[dict]:
    """Load and parse a JSON result file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load {json_path}: {e}")
        return None


def extract_metrics(data: dict, eval_type: str) -> dict:
    """
    Extract metrics from a result JSON.

    Args:
        data: Parsed JSON data
        eval_type: 'ppl', 'base_acc', or 'sft_acc'

    Returns dict with metric names as keys and values as values.
    """
    metrics = {}

    # Check if incomplete
    metrics["_incomplete"] = data.get("_incomplete", False)

    # Extract PPL metrics
    ppl_data = data.get("ppl", {})
    for task, value in ppl_data.items():
        metrics[f"ppl_{task}"] = value

    # Extract ACC metrics (summary)
    acc_summary = data.get("lm_eval", {}).get("acc_summary", {})
    for task, value in acc_summary.items():
        metrics[f"acc_{task}"] = value

    return metrics


def analyze_directory(results_dir: Path, eval_type: str = "auto") -> pd.DataFrame:
    """
    Analyze all JSON files in a directory.

    Args:
        results_dir: Path to directory containing JSON files
        eval_type: 'ppl', 'base_acc', 'sft_acc', or 'auto' (detect from path)

    Returns DataFrame with parsed results.
    """
    # Auto-detect eval type from directory name
    if eval_type == "auto":
        dir_name = results_dir.name.lower()
        if "ppl" in dir_name:
            eval_type = "ppl"
        elif "sft" in dir_name or "acc" in dir_name:
            if "base" in dir_name:
                eval_type = "base_acc"
            else:
                eval_type = "sft_acc"
        else:
            eval_type = "unknown"

    rows = []
    json_files = list(results_dir.glob("*.json"))

    print(f"Found {len(json_files)} JSON files in {results_dir}")

    for json_path in tqdm(json_files, desc=f"Loading {results_dir.name}", unit="file"):
        # Parse filename
        file_info = parse_filename(json_path.name)

        # Parse run name
        run_info = parse_run_name(file_info["run_name"])

        # Load JSON
        data = load_json_results(json_path)
        if data is None:
            continue

        # Extract metrics
        metrics = extract_metrics(data, eval_type)

        # Combine all info
        row = {
            **file_info,
            **run_info,
            "eval_type": eval_type,
            "source_dir": str(results_dir),
            **metrics,
        }
        rows.append(row)

    if not rows:
        print(f"No valid results found in {results_dir}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def merge_results(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple result DataFrames, combining PPL and ACC results for same runs.
    """
    if not dfs:
        return pd.DataFrame()

    if len(dfs) == 1:
        return dfs[0]

    # Concatenate all
    combined = pd.concat(dfs, ignore_index=True)

    # Group by run_name and step, merge metrics
    # Identify metric columns (ppl_* and acc_*)
    metric_cols = [c for c in combined.columns if c.startswith("ppl_") or c.startswith("acc_")]
    info_cols = [c for c in combined.columns if c not in metric_cols and c not in ["eval_type", "source_dir", "_incomplete"]]

    # For each unique (run_name, step), merge all metrics
    grouped = combined.groupby(["run_name", "step"], dropna=False)

    merged_rows = []
    for (run_name, step), group in grouped:
        row = {}
        # Take first non-null value for info columns
        for col in info_cols:
            vals = group[col].dropna()
            row[col] = vals.iloc[0] if len(vals) > 0 else None

        # Merge metrics (take first non-null for each)
        for col in metric_cols:
            if col in group.columns:
                vals = group[col].dropna()
                row[col] = vals.iloc[0] if len(vals) > 0 else None

        # Track eval types
        row["eval_types"] = ",".join(sorted(group["eval_type"].dropna().unique()))

        # Incomplete if any source is incomplete
        row["_incomplete"] = group["_incomplete"].any() if "_incomplete" in group.columns else False

        merged_rows.append(row)

    return pd.DataFrame(merged_rows)


def create_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary DataFrame organized by experiment configuration.
    """
    if df.empty:
        return df

    # Define column order
    info_cols = [
        "run_name", "is_vanilla", "is_kd", "is_sft",
        "exp_name", "model_arch", "teacher_arch", "teacher_tokens",
        "alpha", "student_seed", "step", "_incomplete"
    ]

    # Get PPL columns (sorted)
    ppl_cols = sorted([c for c in df.columns if c.startswith("ppl_")])

    # Get ACC columns (sorted)
    acc_cols = sorted([c for c in df.columns if c.startswith("acc_")])

    # Build final column order
    final_cols = []
    for col in info_cols:
        if col in df.columns:
            final_cols.append(col)
    final_cols.extend(ppl_cols)
    final_cols.extend(acc_cols)

    # Add any remaining columns
    for col in df.columns:
        if col not in final_cols:
            final_cols.append(col)

    # Reorder and sort
    result = df[[c for c in final_cols if c in df.columns]].copy()

    # Sort by: is_vanilla (vanilla first), exp_name, teacher_arch, alpha
    sort_cols = []
    if "is_vanilla" in result.columns:
        sort_cols.append("is_vanilla")
    if "exp_name" in result.columns:
        sort_cols.append("exp_name")
    if "teacher_arch" in result.columns:
        sort_cols.append("teacher_arch")
    if "alpha" in result.columns:
        sort_cols.append("alpha")

    if sort_cols:
        result = result.sort_values(sort_cols, ascending=[False] + [True] * (len(sort_cols) - 1))

    return result


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    total = len(df)
    incomplete = df["_incomplete"].sum() if "_incomplete" in df.columns else 0
    complete = total - incomplete

    print(f"Total runs: {total}")
    print(f"  Complete: {complete}")
    print(f"  Incomplete: {incomplete}")

    if "is_vanilla" in df.columns:
        vanilla = df["is_vanilla"].sum()
        print(f"  Vanilla: {vanilla}")

    if "is_kd" in df.columns:
        kd = df["is_kd"].sum()
        print(f"  KD: {kd}")

    if "is_sft" in df.columns:
        sft = df["is_sft"].sum()
        print(f"  SFT: {sft}")

    # PPL stats
    ppl_cols = [c for c in df.columns if c.startswith("ppl_")]
    if ppl_cols:
        print(f"\nPPL Tasks: {len(ppl_cols)}")
        for col in ppl_cols:
            non_null = df[col].notna().sum()
            if non_null > 0:
                mean_val = df[col].mean()
                print(f"  {col}: {non_null} results, mean={mean_val:.4f}")

    # ACC stats
    acc_cols = [c for c in df.columns if c.startswith("acc_")]
    if acc_cols:
        print(f"\nACC Tasks: {len(acc_cols)}")
        for col in acc_cols:
            non_null = df[col].notna().sum()
            if non_null > 0:
                mean_val = df[col].mean()
                print(f"  {col}: {non_null} results, mean={mean_val:.4f}")

    # By experiment
    if "exp_name" in df.columns:
        print("\nBy Experiment:")
        for exp in df["exp_name"].dropna().unique():
            exp_df = df[df["exp_name"] == exp]
            print(f"  {exp}: {len(exp_df)} runs")

    # By teacher arch
    if "teacher_arch" in df.columns:
        print("\nBy Teacher Architecture:")
        for arch in sorted(df["teacher_arch"].dropna().unique()):
            arch_df = df[df["teacher_arch"] == arch]
            print(f"  {arch}: {len(arch_df)} runs")

    # By alpha
    if "alpha" in df.columns:
        print("\nBy Alpha:")
        for alpha in sorted(df["alpha"].dropna().unique()):
            alpha_df = df[df["alpha"] == alpha]
            print(f"  {alpha}: {len(alpha_df)} runs")


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument(
        "dirs",
        nargs="+",
        help="Directories containing JSON result files",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge results from multiple directories by run_name",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing summary statistics",
    )
    parser.add_argument(
        "--show-incomplete",
        action="store_true",
        help="Only show incomplete results",
    )
    parser.add_argument(
        "--show-complete",
        action="store_true",
        help="Only show complete results",
    )

    args = parser.parse_args()

    # Analyze each directory
    dfs = []
    for dir_path in args.dirs:
        path = Path(dir_path)
        if not path.exists():
            print(f"Warning: Directory not found: {path}")
            continue
        if not path.is_dir():
            print(f"Warning: Not a directory: {path}")
            continue

        df = analyze_directory(path)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("No results found!")
        return

    # Merge or concatenate
    if args.merge:
        result = merge_results(dfs)
    else:
        result = pd.concat(dfs, ignore_index=True)

    # Filter by completion status
    if args.show_incomplete and "_incomplete" in result.columns:
        result = result[result["_incomplete"] == True]
    elif args.show_complete and "_incomplete" in result.columns:
        result = result[result["_incomplete"] == False]

    # Create summary
    result = create_summary(result)

    # Print summary
    if not args.no_summary:
        print_summary_stats(result)

    # Save to CSV
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nSaved to: {args.output}")
    else:
        # Print to stdout
        print("\n" + "=" * 60)
        print("RESULTS TABLE")
        print("=" * 60)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        print(result.to_string())


if __name__ == "__main__":
    main()
