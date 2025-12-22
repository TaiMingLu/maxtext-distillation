#!/usr/bin/env python3
"""
Analyze evaluation results from JSON files.

Parses result JSONs from PPL and ACC evaluations, extracts run metadata
from filenames, and organizes results by experiment configuration.

Supports reading from both local paths and GCS (gs://) paths directly.

Usage:
    python analyze_results.py                    # Analyze all (ppl + base_acc + sft_acc)
    python analyze_results.py ppl                # PPL only
    python analyze_results.py base_acc           # Base ACC only
    python analyze_results.py ppl base_acc       # PPL + Base ACC
    python analyze_results.py -o results.csv     # Save all to CSV
    python analyze_results.py --show-incomplete  # Only incomplete runs
    python analyze_results.py --base-dir gs://bucket/eval_1218  # Read from GCS directly
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union
import pandas as pd
from tqdm import tqdm

# GCS support
try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False


# Known PPL tasks (should match test_orbax_eval.py PPL_TASKS)
PPL_TASKS = [
    "c4", "wikitext", "cnn_dailymail", "finewebedu-train-0.001",
    "finewebedu-test-100M", "dm_mathematics", "gsm8k", "arxiv",
    "humaneval", "pg19", "codesearchnet", "pubmed_qa", "echr", "xquad"
]

# Known ACC tasks (should match test_orbax_eval.py ACC_TASKS)
ACC_TASKS = [
    "hellaswag", "winogrande", "arc_easy", "piqa", "boolq",
    "sciq", "mmlu", "mathqa"
]


def parse_run_name(run_name: str) -> dict:
    """
    Parse run name to extract experiment configuration.

    Examples:
        - exp1_llama3.1-1b-A05BT50BS42-a04-s43 -> KD run
        - exp2_llama3.1-1b-A8BT100BS42-a1-s43 -> KD run
        - llama05b-vanilla-100B-s42 -> teacher run
        - llama3.1-1b-finewebedu-vanilla-s43-50b -> baseline run
        - sft_exp1_llama3.1-1b-A05BT50BS42-a04-s43 -> SFT run

    Returns dict with:
        - is_teacher: bool (teacher models like llama05b-vanilla-100B-s42)
        - is_baseline: bool (single baseline like llama3.1-1b-finewebedu-vanilla-s43-50b)
        - is_sft: bool
        - is_kd: bool
        - exp_name: str (exp1, exp2, or None for teacher/baseline)
        - model_arch: str (e.g., llama3.1-1b)
        - teacher_arch: str or None (e.g., 05b, 1b, 3b, 8b)
        - teacher_tokens: str or None (e.g., 50B, 30B, 100B)
        - teacher_seed: int or None
        - alpha: float or None (KD alpha, e.g., 0.4, 0.5, 1.0)
        - student_seed: int or None
    """
    result = {
        "run_name": run_name,
        "is_teacher": False,
        "is_baseline": False,
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

    # Check for teacher model pattern: llama{size}-vanilla-{tokens}-s{seed}
    # Example: llama05b-vanilla-100B-s42, llama8b-vanilla-300B-s42
    teacher_pattern = r"llama(\d+b)-vanilla-(\d+B)-s(\d+)"
    match = re.match(teacher_pattern, name, re.IGNORECASE)
    if match:
        result["is_teacher"] = True
        size = match.group(1).lower()
        result["model_arch"] = f"llama3.1-{size}"
        result["teacher_arch"] = size
        result["teacher_tokens"] = match.group(2).upper()
        result["teacher_seed"] = int(match.group(3))
        return result

    # Check for baseline pattern: llama3.1-1b-finewebedu-vanilla-s43-50b
    baseline_pattern = r"(llama[\d.]+-(\d+b))-finewebedu-vanilla-s(\d+)-(\d+b)"
    match = re.match(baseline_pattern, name, re.IGNORECASE)
    if match:
        result["is_baseline"] = True
        result["model_arch"] = match.group(1)
        result["student_seed"] = int(match.group(3))
        result["teacher_tokens"] = match.group(4).upper()
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


def parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """Parse gs://bucket/path into (bucket, path)."""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Not a GCS path: {gcs_path}")
    path = gcs_path[5:]  # Remove gs://
    parts = path.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def list_gcs_json_files(bucket_name: str, prefix: str) -> list[str]:
    """List all JSON files in a GCS path."""
    if not HAS_GCS:
        raise ImportError("google-cloud-storage not installed. Run: pip install google-cloud-storage")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    json_files = []
    for blob in blobs:
        if blob.name.endswith(".json"):
            json_files.append(blob.name)

    return json_files


def load_gcs_json(bucket_name: str, blob_name: str) -> dict:
    """Load a JSON file directly from GCS."""
    if not HAS_GCS:
        raise ImportError("google-cloud-storage not installed. Run: pip install google-cloud-storage")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    content = blob.download_as_text()
    return json.loads(content)


def analyze_directory(results_dir: Union[str, Path], eval_type: str = "auto") -> pd.DataFrame:
    """
    Analyze all JSON files in a directory.

    Args:
        results_dir: Path to directory containing JSON files, or GCS path (gs://bucket/path)
        eval_type: 'ppl', 'base_acc', 'sft_acc', or 'auto' (detect from path)

    Returns DataFrame with parsed results.
    """
    results_dir_str = str(results_dir)
    is_gcs = results_dir_str.startswith("gs://")

    # Get directory name for auto-detection and display
    if is_gcs:
        dir_name = results_dir_str.rstrip("/").split("/")[-1].lower()
    else:
        results_dir = Path(results_dir)
        dir_name = results_dir.name.lower()

    # Auto-detect eval type from directory name
    if eval_type == "auto":
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

    # List JSON files (GCS or local)
    if is_gcs:
        bucket_name, prefix = parse_gcs_path(results_dir_str)
        # Ensure prefix ends with /
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        json_blobs = list_gcs_json_files(bucket_name, prefix)
        print(f"Found {len(json_blobs)} JSON files in {results_dir_str}")

        for blob_name in tqdm(json_blobs, desc=f"Loading {dir_name} (GCS)", unit="file"):
            # Parse filename
            filename = blob_name.split("/")[-1]
            file_info = parse_filename(filename)

            # Parse run name
            run_info = parse_run_name(file_info["run_name"])

            # Load JSON from GCS
            try:
                data = load_gcs_json(bucket_name, blob_name)
            except Exception as e:
                print(f"Error loading {blob_name}: {e}")
                continue

            # Extract metrics
            metrics = extract_metrics(data, eval_type)

            # Build row
            row = {
                "run_name": file_info["run_name"],
                "step": file_info["step"],
                "source_dir": results_dir_str,
                "eval_type": eval_type,
                **run_info,
                **metrics,
            }
            rows.append(row)
    else:
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
    Merge multiple result DataFrames, combining PPL and ACC results for same experiment config.

    PPL results come from base models, ACC results come from SFT models.
    We merge by experiment configuration (exp_name, teacher_arch, alpha, etc.)
    rather than run_name, since SFT runs have different names and steps.
    """
    if not dfs:
        return pd.DataFrame()

    if len(dfs) == 1:
        return dfs[0]

    # Concatenate all
    combined = pd.concat(dfs, ignore_index=True)

    # Define the experiment config columns to group by
    # These identify the same experiment across base/SFT evaluations
    exp_config_cols = ["exp_name", "teacher_arch", "alpha", "teacher_tokens", "student_seed",
                       "is_baseline", "is_teacher", "is_kd"]

    # Filter to only columns that exist
    exp_config_cols = [c for c in exp_config_cols if c in combined.columns]

    # Identify metric columns (ppl_* and acc_*)
    metric_cols = [c for c in combined.columns if c.startswith("ppl_") or c.startswith("acc_")]

    # Info columns that should be carried over (prefer non-SFT run_name for display)
    info_cols = [c for c in combined.columns if c not in metric_cols
                 and c not in ["eval_type", "source_dir", "_incomplete", "is_sft", "run_name", "step"]
                 and c not in exp_config_cols]

    # For each unique experiment config, merge all metrics
    grouped = combined.groupby(exp_config_cols, dropna=False)

    merged_rows = []
    for config_vals, group in grouped:
        row = dict(zip(exp_config_cols, config_vals if isinstance(config_vals, tuple) else [config_vals]))

        # Prefer non-SFT run_name for display (base model name)
        is_sft_col = group["is_sft"] if "is_sft" in group.columns else pd.Series([False] * len(group))
        non_sft = group[~is_sft_col.fillna(False)]
        if len(non_sft) > 0:
            row["run_name"] = non_sft["run_name"].iloc[0]
            row["step"] = non_sft["step"].iloc[0]
        else:
            # Fall back to SFT run_name (strip sft_ prefix for display)
            row["run_name"] = group["run_name"].iloc[0]
            if row["run_name"] and row["run_name"].startswith("sft_"):
                row["run_name"] = row["run_name"][4:]  # Strip sft_ prefix
            row["step"] = group["step"].iloc[0]

        # Take first non-null value for other info columns
        for col in info_cols:
            if col in group.columns:
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
        "run_name", "is_baseline", "is_teacher", "is_kd", "is_sft",
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

    # Sort by: is_baseline (first), is_teacher, exp_name, teacher_arch, alpha
    sort_cols = []
    ascending = []
    if "is_baseline" in result.columns:
        sort_cols.append("is_baseline")
        ascending.append(False)  # baseline first
    if "is_teacher" in result.columns:
        sort_cols.append("is_teacher")
        ascending.append(False)  # teacher second
    if "exp_name" in result.columns:
        sort_cols.append("exp_name")
        ascending.append(True)
    if "teacher_arch" in result.columns:
        sort_cols.append("teacher_arch")
        ascending.append(True)
    if "alpha" in result.columns:
        sort_cols.append("alpha")
        ascending.append(True)

    if sort_cols:
        result = result.sort_values(sort_cols, ascending=ascending)

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

    if "is_baseline" in df.columns:
        baseline = df["is_baseline"].sum()
        print(f"  Baseline: {baseline}")

    if "is_teacher" in df.columns:
        teacher = df["is_teacher"].sum()
        print(f"  Teacher: {teacher}")

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
    # Hardcoded base path for results
    # Default to GCS path for fresh reads (bypasses gcsfuse cache)
    BASE_RESULTS_DIR = "gs://taiming_us_central1/eval_1218"

    # Available result types and their directory names
    RESULT_TYPES = {
        "ppl": "ppl_results",
        "base_acc": "base_acc_results",
        "sft_acc": "acc_results",
    }

    parser = argparse.ArgumentParser(
        description="Analyze evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_results.py                    # Analyze all (ppl + base_acc + sft_acc)
    python analyze_results.py ppl                # PPL only
    python analyze_results.py base_acc           # Base ACC only
    python analyze_results.py ppl base_acc       # PPL + Base ACC
    python analyze_results.py -o results.csv     # Save to CSV
        """
    )
    parser.add_argument(
        "types",
        nargs="*",
        default=[],
        help="Result types to analyze: ppl, base_acc, sft_acc, or all (default: all)",
    )
    parser.add_argument(
        "--output", "-o",
        default="analysis_results.json",
        help="Output file path (default: analysis_results.json)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to file, only print to terminal",
    )
    parser.add_argument(
        "--base-dir",
        default=BASE_RESULTS_DIR,
        help=f"Base directory for results, supports gs:// paths (default: {BASE_RESULTS_DIR})",
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

    base_dir = args.base_dir
    is_gcs = base_dir.startswith("gs://")

    # Validate types
    valid_types = ["ppl", "base_acc", "sft_acc", "all"]
    for t in args.types:
        if t not in valid_types:
            print(f"Error: Invalid type '{t}'. Choose from: {', '.join(valid_types)}")
            return

    # Determine which result types to analyze
    if "all" in args.types or not args.types:
        types_to_analyze = list(RESULT_TYPES.keys())
    else:
        types_to_analyze = args.types

    # Analyze each directory
    dfs = []
    for result_type in types_to_analyze:
        dir_name = RESULT_TYPES.get(result_type)
        if not dir_name:
            print(f"Warning: Unknown result type: {result_type}")
            continue

        if is_gcs:
            # GCS path: just concatenate
            path = base_dir.rstrip("/") + "/" + dir_name
            print(f"Analyzing GCS path: {path}")
        else:
            # Local path
            path = Path(base_dir) / dir_name
            if not path.exists():
                print(f"Warning: Directory not found: {path}")
                continue
            if not path.is_dir():
                print(f"Warning: Not a directory: {path}")
                continue

        df = analyze_directory(path, result_type)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("No results found!")
        return

    # Always merge when analyzing multiple types (combines PPL + ACC for same runs)
    if len(dfs) > 1:
        result = merge_results(dfs)
    else:
        result = dfs[0]

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

    # Save to file (default behavior)
    if not args.no_save:
        output_path = args.output

        # Convert DataFrame to JSON-serializable format
        # Replace NaN/None with None for valid JSON
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(v) for v in obj]
            elif isinstance(obj, float) and (pd.isna(obj) or obj != obj):  # NaN check
                return None
            elif pd.isna(obj):
                return None
            return obj

        runs_list = result.to_dict(orient="records")
        runs_clean = clean_for_json(runs_list)

        result_dict = {
            "metadata": {
                "types_analyzed": types_to_analyze,
                "base_dir": str(base_dir),
                "total_runs": len(result),
                "incomplete_runs": int(result["_incomplete"].sum()) if "_incomplete" in result.columns else 0,
            },
            "runs": runs_clean,
        }

        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        print(f"\nSaved to: {output_path}")
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
