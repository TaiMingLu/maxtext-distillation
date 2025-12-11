# import hydra
# requirements:
# pip install sacrebleu accelerate peft
import os

# Set trust_remote_code for HuggingFace datasets (required by some tasks like mathqa)
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"
import json
import shutil
import tempfile
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
import errno
from datetime import datetime

from tqdm import tqdm
from functools import partial
import datasets as hf_datasets
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator
from lm_eval.models.orbax_lm import OrbaxLM

from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.layers import models
from MaxText.layers import quantizations

from jax.sharding import Mesh
from jax.experimental import mesh_utils

import math

def _human_readable_bytes(num_bytes):
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(num_bytes)
    unit_idx = 0
    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"

def print_device_memory(note=""):
    try:
        devices = jax.devices()
    except Exception:
        devices = []
    try:
        backend = jax.default_backend()
    except Exception:
        backend = "unknown"
    header = f"[Device Memory]{' ' + note if note else ''} | backend={backend} | devices={len(devices)}"
    print(header)
    for dev in devices:
        # Compose a friendly name
        dev_kind = getattr(dev, "device_kind", getattr(dev, "kind", ""))
        dev_name = f"{getattr(dev, 'platform', 'unknown')}:{getattr(dev, 'id', '?')} ({dev_kind})"

        printed = False
        # Preferred: memory_stats() if available (often on TPU)
        try:
            if hasattr(dev, "memory_stats"):
                stats = dev.memory_stats()
                if isinstance(stats, dict) and stats:
                    in_use = stats.get("bytes_in_use") or stats.get("kb_in_use", 0) * 1024
                    peak = stats.get("peak_bytes_in_use") or stats.get("peak_kb_in_use", 0) * 1024
                    total = stats.get("total_memory") or stats.get("kb_total", 0) * 1024
                    parts = []
                    if in_use:
                        parts.append(f"in_use={_human_readable_bytes(in_use)}")
                    if peak:
                        parts.append(f"peak={_human_readable_bytes(peak)}")
                    if total:
                        parts.append(f"total={_human_readable_bytes(total)}")
                    if parts:
                        print(f"  {dev_name}: " + ", ".join(parts))
                        printed = True
        except Exception:
            pass

        # Fallbacks commonly available on GPU/others
        if not printed:
            alloc = None
            limit = None
            try:
                if hasattr(dev, "memory_allocated"):
                    alloc = dev.memory_allocated()
            except Exception:
                pass
            try:
                if hasattr(dev, "memory_limit"):
                    limit = dev.memory_limit()
            except Exception:
                pass
            try:
                if limit is None and hasattr(dev, "total_memory"):
                    limit = dev.total_memory()
            except Exception:
                pass

            if alloc is not None or limit is not None:
                parts = []
                if alloc is not None:
                    parts.append(f"in_use={_human_readable_bytes(int(alloc))}")
                if limit is not None:
                    parts.append(f"total={_human_readable_bytes(int(limit))}")
                print(f"  {dev_name}: " + ", ".join(parts))
                printed = True

        if not printed:
            print(f"  {dev_name}: memory stats unavailable")

def str2bool(v):
    if isinstance(v, bool):
        return v
    val = v.lower()
    if val in ("yes", "true", "t", "y", "1"):
        return True
    elif val in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (yes/no/true/false)")


def parse_task_limit_overrides(spec: str) -> dict:
    overrides = {}
    if not spec:
        return overrides

    def split_entry(entry: str):
        entry = entry.strip()
        if not entry:
            return None
        for sep in (":", "="):
            if sep in entry:
                return [part.strip() for part in entry.split(sep, 1)]
        parts = entry.split()
        if len(parts) == 2:
            return [parts[0].strip(), parts[1].strip()]
        raise ValueError(f"Invalid task limit entry '{entry}'. Use format task:limit")

    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        name_value = split_entry(raw)
        if not name_value:
            continue
        name, value = name_value
        if not name:
            raise ValueError(f"Invalid task name in task limit entry '{raw}'")
        try:
            overrides[name] = int(value)
        except ValueError as err:
            raise ValueError(f"Invalid integer value in task limit entry '{raw}'") from err
    return overrides


def copy_file_with_retries(src: Path, dst: Path, retries: int = 3, delay_s: float = 1.0):
    last_err = None
    for attempt in range(retries):
        try:
            if dst.exists():
                dst.unlink()
        except OSError:
            pass
        try:
            shutil.copy2(src, dst)
            return
        except OSError as err:
            last_err = err
            time.sleep(delay_s)
    if last_err is not None:
        raise last_err

PPL_TASKS = [
    "c4",
    "wikitext",
    "cnn_dailymail",
    "finewebedu-train-0.001",
    "dm_mathematics",
    "gsm8k",
    "arxiv",
    "humaneval",
    # "pg19",
    "codesearchnet",
    "pubmed_qa",
    "echr",
    "xquad",
]
    # "wikitext2",
    # "finewebedu-test-100M",
    # "dclm",
    # "arxiv_full",  # nick007x/arxiv-papers - huge dataset, use streaming

ACC_TASKS = [
    # Commonsense reasoning
    {
        "name": "hellaswag",  # uses validation split by default
        "num_fewshot": 0,
        "acc_key": "acc_norm,none",
        "acc_seq_length": 256,
        "acc_batch_size": 64,
    },
    {
        "name": "winogrande",
        "num_fewshot": 5,
        "acc_key": "acc,none",
        "acc_seq_length": 1024,
        "acc_batch_size": 16,
    },
    {
        "name": "arc_easy",
        "num_fewshot": 0,
        "acc_key": "acc_norm,none",
        "acc_seq_length": 256,
        "acc_batch_size": 64,
    },
    {
        "name": "piqa",
        "num_fewshot": 0,
        "acc_key": "acc_norm,none",
        "acc_seq_length": 512,
        "acc_batch_size": 32,
    },
    {
        "name": "boolq",
        "num_fewshot": 5,
        "acc_key": "acc,none",
        "acc_seq_length": 8192,
        "acc_batch_size": 2,
    },
    {
        "name": "sciq",
        "num_fewshot": 0,
        "acc_key": "acc,none",
        "acc_seq_length": 1024,
        "acc_batch_size": 16,
    },
    # Knowledge & QA
    {
        "name": "mmlu",
        "num_fewshot": 5,
        "acc_key": "acc,none",
        "acc_seq_length": 8192,
        "acc_batch_size": 2,
    },
    # Math - multiple choice (uses loglikelihood)
    {
        "name": "mathqa",
        "num_fewshot": 5,
        "acc_key": "acc,none",
        "acc_seq_length": 2048,
        "acc_batch_size": 8,
    },
    # # Truthfulness - uses loglikelihood (multiple choice)
    # {
    #     "name": "truthfulqa_mc1",
    #     "num_fewshot": 5,
    #     "acc_key": "acc,none",
    #     "acc_seq_length": 1024,
    #     "acc_batch_size": 16,
    # },
    # # NLI - uses loglikelihood
    # {
    #     "name": "rte",
    #     "num_fewshot": 5,
    #     "acc_key": "acc,none",
    #     "acc_seq_length": 4096,
    #     "acc_batch_size": 4,
    # },
]
    # NOTE: Tasks below require generate_until (text generation), which is not implemented
    # {
    #     "name": "nq_open",  # Natural Questions open-domain QA - requires generation
    #     "num_fewshot": 0,
    #     "acc_key": "exact_match,none",
    #     "acc_seq_length": 512,
    #     "acc_batch_size": 32,
    # },
    # Math - requires generation
    # {
    #     "name": "gsm8k",
    #     "num_fewshot": 0,
    #     "acc_key": "exact_match,strict-match",
    #     "acc_seq_length": 512,
    #     "acc_batch_size": 32,
    # },
    # {
    #     "name": "minerva_math",
    #     "num_fewshot": 0,
    #     "acc_key": "exact_match,none",
    #     "acc_seq_length": 256,
    #     "acc_batch_size": 64,
    # },
    
    # Code - requires generation (not implemented)
    # {
    #     "name": "humaneval",
    #     "num_fewshot": 0,
    #     "acc_key": "pass@1,none",
    #     "acc_seq_length": 512,
    #     "acc_batch_size": 32,
    # },
    # {
    #     "name": "mbpp",
    #     "num_fewshot": 0,
    #     "acc_key": "pass@1,none",
    #     "acc_seq_length": 512,
    #     "acc_batch_size": 32,
    # },

    # {
    #     "name": "anli_r1",
    #     "num_fewshot": 0,
    #     "acc_key": "acc,none",     
    #     "acc_seq_length": 256,
    #     "acc_batch_size": 64,
    # },
    # {
    #     "name": "anli_r2",
    #     "num_fewshot": 0,
    #     "acc_key": "acc,none",
    #     "acc_seq_length": 256,
    #     "acc_batch_size": 64,
    # },
    # {
    #     "name": "anli_r3",
    #     "num_fewshot": 0,
    #     "acc_key": "acc,none",
    #     "acc_seq_length": 256,
    #     "acc_batch_size": 64,
    # },

load_dataset = hf_datasets.load_dataset


def _select_metric_value(task_name, task_metrics, preferred_keys):
    """Return the first matching metric that should be surfaced."""

    def is_allowed(key: str) -> bool:
        if not key or key not in task_metrics:
            return False
        metric_name = key.split(",")[0]
        if metric_name.endswith("_stderr"):
            return False
        return True

    for key in preferred_keys:
        if is_allowed(key):
            return task_metrics[key]

    for key, value in task_metrics.items():
        if is_allowed(key):
            return value

    return None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_text_for_tokenizer(text: str) -> str:
    """Clean text to avoid tokenizer crashes on unusual Unicode characters.

    The Rust tokenizer can crash with 'NormalizedString bad split' on:
    - Surrogate pairs (U+D800 to U+DFFF)
    - Null bytes
    - Other malformed Unicode sequences
    - Certain control characters
    - Various problematic Unicode ranges

    This function aggressively cleans text to ensure tokenization succeeds.
    """
    if not text:
        return text

    # Step 1: Encode to bytes and decode back, removing any invalid sequences
    # This handles most encoding issues
    try:
        text = text.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='replace')
    except Exception:
        text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')

    # Step 2: Filter character by character - be aggressive
    cleaned_chars = []
    for char in text:
        try:
            codepoint = ord(char)
        except Exception:
            continue  # Skip any char that fails ord()

        # Skip null bytes
        if codepoint == 0:
            continue

        # Skip surrogates (0xD800-0xDFFF) - main cause of the error
        if 0xD800 <= codepoint <= 0xDFFF:
            continue

        # Skip control characters except tab, newline, carriage return
        if codepoint < 0x20 and codepoint not in (0x09, 0x0A, 0x0D):
            continue

        # Skip DELETE character
        if codepoint == 0x7F:
            continue

        # Skip C1 control characters (0x80-0x9F)
        if 0x80 <= codepoint <= 0x9F:
            continue

        # Skip replacement character (often indicates prior encoding issues)
        if codepoint == 0xFFFD:
            cleaned_chars.append(' ')  # Replace with space instead of skipping
            continue

        # Skip private use area characters that can cause issues
        if 0xE000 <= codepoint <= 0xF8FF:
            continue

        # Skip supplementary private use areas
        if 0xF0000 <= codepoint <= 0xFFFFD or 0x100000 <= codepoint <= 0x10FFFD:
            continue

        # Skip invalid Unicode codepoints
        if codepoint > 0x10FFFF:
            continue

        # Skip byte order marks
        if codepoint in (0xFEFF, 0xFFFE):
            continue

        # Skip noncharacters
        if codepoint in (0xFFFE, 0xFFFF) or (0xFDD0 <= codepoint <= 0xFDEF):
            continue

        cleaned_chars.append(char)

    text = ''.join(cleaned_chars)

    # Step 3: Final safety encode/decode
    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    except Exception:
        pass

    return text

def safe_tokenize(tokenizer, text: str, add_special_tokens: bool = True):
    """Tokenize text with automatic cleaning to prevent tokenizer crashes.

    If tokenization still fails after cleaning, falls back to ASCII-only.
    Uses BaseException to catch Rust panics (pyo3_runtime.PanicException).
    """
    cleaned_text = clean_text_for_tokenizer(text)

    try:
        return tokenizer.encode(cleaned_text, return_tensors='pt', add_special_tokens=add_special_tokens)
    except BaseException as e:
        # BaseException catches Rust panics (pyo3_runtime.PanicException) that Exception doesn't
        print(f"Warning: Tokenization failed after cleaning, falling back to ASCII-only. Error: {type(e).__name__}: {e}")
        ascii_text = cleaned_text.encode('ascii', errors='ignore').decode('ascii')
        try:
            return tokenizer.encode(ascii_text, return_tensors='pt', add_special_tokens=add_special_tokens)
        except BaseException as e2:
            # If even ASCII fails, return empty tensor
            print(f"Warning: Even ASCII tokenization failed. Error: {type(e2).__name__}: {e2}")
            import torch
            return torch.tensor([[tokenizer.bos_token_id or 0]], dtype=torch.long)

def safe_tokenize_batch(tokenizer, texts: list, add_special_tokens: bool = True, separator: str = "\n\n"):
    """Tokenize a list of texts individually and concatenate, skipping problematic ones.

    This is more robust than joining all texts and tokenizing at once, because
    if one text fails, we can skip it and continue with the rest.
    """
    import torch
    all_tokens = []
    skipped = 0

    for i, text in enumerate(texts):
        cleaned_text = clean_text_for_tokenizer(text)
        if separator and i > 0:
            cleaned_text = separator + cleaned_text

        try:
            tokens = tokenizer.encode(cleaned_text, return_tensors='pt', add_special_tokens=(add_special_tokens and i == 0))
            all_tokens.append(tokens.squeeze(0))
        except BaseException as e:
            # Try ASCII fallback
            ascii_text = cleaned_text.encode('ascii', errors='ignore').decode('ascii')
            try:
                tokens = tokenizer.encode(ascii_text, return_tensors='pt', add_special_tokens=(add_special_tokens and i == 0))
                all_tokens.append(tokens.squeeze(0))
            except BaseException as e2:
                skipped += 1
                if skipped <= 10:  # Only log first 10 skips
                    print(f"Warning: Skipping text {i} due to tokenization failure: {type(e2).__name__}")

    if skipped > 0:
        print(f"Total skipped texts due to tokenization failures: {skipped}/{len(texts)}")

    if not all_tokens:
        # All texts failed - return minimal tensor
        print(f"Warning: All {len(texts)} texts failed to tokenize!")
        return torch.tensor([[tokenizer.bos_token_id or 0]], dtype=torch.long)

    return torch.cat(all_tokens).unsqueeze(0)

def get_ppl_enc(task, tokenizer, add_special_tokens: bool = True):
    if task == 'wikitext':
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", trust_remote_code=True)
        text_column = "text"
        testenc = safe_tokenize(tokenizer, "\n\n".join(dataset[:32768][text_column]), add_special_tokens=add_special_tokens)
    elif task == 'wikitext2':
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
        text_column = "text"
        testenc = safe_tokenize(tokenizer, "\n\n".join(dataset[:32768][text_column]), add_special_tokens=add_special_tokens)
    elif task == 'cnn_dailymail':
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="train", trust_remote_code=True)
        text_column = "article"
        testenc = safe_tokenize(tokenizer, " ".join(dataset[:16384][text_column]), add_special_tokens=add_special_tokens)
    elif task == 'c4':
        dataset = load_dataset(
            "allenai/c4",
            data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
            split="train",
            verification_mode="no_checks",
            trust_remote_code=True
        )
        text_column = "text"
        testenc = safe_tokenize(tokenizer, " ".join(dataset[:8192][text_column]), add_special_tokens=add_special_tokens)
    elif task == 'dclm':
        data_path = "/home/zephyr/gcs-bucket/datasets/dclm/dclm_baseline_1.0.val.jsonl"
        dataset = load_dataset(
            "json",
            data_files={"train": data_path},
            split="train",
            verification_mode="no_checks"
        )
        text_column = "text"
        testenc = safe_tokenize(tokenizer, " ".join(dataset[:8192][text_column]), add_special_tokens=add_special_tokens)
    elif task == 'finewebedu-test-100M':
        dataset = load_dataset(
            "TaiMingLu/finewebedu-test-100M",
            split="test",
            trust_remote_code=True
        )
        text_column = "text"
        testenc = safe_tokenize(tokenizer, " ".join(dataset[:32768][text_column]), add_special_tokens=add_special_tokens)
    elif task == 'finewebedu-train-0.001':
        dataset = load_dataset(
            "TaiMingLu/finewebedu-train-0.001",
            split="train",
            trust_remote_code=True
        )
        text_column = "text"
        testenc = safe_tokenize(tokenizer, " ".join(dataset[:32768][text_column]), add_special_tokens=add_special_tokens)
    elif task == 'dm_mathematics':
        # DM-Mathematics from The Pile
        dataset = load_dataset(
            "timaeus/pile-dm_mathematics",
            split="train",
            trust_remote_code=True
        )
        text_column = "text"
        testenc = safe_tokenize(tokenizer, " ".join(dataset[:100000][text_column]), add_special_tokens=add_special_tokens)
    elif task == 'gsm8k':
        # GSM8K math word problems
        dataset = load_dataset(
            "openai/gsm8k",
            "main",
            split="train",
            trust_remote_code=True
        )
        # Combine question and answer with newlines
        # dataset[:N] returns dict of lists, so we need to zip them
        subset = dataset[:100000]
        texts = [f"{q}\n\n{a}" for q, a in zip(subset['question'], subset['answer'])]
        testenc = safe_tokenize(tokenizer, "\n\n".join(texts), add_special_tokens=add_special_tokens)
    elif task == 'arxiv':
        # arXiv summarization dataset
        dataset = load_dataset(
            "ccdv/arxiv-summarization",
            "section",
            split="train",
            trust_remote_code=True
        )
        text_column = "abstract"
        testenc = safe_tokenize(tokenizer, "\n\n".join(dataset[:100000][text_column]), add_special_tokens=add_special_tokens)
    elif task == 'arxiv_full':
        # arXiv papers (full) - use streaming to handle large dataset efficiently
        dataset = load_dataset(
            "nick007x/arxiv-papers",
            "section",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        # Shuffle with a buffer and take limited samples
        dataset = dataset.shuffle(seed=42, buffer_size=10000)
        texts = []
        for i, row in enumerate(dataset):
            if i >= 4096:
                break
            title = row.get('title', '') or ''
            abstract = row.get('abstract', '') or ''
            texts.append(f"{title}\n{abstract}")
        testenc = safe_tokenize(tokenizer, "\n\n".join(texts), add_special_tokens=add_special_tokens)
    elif task == 'humaneval':
        # HumanEval coding problems
        dataset = load_dataset(
            "openai/openai_humaneval",
            split="test",
            trust_remote_code=True
        )
        # Combine prompt and canonical_solution with newlines
        texts = [f"{p}\n\n{s}" for p, s in zip(dataset['prompt'], dataset['canonical_solution'])]
        testenc = safe_tokenize(tokenizer, "\n\n".join(texts), add_special_tokens=add_special_tokens)
    elif task == 'pg19':
        # PG19 books - each row is very long, so use fewer rows
        dataset = load_dataset(
            "emozilla/pg19",
            split="train",
            trust_remote_code=True
        )
        # Only take first 256 rows since each is very long
        texts = dataset[:8192]["text"]
        testenc = safe_tokenize(tokenizer, "\n\n".join(texts), add_special_tokens=add_special_tokens)
    elif task == 'codesearchnet':
        # CodeSearchNet - code documentation
        dataset = load_dataset(
            "claudios/code_search_net",
            "all",
            split="train",
            trust_remote_code=True
        )
        subset = dataset[:8192]
        texts = [f"{doc}\n\n{code}" for doc, code in zip(subset['func_documentation_string'], subset['whole_func_string'])]
        testenc = safe_tokenize(tokenizer, "\n\n".join(texts), add_special_tokens=add_special_tokens)
    elif task == 'pubmed_qa':
        # PubMedQA - biomedical QA
        dataset = load_dataset(
            "qiaojin/PubMedQA",
            "pqa_labeled",
            split="train",
            trust_remote_code=True
        )
        subset = dataset[:1000]
        texts = [f"{q}\n\n{a}" for q, a in zip(subset['question'], subset['long_answer'])]
        testenc = safe_tokenize(tokenizer, "\n\n".join(texts), add_special_tokens=add_special_tokens)
    elif task == 'echr':
        # ECHR - European Court of Human Rights cases
        dataset = load_dataset(
            "glnmario/ECHR",
            split="train",
            trust_remote_code=True
        )
        subset = dataset[:4096]
        texts = [f"{docname}\n\n{text}\n\n{conclusion}"
                 for docname, text, conclusion in zip(subset['docname'], subset['text'], subset['conclusion'])]
        testenc = safe_tokenize(tokenizer, "\n\n".join(texts), add_special_tokens=add_special_tokens)
    elif task == 'xquad':
        # XQuAD - multilingual QA (all 12 subsets)
        xquad_subsets = [
            "xquad.ar", "xquad.de", "xquad.el", "xquad.en", "xquad.es", "xquad.hi",
            "xquad.ro", "xquad.ru", "xquad.th", "xquad.tr", "xquad.vi", "xquad.zh"
        ]
        all_texts = []
        for subset_name in xquad_subsets:
            dataset = load_dataset(
                "google/xquad",
                subset_name,
                split="validation",
                trust_remote_code=True
            )
            for ctx, q, ans in zip(dataset['context'], dataset['question'], dataset['answers']):
                # Extract first answer text from the answers dict
                answer_text = ans['text'][0] if ans['text'] else ""
                all_texts.append(f"{ctx}\n\n{q}\n\n{answer_text}")
        testenc = safe_tokenize(tokenizer, "\n\n".join(all_texts), add_special_tokens=add_special_tokens)
    else:
        raise NotImplementedError(f"Unsupported task: {task}")
    return testenc

def get_ppl(
    model,
    tokenizer,
    tasks,
    batch_size: int = 1,
    calib_size: int = 256,
    max_length: int = 8192,
    add_special_tokens: bool = True,
    task_range: list = [],
    existing_ppl_results: dict = None,
    existing_ppl_times: dict = None,
    save_callback = None,
):
    """
    Args:
        existing_ppl_results: Previously completed PPL results to skip (for resume)
        existing_ppl_times: Previously completed PPL times (for resume)
        save_callback: Callable to save intermediate results after each task
    """
    # devices_in_data_fsdp = model.devices_in_data_fsdp
    # if batch_size % devices_in_data_fsdp != 0:
    #     print(f"🔁 Adjusting batch_size {batch_size} → {devices_in_data_fsdp * ((batch_size + devices_in_data_fsdp - 1) // devices_in_data_fsdp)} for device mesh compatibility.")
    #     batch_size = devices_in_data_fsdp * ((batch_size + devices_in_data_fsdp - 1) // devices_in_data_fsdp)
    if task_range:
        tasks = [t for t in tasks if t in task_range]

    # Initialize with existing results if resuming
    ppl_res = dict(existing_ppl_results) if existing_ppl_results else {}
    ppl_times = dict(existing_ppl_times) if existing_ppl_times else {}

    # Filter out already completed tasks
    remaining_tasks = [t for t in tasks if t not in ppl_res]
    if len(remaining_tasks) < len(tasks):
        skipped = len(tasks) - len(remaining_tasks)
        print(f"Resuming PPL evaluation: skipping {skipped} already completed tasks")
        print(f"  Completed: {[t for t in tasks if t in ppl_res]}")

    print(f"Starting PPL evaluation for tasks: {remaining_tasks}")
    for task in remaining_tasks:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Currently evaluating PPL task: {task}")
        start_ts = time.perf_counter()
        testenc = get_ppl_enc(task, tokenizer, add_special_tokens=add_special_tokens)
        tot_loss = 0
        tot_tokens = 0
        bs = batch_size
        seq_len = max_length
        nsamples = min(testenc.numel() // seq_len, calib_size)
        with torch.no_grad():
            for i in tqdm(range(0, nsamples, bs), desc=f"Evaluating PPL for {task}"):
                j = min(i + bs, nsamples)
                inputs = testenc[:,(i * seq_len):(j * seq_len)]
                inputs = inputs.reshape(j - i, seq_len)
                # import pdb; pdb.set_trace()
                
                outputs = model.forward(inputs)
                if hasattr(outputs, "logits"):
                    lm_logits = outputs.logits
                else:
                    lm_logits = outputs
                
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = inputs[:, 1:]
                
                loss_fct = nn.CrossEntropyLoss().to(shift_logits.device)
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
                
                tot_loss += loss.item() * seq_len * (j - i)
                tot_tokens += seq_len * (j - i)
                
            ppl_res[task] = torch.exp(torch.tensor(tot_loss / tot_tokens)).item()
            duration_s = time.perf_counter() - start_ts
            ppl_times[task] = duration_s
            print(f"{task} PPL: {ppl_res[task]} (time: {duration_s:.2f}s)")
            if task == "dclm":
                print("dclm val loss", math.log(ppl_res[task]))
            print_device_memory(f"after PPL task {task}")

            # Save intermediate results after each task
            if save_callback:
                save_callback(current_ppl_res=ppl_res, current_ppl_times=ppl_times)
                print(f"  -> Saved intermediate results after {task}")

    return ppl_res, ppl_times

def get_acc(
    model,
    tokenizer,
    tasks,
    task_range=[],
    limit=1000000,
    batch_size=32,
    per_task_limit=None,
    task_limit_overrides=None,
    task_seq_overrides=None,
    task_batch_overrides=None,
    existing_acc_results: dict = None,
    existing_acc_full: dict = None,
    existing_acc_times: dict = None,
    save_callback = None,
):
    """
    Args:
        existing_acc_results: Previously completed ACC summary results (for resume)
        existing_acc_full: Previously completed ACC full results (for resume)
        existing_acc_times: Previously completed ACC times (for resume)
        save_callback: Callable to save intermediate results after each task
    """
    # lm_eval_model = models.orbax_lm.HFLM(
    #     pretrained=model,
    #     tokenizer=tokenizer,
    #     generation_kwargs={
    #         "do_sample": True,
    #         "temperature": 0.2,
    #         "top_p": 0.95,
    #     }
    # )
    if task_range:
        tasks = [cfg for cfg in tasks if cfg["name"] in task_range]

    # Initialize with existing results if resuming
    acc_res = dict(existing_acc_results) if existing_acc_results else {}
    full_res_by_task = dict(existing_acc_full) if existing_acc_full else {}
    acc_times = dict(existing_acc_times) if existing_acc_times else {}

    # Filter out already completed tasks
    completed_task_names = set(acc_res.keys())
    remaining_tasks = [cfg for cfg in tasks if cfg["name"] not in completed_task_names]
    if len(remaining_tasks) < len(tasks):
        skipped = len(tasks) - len(remaining_tasks)
        print(f"Resuming ACC evaluation: skipping {skipped} already completed tasks")
        print(f"  Completed: {list(completed_task_names)}")

    print("tasks to evaluate:")
    print(json.dumps(remaining_tasks, indent=2))
    print(f"Starting accuracy evaluation with batch_size={batch_size}...")
    task_limit_overrides = task_limit_overrides or {}
    task_seq_overrides = task_seq_overrides or {}
    task_batch_overrides = task_batch_overrides or {}
    default_seq_len = getattr(model, "eval_seq_len", None)
    default_loglikelihood_bs = getattr(model, "loglikelihood_batch_size", None)

    for cfg in remaining_tasks:
        task = cfg["name"]
        cfg_limit = cfg.get("limit")
        cfg_batch_size = cfg.get("acc_batch_size", batch_size)

        if task in task_batch_overrides:
            task_batch_size = task_batch_overrides[task]
        else:
            task_batch_size = cfg_batch_size

        if task in task_limit_overrides:
            eval_limit = task_limit_overrides[task]
        elif per_task_limit is not None:
            eval_limit = per_task_limit
        elif cfg_limit is not None:
            eval_limit = cfg_limit
        else:
            eval_limit = limit

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Currently evaluating ACC task: {task} "
            f"(fewshot={cfg['num_fewshot']}, batch_size={task_batch_size}, limit={eval_limit})"
        )
        start_ts = time.perf_counter()
        cfg_seq_len = cfg.get("acc_seq_length")
        if task in task_seq_overrides:
            desired_seq_len = task_seq_overrides[task]
        elif cfg_seq_len is not None:
            desired_seq_len = cfg_seq_len
        else:
            desired_seq_len = default_seq_len

        if desired_seq_len is not None and hasattr(model, "set_eval_seq_length"):
            model.set_eval_seq_length(desired_seq_len)
        print(f"  -> Using acc_seq_length={getattr(model, 'eval_seq_len', desired_seq_len)}")

        if task_batch_size is not None and hasattr(model, "set_loglikelihood_batch_size"):
            model.set_loglikelihood_batch_size(task_batch_size)

        res = evaluator.simple_evaluate(
            model=model,
            tasks=[task],
            num_fewshot=cfg["num_fewshot"],
            max_batch_size=task_batch_size,
            log_samples=True,
            # task_kwargs={"limit": 256},
            confirm_run_unsafe_code=True,
            limit=eval_limit,
            apply_chat_template=test_args.apply_chat_template,
        )
        
        task_metrics = res['results'][task]
        print(task_metrics)
        duration_s = time.perf_counter() - start_ts
        times_key = f"{task}/fewshot={cfg['num_fewshot']}"
        acc_times[times_key] = duration_s
        print(f"{times_key} ACC eval time: {duration_s:.2f}s")
        acc_key = cfg["acc_key"]
        preferred_keys = []
        if acc_key is not None:
            preferred_keys.append(acc_key)
        preferred_keys.extend(["acc,none", "acc_norm,none", "acc", "acc_norm"])
        summary_value = _select_metric_value(task, task_metrics, preferred_keys)
        if summary_value is not None:
            acc_res[task] = summary_value
            print(f"{task} ACC: {summary_value:.4f} (time: {duration_s:.2f}s)")
        full_res_by_task[task] = res
        print_device_memory(f"after ACC task {times_key}")

        # Save intermediate results after each task
        if save_callback:
            save_callback(current_acc_res=acc_res, current_acc_full=full_res_by_task, current_acc_times=acc_times)
            print(f"  -> Saved intermediate results after {task}")

    if default_seq_len is not None and hasattr(model, "set_eval_seq_length"):
        model.set_eval_seq_length(default_seq_len)
    if default_loglikelihood_bs is not None and hasattr(model, "set_loglikelihood_batch_size"):
        model.set_loglikelihood_batch_size(default_loglikelihood_bs)

    return acc_res, full_res_by_task, acc_times

def cast_orbax_state_to_bf16(orbax_state):
    def cast_fn(x):
        # Skip ShapeDtypeStruct and other non-array objects
        if not hasattr(x, "astype"):
            return x
        if hasattr(x, "dtype") and x.dtype == jnp.float32:
            return x.astype(jnp.bfloat16)
        return x
    casted_params = jax.tree_util.tree_map(cast_fn, orbax_state.params)
    orbax_state = orbax_state.replace(params=casted_params)
    return orbax_state

def main(config, test_args):
    ppl_seq_len = test_args.ppl_seq_length or getattr(config, "max_target_length", None)
    if ppl_seq_len is None:
        raise ValueError("ppl sequence length must be set either via config.max_target_length or --ppl_seq_length")
    acc_seq_len = test_args.acc_seq_length or ppl_seq_len

    if getattr(config, "max_target_length", None) != ppl_seq_len:
        print(f"Overriding config.max_target_length from {getattr(config, 'max_target_length', None)} to {ppl_seq_len} for model init/PPL")
        config.max_target_length = ppl_seq_len
    else:
        print(f"Using config.max_target_length={config.max_target_length} for model init/PPL")
    print(f"Accuracy evaluation sequence length set to {acc_seq_len}")

    tokenizer = AutoTokenizer.from_pretrained(test_args.hf_model_path)
    task_limit_overrides = parse_task_limit_overrides(test_args.acc_task_limits)
    task_seq_overrides = parse_task_limit_overrides(test_args.acc_task_seq_lens)
    task_batch_overrides = parse_task_limit_overrides(test_args.acc_task_batch_sizes)
    
    init_rng = jax.random.PRNGKey(config.init_weights_seed)
    init_rng, rng1 = jax.random.split(init_rng)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
    quant = quantizations.configure_quantization(config)
    orbax_model = models.Transformer(config, mesh, quant=quant)
    orbax_state, _ = maxtext_utils.setup_decode_state(orbax_model, config, rng1, mesh, None)

    # Debug: Check if params are actual arrays or abstract shapes
    def check_params_type(params, prefix=""):
        leaves = jax.tree_util.tree_leaves(params)
        print(f"[DEBUG] {prefix}Number of leaves: {len(leaves)}")
        if leaves:
            sample_leaf = leaves[0]
            print(f"[DEBUG] {prefix}First param leaf type: {type(sample_leaf)}, shape: {getattr(sample_leaf, 'shape', 'N/A')}")
            if hasattr(sample_leaf, 'sharding'):
                print(f"[DEBUG] {prefix}Sharding: {sample_leaf.sharding}")
            # Check if it's a ShapeDtypeStruct (abstract) - it won't have .astype method
            if not hasattr(sample_leaf, 'astype'):
                print(f"[DEBUG] {prefix}WARNING: Params appear to be abstract (ShapeDtypeStruct), not actual arrays!")
                return False
            # Also check a few more leaves to be sure
            for i, leaf in enumerate(leaves[:5]):
                if not hasattr(leaf, 'astype'):
                    print(f"[DEBUG] {prefix}Leaf {i} is abstract: {type(leaf)}")
                    return False
        return True

    # Check params structure
    print(f"[DEBUG] orbax_state.params type: {type(orbax_state.params)}")
    if hasattr(orbax_state.params, 'keys'):
        print(f"[DEBUG] orbax_state.params keys: {list(orbax_state.params.keys())[:10]}")
        # Check if params has nested 'params' key (common in Flax)
        if 'params' in orbax_state.params:
            print(f"[DEBUG] Found nested 'params' key, checking inner structure...")
            inner_params = orbax_state.params['params']
            if hasattr(inner_params, 'keys'):
                print(f"[DEBUG] inner params keys: {list(inner_params.keys())[:10]}")

    params_ok = check_params_type(orbax_state.params, "orbax_state.params: ")
    if not params_ok:
        # Try to provide more diagnostic info
        print(f"[DEBUG] Checkpoint path: {config.load_parameters_path}")
        print(f"[DEBUG] Attempting to list checkpoint contents...")
        import subprocess
        try:
            result = subprocess.run(
                ["gsutil", "ls", config.load_parameters_path],
                capture_output=True, text=True, timeout=30
            )
            print(f"[DEBUG] gsutil ls output: {result.stdout[:500] if result.stdout else 'empty'}")
            if result.stderr:
                print(f"[DEBUG] gsutil ls stderr: {result.stderr[:500]}")
        except Exception as e:
            print(f"[DEBUG] Failed to list checkpoint: {e}")

        raise RuntimeError(
            "Checkpoint loading failed: params are ShapeDtypeStruct instead of actual arrays. "
            "This usually means the checkpoint path is incorrect or the checkpoint structure doesn't match. "
            f"Checkpoint path: {config.load_parameters_path}"
        )

    orbax_state = cast_orbax_state_to_bf16(orbax_state)
    print_device_memory("after model init")
    
    _, _, state_mesh_shardings = maxtext_utils.get_abstract_state(
        orbax_model, None, config, rng1, mesh, is_training=False
    )

    model = OrbaxLM(
        orbax_model,
        orbax_state,
        tokenizer,
        config,
        state_mesh_shardings,
        mesh,
        loglikelihood_batch_size=test_args.acc_batch_size,
        max_loglikelihood_seq_length=acc_seq_len,
    )
    
    # Initialize results
    ppl_res, ppl_times = {}, {}
    acc_res, acc_full, acc_times = {}, {}, {}

    # Determine save path for incremental saves
    save_path = None
    if getattr(test_args, "eval_save_dir", ""):
        os.makedirs(test_args.eval_save_dir, exist_ok=True)
        save_dir = Path(test_args.eval_save_dir)
        save_path = save_dir / f"{getattr(config, 'run_name', 'results')}.json"

    # Load existing results if resuming
    if test_args.resume:
        if save_path and save_path.exists():
            print(f"Resuming from existing results: {save_path}")
            try:
                with open(save_path, 'r') as f:
                    existing_results = json.load(f)
                ppl_res = existing_results.get("ppl", {})
                ppl_times = existing_results.get("timing", {}).get("ppl", {})
                acc_res = existing_results.get("lm_eval", {}).get("acc_summary", {})
                acc_full = existing_results.get("lm_eval", {}).get("per_task", {})
                acc_times = existing_results.get("timing", {}).get("lm_eval", {})
                print(f"  Loaded {len(ppl_res)} PPL results, {len(acc_res)} ACC results")
            except Exception as e:
                print(f"  Warning: Failed to load existing results: {e}")
                print(f"  Starting fresh evaluation")
                ppl_res, ppl_times = {}, {}
                acc_res, acc_full, acc_times = {}, {}, {}
        else:
            print(f"Resume enabled but no existing results found at: {save_path}")
            print(f"  Starting fresh evaluation (results will be saved incrementally)")

    # Helper to serialize numpy/jax types
    def to_serializable(obj):
        try:
            import numpy as _np
            import jax.numpy as _jnp
        except Exception:
            _np, _jnp = None, None
        if _np is not None and isinstance(obj, _np.generic):
            return obj.item()
        if _jnp is not None and hasattr(obj, "dtype") and hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    # Create save callback for incremental saves
    # NOTE: The callback receives current results as parameters because get_ppl/get_acc
    # create local dicts that would not be captured by closure
    def save_intermediate_results(current_ppl_res=None, current_ppl_times=None,
                                   current_acc_res=None, current_acc_full=None, current_acc_times=None):
        if not save_path:
            return
        # Use provided values if given, otherwise fall back to outer scope (for final save)
        _ppl_res = current_ppl_res if current_ppl_res is not None else ppl_res
        _ppl_times = current_ppl_times if current_ppl_times is not None else ppl_times
        _acc_res = current_acc_res if current_acc_res is not None else acc_res
        _acc_full = current_acc_full if current_acc_full is not None else acc_full
        _acc_times = current_acc_times if current_acc_times is not None else acc_times
        results_payload = {
            "run_name": getattr(config, "run_name", ""),
            "model_name": getattr(config, "model_name", ""),
            "eval_mode": test_args.eval_mode,
            "limit": test_args.limit,
            "tasks_requested": test_args.tasks,
            "add_special_tokens": test_args.add_special_tokens,
            "ppl": _ppl_res,
            "lm_eval": {
                "acc_summary": _acc_res,
                "per_task": _acc_full,
            },
            "timing": {
                "ppl": _ppl_times,
                "lm_eval": _acc_times,
            },
            "_incomplete": True,  # Mark as incomplete until final save
        }
        temp_path = save_path.with_suffix('.json.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(results_payload, f, indent=2, default=to_serializable)
            shutil.move(str(temp_path), save_path)
        except Exception as e:
            print(f"  Warning: Failed to save intermediate results: {e}")

    # Run PPL evaluation if mode is 'ppl' or 'all'
    if test_args.eval_mode in ["ppl", "all"]:
        print_device_memory("before PPL eval")
        ppl_res, ppl_times = get_ppl(
            model,
            tokenizer,
            batch_size=test_args.ppl_batch_size,
            calib_size=min(256, test_args.limit),
            max_length=config.max_target_length,
            tasks=PPL_TASKS,
            add_special_tokens=test_args.add_special_tokens,
            task_range=test_args.tasks,
            existing_ppl_results=ppl_res if test_args.resume else None,
            existing_ppl_times=ppl_times if test_args.resume else None,
            save_callback=save_intermediate_results if save_path else None,
        )
        print(ppl_res)
        print({"ppl_times_s": ppl_times})

    # Run ACC evaluation if mode is 'acc' or 'all'
    if test_args.eval_mode in ["acc", "all"]:
        print_device_memory("before ACC eval")
        acc_res, acc_full, acc_times = get_acc(
            model,
            tokenizer,
            tasks=ACC_TASKS,
            task_range=test_args.tasks,
            limit=test_args.limit,
            batch_size=test_args.acc_batch_size,
            per_task_limit=test_args.acc_limit,
            task_limit_overrides=task_limit_overrides,
            task_seq_overrides=task_seq_overrides,
            task_batch_overrides=task_batch_overrides,
            existing_acc_results=acc_res if test_args.resume else None,
            existing_acc_full=acc_full if test_args.resume else None,
            existing_acc_times=acc_times if test_args.resume else None,
            save_callback=save_intermediate_results if save_path else None,
        )
        print(acc_res)
        print({"acc_times_s": acc_times})

    # Final save (mark as complete by removing _incomplete flag)
    if save_path:
        results_payload = {
            "run_name": getattr(config, "run_name", ""),
            "model_name": getattr(config, "model_name", ""),
            "eval_mode": test_args.eval_mode,
            "limit": test_args.limit,
            "tasks_requested": test_args.tasks,
            "add_special_tokens": test_args.add_special_tokens,
            "ppl": ppl_res,
            "lm_eval": {
                "acc_summary": acc_res,
                "per_task": acc_full,
            },
            "timing": {
                "ppl": ppl_times,
                "lm_eval": acc_times,
            },
            # No _incomplete flag = evaluation is complete
        }

        temp_root = Path.home() / ".maxtext_eval_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)

        tmp_path = None
        with tempfile.NamedTemporaryFile("w", dir=str(temp_root), suffix=".json", delete=False) as tmp_file:
            json.dump(results_payload, tmp_file, indent=2, default=to_serializable)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            tmp_path = Path(tmp_file.name)

        try:
            shutil.move(str(tmp_path), save_path)
            tmp_path = None
        except OSError as err:
            needs_fallback = err.errno in (errno.EXDEV, errno.ESTALE)
            if needs_fallback:
                print(
                    f"shutil.move failed with errno {err.errno}; falling back to copy2 with retries"
                )
                try:
                    copy_file_with_retries(tmp_path, save_path)
                    tmp_path.unlink(missing_ok=True)
                    tmp_path = None
                except Exception:
                    try:
                        if tmp_path is not None and tmp_path.exists():
                            tmp_path.unlink()
                    except Exception:
                        pass
                    raise
            else:
                try:
                    if tmp_path is not None and tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
                raise

        print(f"Saved results to {save_path}")
    
if __name__ == "__main__":
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, required=False, default=0.1)
    parser.add_argument("--rtol", type=float, required=False, default=0.1)
    parser.add_argument("--token_size", type=int, required=False)
    parser.add_argument("--max_kl_div", type=float, required=False, default=None)
    parser.add_argument("--golden_logits_path", type=str, required=False, default="")
    parser.add_argument("--hf_model_path", type=str, required=False, default="")
    parser.add_argument("--run_hf_model", type=bool, required=False, default=False)
    parser.add_argument('--add_special_tokens', type=str2bool, default=False)
    parser.add_argument("--limit", type=int, default=1000000)
    parser.add_argument("--tasks", type=lambda x: [] if not x else x.split(","), default=[])
    parser.add_argument("--eval_save_dir", type=str, required=False, default="")
    parser.add_argument("--ppl_batch_size", type=int, default=1, help="Batch size for PPL evaluation (default: 1)")
    parser.add_argument("--acc_batch_size", type=int, default=32, help="Batch size for accuracy evaluation (default: 32)")
    parser.add_argument("--ppl_seq_length", type=int, default=None, help="Override the context length used for PPL/model init")
    parser.add_argument("--acc_seq_length", type=int, default=None, help="Override the context length used for accuracy evaluation")
    parser.add_argument("--acc_limit", type=int, default=None, help="Limit number of evaluation examples per accuracy task")
    parser.add_argument("--acc_task_limits", type=str, default="", help="Comma separated overrides like 'mmlu:10,arc_easy:50'")
    parser.add_argument("--acc_task_seq_lens", type=str, default="", help="Per-task accuracy sequence lengths, e.g. 'piqa:2304,arc_easy:3000'")
    parser.add_argument("--acc_task_batch_sizes", type=str, default="", help="Per-task accuracy batch sizes, e.g. 'piqa:4,arc_easy:8'")
    parser.add_argument("--eval_mode", type=str, choices=["ppl", "acc", "all"], default="all", help="Evaluation mode: 'ppl' (perplexity only), 'acc' (accuracy only), or 'all' (both)")
    parser.add_argument("--apply_chat_template", type=str2bool, default=False, help="Apply chat template for ACC evaluation (use True for SFT models, False for pretrained)")
    parser.add_argument("--resume", type=str2bool, default=False, help="Resume from existing JSON results file, skipping already completed tasks")
    test_args, _ = parser.parse_known_args()

    # Remove args defined in this test file to avoid error from pyconfig
    model_args = sys.argv
    to_remove_args = [
        "--atol",
        "--rtol",
        "--token_size",
        "--max_kl_div",
        "--golden_logits_path",
        "--hf_model_path",
        "--run_hf_model",
        "--add_special_tokens",
        "--limit",
        "--tasks",
        "--save_dir",
        "--eval_save_dir",
        "--ppl_batch_size",
        "--acc_batch_size",
        "--ppl_seq_length",
        "--acc_seq_length",
        "--acc_limit",
        "--acc_task_limits",
        "--acc_task_seq_lens",
        "--acc_task_batch_sizes",
        "--eval_mode",
        "--apply_chat_template",
        "--resume",
    ]
    for arg in to_remove_args:
        model_args = [s for s in model_args if not s.startswith(arg)]

    cfg = pyconfig.initialize(model_args)
    main(cfg, test_args)
