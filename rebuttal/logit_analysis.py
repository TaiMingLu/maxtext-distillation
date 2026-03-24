"""
Comprehensive logit analysis for mechanism study (Exp 5).

Runs forward passes on held-out data for all models (baseline, teachers,
distilled students), collects logit statistics, and saves raw metrics
for later plotting.

Usage (on TPU):
    cd ~/maxtext
    export PYTHONPATH="$(pwd):$(pwd)/lm-evaluation-harness:$PYTHONPATH"
    python3.10 -u rebuttal/logit_analysis.py \
        --output_dir /home/terry/gcs-bucket/rebuttal/exp5_mechanism \
        --data_path /home/terry/gcs-bucket/rebuttal/data/fineweb-edu \
        --tokenizer_path /home/terry/gcs-bucket/rebuttal/hf_models/Llama-3.1-8B \
        --num_sequences 200 \
        --seq_length 2048

The script produces:
    <output_dir>/
        raw/                    # Per-model raw statistics (numpy)
            baseline_1b.npz
            teacher_05b.npz
            ...
        pairwise/               # Pairwise metrics between model pairs
            baseline_1b__distilled_A05B_a02.npz
            ...
        summary.json            # Aggregate statistics for all models
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from functools import partial

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from flax.linen import partitioning as nn_partitioning
from jax.experimental.pjit import pjit
from transformers import AutoTokenizer

# MaxText imports
from MaxText import maxtext_utils, pyconfig
from MaxText.layers import models, quantizations

# ---------------------------------------------------------------------------
# Model registry: all models to analyze
# ---------------------------------------------------------------------------
BUCKET_PREFIX = "/home/terry/gcs-bucket/rebuttal"

MODELS = {
    # Baselines (no KD)
    "baseline_1b": {
        "ckpt": f"{BUCKET_PREFIX}/baselines/llama3.1-1b-s43/24999/items",
        "model_name": "llama3.1-1b",
        "role": "baseline",
    },
    # Teachers (50B tokens, seed 42)
    "teacher_05b": {
        "ckpt": f"{BUCKET_PREFIX}/teachers/llama05b-50B-s42/checkpoint_24999/0/items",
        "model_name": "llama3.1-05b",
        "role": "teacher",
    },
    "teacher_1b": {
        "ckpt": f"{BUCKET_PREFIX}/teachers/llama1b-50B-s42/checkpoint_24999/0/items",
        "model_name": "llama3.1-1b",
        "role": "teacher",
    },
    "teacher_3b": {
        "ckpt": f"{BUCKET_PREFIX}/teachers/llama3b-50B-s42/checkpoint_24999/0/items",
        "model_name": "llama3.1-3b",
        "role": "teacher",
    },
    "teacher_8b": {
        "ckpt": f"{BUCKET_PREFIX}/teachers/llama8b-50B-s42/checkpoint_24999/0/items",
        "model_name": "llama3.1-8b",
        "role": "teacher",
    },
    # Distilled students — best alpha per teacher
    "distilled_A05B_a02": {
        "ckpt": f"{BUCKET_PREFIX}/distilled/A05B-a02/24999/0/items",
        "model_name": "llama3.1-1b",
        "role": "distilled",
        "teacher": "teacher_05b",
        "alpha": 0.2,
    },
    "distilled_A1B_a04": {
        "ckpt": f"{BUCKET_PREFIX}/distilled/A1B-a04/24999/0/items",
        "model_name": "llama3.1-1b",
        "role": "distilled",
        "teacher": "teacher_1b",
        "alpha": 0.4,
    },
    "distilled_A3B_a06": {
        "ckpt": f"{BUCKET_PREFIX}/distilled/A3B-a06/24999/0/items",
        "model_name": "llama3.1-1b",
        "role": "distilled",
        "teacher": "teacher_3b",
        "alpha": 0.6,
    },
    "distilled_A8B_a06": {
        "ckpt": f"{BUCKET_PREFIX}/distilled/A8B-a06/24999/0/items",
        "model_name": "llama3.1-1b",
        "role": "distilled",
        "teacher": "teacher_8b",
        "alpha": 0.6,
    },
    # Pure KD (alpha=1.0) — for comparison
    "distilled_A05B_a1": {
        "ckpt": f"{BUCKET_PREFIX}/distilled/A05B-a1/24999/0/items",
        "model_name": "llama3.1-1b",
        "role": "distilled",
        "teacher": "teacher_05b",
        "alpha": 1.0,
    },
    "distilled_A1B_a1": {
        "ckpt": f"{BUCKET_PREFIX}/distilled/A1B-a1/24999/0/items",
        "model_name": "llama3.1-1b",
        "role": "distilled",
        "teacher": "teacher_1b",
        "alpha": 1.0,
    },
    "distilled_A3B_a1": {
        "ckpt": f"{BUCKET_PREFIX}/distilled/A3B-a1/24999/0/items",
        "model_name": "llama3.1-1b",
        "role": "distilled",
        "teacher": "teacher_3b",
        "alpha": 1.0,
    },
    "distilled_A8B_a1": {
        "ckpt": f"{BUCKET_PREFIX}/distilled/A8B-a1/24999/0/items",
        "model_name": "llama3.1-1b",
        "role": "distilled",
        "teacher": "teacher_8b",
        "alpha": 1.0,
    },
}

# Pairwise comparisons to compute
PAIRWISE = []
for name, info in MODELS.items():
    if info["role"] == "distilled":
        # Compare each distilled student to its teacher and to baseline
        PAIRWISE.append((name, info["teacher"]))
        PAIRWISE.append((name, "baseline_1b"))
# Also compare teachers to baseline
for name, info in MODELS.items():
    if info["role"] == "teacher":
        PAIRWISE.append((name, "baseline_1b"))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_eval_sequences(tokenizer, data_path, num_sequences, seq_length):
    """Load and tokenize evaluation sequences.

    Uses HuggingFace datasets (wikitext) for consistent eval data,
    matching the existing eval pipeline in test_orbax_eval.py.
    data_path is ignored — we use a standard eval corpus instead.
    """
    import datasets as hf_datasets

    print(f"Loading {num_sequences} sequences (seq_len={seq_length}) from wikitext-103...")

    dataset = hf_datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-103-v1", split="train", trust_remote_code=True
    )
    # Concatenate text and tokenize
    raw_text = "\n\n".join(dataset[:32768]["text"])

    print(f"  Tokenizing...")

    all_tokens = tokenizer.encode(raw_text, add_special_tokens=False)

    # Chunk into sequences
    total_tokens = (len(all_tokens) // seq_length) * seq_length
    all_tokens = all_tokens[:total_tokens]
    token_array = np.array(all_tokens, dtype=np.int32).reshape(-1, seq_length)
    token_array = token_array[:num_sequences]

    print(f"  Prepared {token_array.shape[0]} sequences of length {seq_length}")
    return token_array


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_name, ckpt_path, seq_length):
    """Load a MaxText model and checkpoint. Returns (forward_fn, params, mesh)."""
    print(f"  Loading model {model_name} from {ckpt_path}")

    # Create config
    config_args = [
        "MaxText/configs/base.yml",
        f"model_name={model_name}",
        f"load_parameters_path={ckpt_path}",
        f"max_target_length={seq_length}",
        "scan_layers=false",
        "attention=dot_product",
        "dtype=bfloat16",
        "per_device_batch_size=1",
        "run_name=logit_analysis",
        "base_output_directory=gs://taiming_us_central1/rebuttal/exp5_mechanism",
    ]
    pyconfig.initialize(config_args)
    config = pyconfig.config

    init_rng = jax.random.PRNGKey(0)
    init_rng, rng1 = jax.random.split(init_rng)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
    quant = quantizations.configure_quantization(config)
    model = models.Transformer(config, mesh, quant=quant)
    state, _ = maxtext_utils.setup_decode_state(model, config, rng1, mesh, None)

    # Cast to bf16
    state = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if hasattr(x, 'astype') and x.dtype == jnp.float32 else x,
        state
    )

    # Create fast forward function
    def extract_sharding(x):
        return x.sharding if hasattr(x, 'sharding') else None
    params_shardings = jax.tree_util.tree_map(extract_sharding, state.params)

    @partial(pjit, in_shardings=(params_shardings, None, None, None), out_shardings=None)
    def fast_forward(params, input_ids, positions, segment_ids):
        return model.apply(
            params, input_ids, positions, segment_ids,
            enable_dropout=False,
            rngs={"aqt": jax.random.PRNGKey(0)},
        )

    return fast_forward, state.params, mesh, config


def run_forward(forward_fn, params, mesh, config, token_ids_batch):
    """Run forward pass, return logits as numpy array."""
    batch_size, seq_len = token_ids_batch.shape
    input_ids = jnp.asarray(token_ids_batch, dtype=jnp.int32)
    positions = jnp.tile(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, 1))
    segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        logits = forward_fn(params, input_ids, positions, segment_ids)

    return np.array(logits, dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------
def compute_per_model_stats(logits, token_ids, top_k=100):
    """
    Compute per-token statistics from logits.

    Args:
        logits: [batch, seq_len, vocab_size]
        token_ids: [batch, seq_len] — ground truth tokens
        top_k: number of top logits to save

    Returns dict of arrays.
    """
    batch, seq_len, vocab = logits.shape

    # Shift: logits[t] predicts token[t+1]
    pred_logits = logits[:, :-1, :]      # [batch, seq_len-1, vocab]
    target_ids = token_ids[:, 1:]         # [batch, seq_len-1]

    flat_logits = pred_logits.reshape(-1, vocab)  # [N, vocab]
    flat_targets = target_ids.reshape(-1)          # [N]
    N = flat_logits.shape[0]

    # Log-softmax for stable computation
    log_probs = flat_logits - np.log(np.sum(np.exp(flat_logits - flat_logits.max(axis=-1, keepdims=True)), axis=-1, keepdims=True)) - flat_logits.max(axis=-1, keepdims=True)
    # Simpler: use scipy-style log_softmax
    max_logits = flat_logits.max(axis=-1, keepdims=True)
    shifted = flat_logits - max_logits
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    probs = np.exp(log_probs)

    # 1. Entropy: H = -sum(p * log(p))
    entropy = -np.sum(probs * log_probs, axis=-1)  # [N]

    # 2. Max probability (confidence)
    max_prob = np.max(probs, axis=-1)  # [N]

    # 3. Ground truth log-likelihood
    gt_log_prob = log_probs[np.arange(N), flat_targets]  # [N]

    # 4. Rank of ground truth token
    # rank = number of tokens with higher probability + 1
    gt_probs = probs[np.arange(N), flat_targets]  # [N]
    gt_rank = (probs > gt_probs[:, None]).sum(axis=-1) + 1  # [N]

    # 5. Top-k logits and token IDs
    top_k_indices = np.argpartition(flat_logits, -top_k, axis=-1)[:, -top_k:]
    # Sort by logit value within top-k
    top_k_logits_unsorted = np.take_along_axis(flat_logits, top_k_indices, axis=-1)
    sort_order = np.argsort(-top_k_logits_unsorted, axis=-1)
    top_k_indices = np.take_along_axis(top_k_indices, sort_order, axis=-1)
    top_k_logits = np.take_along_axis(flat_logits, top_k_indices, axis=-1)
    top_k_probs = np.take_along_axis(probs, top_k_indices, axis=-1)

    # 6. Top-k cumulative probability mass
    top_k_cumprob = np.cumsum(top_k_probs, axis=-1)

    # 7. Effective vocabulary size (number of tokens with prob > 1/vocab)
    threshold = 1.0 / vocab
    effective_vocab = (probs > threshold).sum(axis=-1)  # [N]

    return {
        "entropy": entropy.astype(np.float32),
        "max_prob": max_prob.astype(np.float32),
        "gt_log_prob": gt_log_prob.astype(np.float32),
        "gt_rank": gt_rank.astype(np.int32),
        "effective_vocab": effective_vocab.astype(np.int32),
        "top_k_indices": top_k_indices.astype(np.int32),
        "top_k_logits": top_k_logits.astype(np.float32),
        "top_k_probs": top_k_probs.astype(np.float32),
        "top_k_cumprob_at_10": top_k_cumprob[:, 9].astype(np.float32) if top_k >= 10 else None,
        "top_k_cumprob_at_50": top_k_cumprob[:, 49].astype(np.float32) if top_k >= 50 else None,
    }


# ---------------------------------------------------------------------------
# Pairwise analysis
# ---------------------------------------------------------------------------
def compute_pairwise_stats(logits_a, logits_b, token_ids):
    """
    Compute pairwise statistics between two models.

    Args:
        logits_a, logits_b: [batch, seq_len, vocab_size]
        token_ids: [batch, seq_len]

    Returns dict of arrays.
    """
    batch, seq_len, vocab = logits_a.shape

    # Shift
    la = logits_a[:, :-1, :].reshape(-1, vocab)
    lb = logits_b[:, :-1, :].reshape(-1, vocab)
    N = la.shape[0]

    # Probabilities
    def to_probs(logits):
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_p = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        return np.exp(log_p), log_p

    probs_a, log_probs_a = to_probs(la)
    probs_b, log_probs_b = to_probs(lb)

    # 1. KL(A || B) per token = sum_v p_A(v) * (log p_A(v) - log p_B(v))
    kl_a_to_b = np.sum(probs_a * (log_probs_a - log_probs_b), axis=-1)  # [N]

    # 2. KL(B || A) per token
    kl_b_to_a = np.sum(probs_b * (log_probs_b - log_probs_a), axis=-1)  # [N]

    # 3. JS divergence = 0.5 * KL(A||M) + 0.5 * KL(B||M) where M = 0.5*(A+B)
    m_probs = 0.5 * (probs_a + probs_b)
    log_m = np.log(m_probs + 1e-10)
    js_div = 0.5 * np.sum(probs_a * (log_probs_a - log_m), axis=-1) + \
             0.5 * np.sum(probs_b * (log_probs_b - log_m), axis=-1)

    # 4. Top-1 agreement
    top1_a = np.argmax(la, axis=-1)
    top1_b = np.argmax(lb, axis=-1)
    top1_agree = (top1_a == top1_b).astype(np.float32)

    # 5. Top-5 overlap (Jaccard of top-5 sets)
    top5_a = np.argpartition(la, -5, axis=-1)[:, -5:]
    top5_b = np.argpartition(lb, -5, axis=-1)[:, -5:]
    top5_overlap = np.array([
        len(set(top5_a[i].tolist()) & set(top5_b[i].tolist())) / 5.0
        for i in range(N)
    ], dtype=np.float32)

    # 6. Logit cosine similarity
    dot = np.sum(la * lb, axis=-1)
    norm_a = np.sqrt(np.sum(la ** 2, axis=-1))
    norm_b = np.sqrt(np.sum(lb ** 2, axis=-1))
    cosine_sim = dot / (norm_a * norm_b + 1e-10)

    # 7. Rank correlation on top-10 (per-token Spearman on top-10 logit positions)
    top10_a = np.argpartition(la, -10, axis=-1)[:, -10:]
    top10_union = top10_a  # Use model A's top-10 as reference
    rank_a = np.argsort(np.argsort(-np.take_along_axis(la, top10_union, axis=-1), axis=-1), axis=-1).astype(np.float32)
    rank_b = np.argsort(np.argsort(-np.take_along_axis(lb, top10_union, axis=-1), axis=-1), axis=-1).astype(np.float32)
    # Spearman = 1 - 6*sum(d^2) / (n*(n^2-1))
    d_sq = (rank_a - rank_b) ** 2
    n = 10
    spearman = 1.0 - 6.0 * d_sq.sum(axis=-1) / (n * (n**2 - 1))

    # 8. Probability-weighted agreement: sum of min(p_A, p_B) — how much distributional mass overlaps
    prob_overlap = np.sum(np.minimum(probs_a, probs_b), axis=-1)

    return {
        "kl_a_to_b": kl_a_to_b.astype(np.float32),
        "kl_b_to_a": kl_b_to_a.astype(np.float32),
        "js_divergence": js_div.astype(np.float32),
        "top1_agreement": top1_agree,
        "top5_jaccard": top5_overlap,
        "cosine_similarity": cosine_sim.astype(np.float32),
        "top10_spearman": spearman.astype(np.float32),
        "prob_overlap": prob_overlap.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Token difficulty binned analysis
# ---------------------------------------------------------------------------
def compute_binned_analysis(model_stats, baseline_stats, num_bins=5):
    """
    Bin tokens by baseline difficulty (entropy) and compute
    per-bin improvement from KD.
    """
    base_entropy = baseline_stats["entropy"]
    bins = np.quantile(base_entropy, np.linspace(0, 1, num_bins + 1))
    bin_indices = np.digitize(base_entropy, bins[1:-1])  # 0 to num_bins-1

    results = {}
    for b in range(num_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        label = f"bin_{b}"
        results[label] = {
            "count": int(mask.sum()),
            "baseline_entropy_mean": float(base_entropy[mask].mean()),
            "baseline_gt_logprob_mean": float(baseline_stats["gt_log_prob"][mask].mean()),
            "model_gt_logprob_mean": float(model_stats["gt_log_prob"][mask].mean()),
            "improvement": float(model_stats["gt_log_prob"][mask].mean() - baseline_stats["gt_log_prob"][mask].mean()),
            "model_entropy_mean": float(model_stats["entropy"][mask].mean()),
            "entropy_change": float(model_stats["entropy"][mask].mean() - base_entropy[mask].mean()),
        }
    return results


# ---------------------------------------------------------------------------
# Position-based analysis
# ---------------------------------------------------------------------------
def compute_position_analysis(model_stats, baseline_stats, seq_length, num_positions=20):
    """Analyze how KD improvement varies by position in the sequence."""
    N = len(model_stats["gt_log_prob"])
    positions = np.arange(N) % (seq_length - 1)  # position within each sequence

    # Bucket positions into groups
    bucket_size = (seq_length - 1) // num_positions
    bucket_ids = positions // max(bucket_size, 1)
    bucket_ids = np.clip(bucket_ids, 0, num_positions - 1)

    results = {}
    for b in range(num_positions):
        mask = bucket_ids == b
        if mask.sum() == 0:
            continue
        pos_start = b * bucket_size
        pos_end = min((b + 1) * bucket_size, seq_length - 1)
        results[f"pos_{pos_start}_{pos_end}"] = {
            "count": int(mask.sum()),
            "baseline_gt_logprob": float(baseline_stats["gt_log_prob"][mask].mean()),
            "model_gt_logprob": float(model_stats["gt_log_prob"][mask].mean()),
            "improvement": float(model_stats["gt_log_prob"][mask].mean() - baseline_stats["gt_log_prob"][mask].mean()),
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Logit analysis for mechanism study")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--num_sequences", type=int, default=200)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated model names, or 'all'")
    parser.add_argument("--skip_pairwise", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    pairwise_dir = output_dir / "pairwise"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pairwise_dir.mkdir(parents=True, exist_ok=True)

    # Select models
    if args.models == "all":
        model_names = list(MODELS.keys())
    else:
        model_names = [m.strip() for m in args.models.split(",")]

    print(f"Analyzing {len(model_names)} models on {args.num_sequences} sequences")
    print(f"Output: {output_dir}")

    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    token_ids = load_eval_sequences(tokenizer, args.data_path, args.num_sequences, args.seq_length)

    # Phase 1: Per-model forward passes and statistics
    all_stats = {}
    all_logits_cache = {}  # Cache logits for pairwise analysis (only for 1.7B models to save memory)

    for model_key in model_names:
        info = MODELS[model_key]
        raw_path = raw_dir / f"{model_key}.npz"

        # Skip if already computed
        if raw_path.exists():
            print(f"\n[{model_key}] Already computed, loading from {raw_path}")
            data = np.load(raw_path, allow_pickle=True)
            all_stats[model_key] = {k: data[k] for k in data.files}
            continue

        print(f"\n{'='*60}")
        print(f"[{model_key}] Loading model ({info['model_name']})...")
        t0 = time.time()

        forward_fn, params, mesh, config = load_model(
            info["model_name"], info["ckpt"], args.seq_length
        )

        print(f"[{model_key}] Model loaded in {time.time()-t0:.1f}s. Running forward passes...")

        # Collect logits in batches
        all_logits = []
        num_batches = (len(token_ids) + args.batch_size - 1) // args.batch_size
        for bi in range(num_batches):
            batch = token_ids[bi * args.batch_size : (bi + 1) * args.batch_size]
            logits = run_forward(forward_fn, params, mesh, config, batch)
            all_logits.append(logits)
            if (bi + 1) % 10 == 0:
                print(f"  Batch {bi+1}/{num_batches}")

        all_logits = np.concatenate(all_logits, axis=0)
        print(f"[{model_key}] Forward done in {time.time()-t0:.1f}s. Computing stats...")

        # Compute per-model stats
        stats = compute_per_model_stats(all_logits, token_ids, top_k=args.top_k)
        all_stats[model_key] = stats

        # Save raw stats
        np.savez_compressed(raw_path, **stats)
        print(f"[{model_key}] Saved to {raw_path}")

        # Cache logits for pairwise (only 1.7B models — teachers need separate loading)
        if info["model_name"] == "llama3.1-1b":
            all_logits_cache[model_key] = all_logits

        # Free memory
        del forward_fn, params, all_logits
        jax.clear_caches()

    # Phase 2: Pairwise analysis
    if not args.skip_pairwise:
        print(f"\n{'='*60}")
        print(f"Phase 2: Pairwise analysis ({len(PAIRWISE)} pairs)")

        for model_a, model_b in PAIRWISE:
            if model_a not in model_names or model_b not in model_names:
                continue

            pair_path = pairwise_dir / f"{model_a}__{model_b}.npz"
            if pair_path.exists():
                print(f"  [{model_a} vs {model_b}] Already computed, skipping")
                continue

            # Need logits for both models
            if model_a not in all_logits_cache or model_b not in all_logits_cache:
                print(f"  [{model_a} vs {model_b}] Logits not cached, skipping (different arch models)")
                continue

            print(f"  [{model_a} vs {model_b}] Computing pairwise stats...")
            pair_stats = compute_pairwise_stats(
                all_logits_cache[model_a],
                all_logits_cache[model_b],
                token_ids,
            )
            np.savez_compressed(pair_path, **pair_stats)

    # Phase 3: Aggregate summary
    print(f"\n{'='*60}")
    print("Phase 3: Computing aggregate summary...")

    summary = {"models": {}, "binned": {}, "position": {}}
    baseline_stats = all_stats.get("baseline_1b")

    for model_key in model_names:
        stats = all_stats[model_key]
        summary["models"][model_key] = {
            "role": MODELS[model_key]["role"],
            "model_name": MODELS[model_key]["model_name"],
            "entropy_mean": float(np.mean(stats["entropy"])),
            "entropy_std": float(np.std(stats["entropy"])),
            "entropy_median": float(np.median(stats["entropy"])),
            "max_prob_mean": float(np.mean(stats["max_prob"])),
            "gt_logprob_mean": float(np.mean(stats["gt_log_prob"])),
            "gt_logprob_std": float(np.std(stats["gt_log_prob"])),
            "gt_rank_mean": float(np.mean(stats["gt_rank"])),
            "gt_rank_median": float(np.median(stats["gt_rank"])),
            "effective_vocab_mean": float(np.mean(stats["effective_vocab"])),
            "ppl": float(np.exp(-np.mean(stats["gt_log_prob"]))),
        }
        if stats.get("top_k_cumprob_at_10") is not None:
            summary["models"][model_key]["top10_cumprob_mean"] = float(np.mean(stats["top_k_cumprob_at_10"]))
        if stats.get("top_k_cumprob_at_50") is not None:
            summary["models"][model_key]["top50_cumprob_mean"] = float(np.mean(stats["top_k_cumprob_at_50"]))

        # Binned and position analysis (compare to baseline)
        if baseline_stats is not None and model_key != "baseline_1b":
            summary["binned"][model_key] = compute_binned_analysis(
                stats, baseline_stats, num_bins=5
            )
            summary["position"][model_key] = compute_position_analysis(
                stats, baseline_stats, args.seq_length
            )

    # Pairwise summary
    summary["pairwise"] = {}
    for model_a, model_b in PAIRWISE:
        pair_path = pairwise_dir / f"{model_a}__{model_b}.npz"
        if pair_path.exists():
            data = np.load(pair_path)
            key = f"{model_a}_vs_{model_b}"
            summary["pairwise"][key] = {
                "kl_a_to_b_mean": float(np.mean(data["kl_a_to_b"])),
                "kl_b_to_a_mean": float(np.mean(data["kl_b_to_a"])),
                "js_divergence_mean": float(np.mean(data["js_divergence"])),
                "top1_agreement": float(np.mean(data["top1_agreement"])),
                "top5_jaccard_mean": float(np.mean(data["top5_jaccard"])),
                "cosine_similarity_mean": float(np.mean(data["cosine_similarity"])),
                "top10_spearman_mean": float(np.mean(data["top10_spearman"])),
                "prob_overlap_mean": float(np.mean(data["prob_overlap"])),
            }

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Print key findings
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")
    if "baseline_1b" in summary["models"]:
        bl = summary["models"]["baseline_1b"]
        print(f"\nBaseline 1.7B: entropy={bl['entropy_mean']:.3f}, ppl={bl['ppl']:.2f}")

    for model_key in model_names:
        if MODELS[model_key]["role"] == "distilled":
            m = summary["models"][model_key]
            bl = summary["models"].get("baseline_1b", {})
            ppl_change = ((m["ppl"] - bl.get("ppl", 0)) / bl.get("ppl", 1)) * 100
            ent_change = m["entropy_mean"] - bl.get("entropy_mean", 0)
            print(f"  {model_key}: ppl={m['ppl']:.2f} ({ppl_change:+.1f}%), "
                  f"entropy={m['entropy_mean']:.3f} ({ent_change:+.3f})")

    print(f"\nPairwise highlights:")
    for key, vals in summary.get("pairwise", {}).items():
        print(f"  {key}: top1_agree={vals['top1_agreement']:.3f}, "
              f"KL={vals['kl_a_to_b_mean']:.3f}, cos_sim={vals['cosine_similarity_mean']:.3f}")

    print(f"\nDone! All results in {output_dir}")


if __name__ == "__main__":
    main()
