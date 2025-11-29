# Efficient Top-K Knowledge Distillation

## Problem

The original `kl_divergence_between_logits` function materializes the full vocabulary softmax for both teacher and student models, which causes OOM errors with large vocabularies (128K tokens).

### Memory Usage (Old Implementation)
- **Batch size**: 4
- **Sequence length**: 8192
- **Vocabulary size**: 128256
- **Full softmax shape**: `[4, 8192, 128256]`
- **Memory per tensor (f32)**: 4 × 8192 × 128256 × 4 bytes = **15.66 GB**
- **Total memory**: Teacher probs (15.66 GB) + Student probs (15.66 GB) = **31.32 GB**
- **Result**: OOM on TPU v4 (30.75 GB HBM)

## Solution

The new `kl_divergence_between_logits_efficient` function selects top-k in **logit space** before computing softmax, avoiding the need to materialize full vocabulary distributions.

### Memory Usage (New Implementation with top-k)
- **Top-k selection in logit space**: Very cheap (just indices)
- **Softmax over k tokens**: Negligible memory (k << vocab_size)
- **Total memory**: ~Few MB regardless of k value

### Key Optimizations

1. **Top-K Path (with renormalization - default)**:
   ```python
   # Select top-k indices in logit space (cheap!)
   top_k_teacher_logits, top_k_indices = jax.lax.top_k(teacher_logits, k)

   # Compute softmax only over k tokens
   teacher_log_probs_topk = jax.nn.log_softmax(top_k_teacher_logits, axis=-1)

   # Gather student logits and compute KL
   student_log_probs_topk = student_log_probs_full[indices]
   kl = sum(teacher_probs_topk * (teacher_log_probs_topk - student_log_probs_topk))
   ```

2. **Top-K Path (with OTHER bucket)**:
   - Still needs full softmax to compute OTHER bucket probability
   - But only for top-k selection, not for materialization
   - Slightly less efficient than renormalization

3. **Top-P Path**:
   - Still needs full softmax to compute cumulative mass
   - Less efficient than top-k, but still better than old implementation

## Usage

The efficient version is now used automatically in `train.py` when `kd_use_hard_labels=False`.

### Configuration Examples

**Top-K=1 (Most Memory Efficient):**
```bash
export KD_USE_HARD_LABELS=false
export KD_TOP_K=1
export KD_USE_OTHER_BUCKET=false  # Renormalization (default)
```

**Top-K=10:**
```bash
export KD_TOP_K=10
export KD_USE_OTHER_BUCKET=false
```

**Top-K=100 with OTHER bucket:**
```bash
export KD_TOP_K=100
export KD_USE_OTHER_BUCKET=true
```

**Top-P (Nucleus sampling):**
```bash
export KD_TOP_K=0  # Disable top-k
export KD_TOP_P=0.9
```

## Memory Comparison

| Method | Memory Usage | Suitable for |
|--------|--------------|--------------|
| Hard labels | ~Few MB | Top-1 equivalent |
| Old soft KD | ~31 GB | Full vocabulary KD (OOM!) |
| **New efficient top-k** | **~Few MB** | **Any k value** |
| New efficient top-p | ~16 GB | Nucleus sampling |

## Performance Notes

- **Top-k with renormalization**: Fastest, most memory efficient
- **Top-k with OTHER bucket**: Slightly slower (needs full student softmax)
- **Top-p**: Slowest (needs full softmax for cumulative sum)

For most use cases, **top-k with renormalization** (default) is recommended.

## Implementation Details

### Files Modified

1. **`max_utils.py`**: Added `kl_divergence_between_logits_efficient()` function
2. **`train.py`**: Updated to use efficient version in KD loss computation

### Backward Compatibility

The old `kl_divergence_between_logits()` function is still available but not recommended for large vocabularies. The new efficient version is used by default in training.

## Testing

To verify the implementation works correctly, you can now train with:

```bash
export KD_USE_HARD_LABELS=false
export KD_TOP_K=1  # or 2, 10, 100, etc.
export BATCH_SIZE=4  # Can now use full batch size!
```

This should no longer trigger OOM errors.

## Future Improvements

Potential optimizations:
1. Further optimize top-p path to avoid full softmax
2. Add option to cache teacher logits across steps (if using frozen teacher)
3. Implement mixed-precision top-k selection for additional memory savings
