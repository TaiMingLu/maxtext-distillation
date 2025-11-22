# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a fork of EleutherAI's lm-evaluation-harness with a custom Orbax/MaxText model backend for evaluating JAX-based language models trained with MaxText on TPU. The repository enables running standard LLM evaluation benchmarks on Orbax checkpoints without converting to HuggingFace format.

**Custom Integration:**
- `lm_eval/models/orbax_lm.py`: Custom OrbaxLM model class that wraps MaxText JAX models for evaluation
- `scripts/test_orbax_eval.py`: Main evaluation script for running benchmarks on Orbax checkpoints
- `scripts/test_orbax_eval*.sh`: Shell scripts for batch evaluation across multiple checkpoints

## Common Commands

### Installation

```bash
pip install -e .
# For development
pip install -e ".[dev]"
```

### Running Evaluations

**Standard lm-eval usage:**
```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

**Custom Orbax evaluation (TPU):**
```bash
# Requires MaxText in PYTHONPATH
export PYTHONPATH="/path/to/maxtext:$(pwd):$PYTHONPATH"

python scripts/test_orbax_eval.py ../MaxText/configs/base.yml \
    load_parameters_path=gs://bucket/checkpoints/run/items \
    run_name=my_eval_run \
    model_name=llama3.1-8b \
    per_device_batch_size=4 \
    max_target_length=8192 \
    dtype=bfloat16 \
    scan_layers=false \
    --hf_model_path=/path/to/tokenizer \
    --eval_save_dir=/path/to/results \
    --ppl_batch_size=1 \
    --acc_batch_size=1024
```

**List available tasks:**
```bash
lm_eval --tasks list
```

### Testing

```bash
pip install -e ".[testing]"
pytest
```

## Architecture

### Custom Orbax Integration

The `OrbaxLM` class in `lm_eval/models/orbax_lm.py`:
- Extends the base `lm_eval.api.model.LM` class
- Loads MaxText models and Orbax checkpoints
- Handles JAX-to-PyTorch tensor conversion for compatibility with lm-eval
- Uses pjit for efficient sharded inference on TPU

**Evaluation Flow:**
1. Load MaxText config and create JAX model
2. Load Orbax checkpoint into sharded state
3. Load HuggingFace tokenizer for text processing
4. Create OrbaxLM wrapper and run lm-eval evaluator

### Task Configuration

Default tasks in `scripts/test_orbax_eval.py`:
- **Perplexity tasks:** c4, wikitext, finewebedu-test-100M
- **Accuracy tasks:** winogrande (0-shot), arc_easy (0-shot), mmlu (5-shot), sciq (0-shot)

Task configurations use YAML files in `lm_eval/tasks/`.

## Key Files

- `lm_eval/__main__.py`: CLI entry point
- `lm_eval/evaluator.py`: Core evaluation logic
- `lm_eval/api/model.py`: Base model interface
- `lm_eval/models/`: Model implementations (hf, vllm, orbax_lm, etc.)
- `lm_eval/tasks/`: Task configurations organized by benchmark

## Configuration

**Model args for HuggingFace models:**
- `pretrained`: Model path or HuggingFace model ID
- `dtype`: Data type (float16, bfloat16, float32)
- `parallelize`: Enable model parallelism across GPUs
- `peft`: Path to PEFT adapter

**Orbax-specific args (in MaxText config):**
- `load_parameters_path`: Path to Orbax checkpoint items
- `model_name`: MaxText model architecture
- `scan_layers`: Whether to use scan for layers
- `attention`: Attention implementation type

## Multi-GPU/TPU Evaluation

**Data parallel (HuggingFace):**
```bash
accelerate launch -m lm_eval --model hf \
    --tasks lambada_openai \
    --batch_size 16
```

**Model parallel (HuggingFace):**
```bash
lm_eval --model hf \
    --model_args parallelize=True \
    --tasks lambada_openai \
    --batch_size 16
```

**vLLM:**
```bash
lm_eval --model vllm \
    --model_args pretrained=model,tensor_parallel_size=4 \
    --tasks lambada_openai \
    --batch_size auto
```

## Output and Caching

- `--output_path`: Save evaluation results to JSON
- `--log_samples`: Log individual sample predictions
- `--use_cache <DIR>`: Cache results for resuming interrupted runs
- `--predict_only`: Generate predictions without scoring

## Recent Changes and Implementation Notes

### MMLU auxiliary_train Fix
- Changed `lm_eval/tasks/mmlu/default/_default_template_yaml` to use `hails/mmlu_no_train` instead of `cais/mmlu`
- This avoids the `auxiliary_train` split which has ~100k examples and adds 10+ hours to evaluation
- Removed the DATASET_SPLIT_EXCLUDES monkey-patching from `test_orbax_eval.py` as it's no longer needed

### OrbaxLM Batching Optimization
- `_loglikelihood_tokens` now batches multiple requests together (default: 32)
- Previously processed 1 request at a time, now ~30x faster
- Controlled via `--acc_batch_size` argument
- Uses **right padding** to preserve correct positional embeddings (left padding breaks positions)

### Tokenization
- `tok_encode` defaults to `add_special_tokens=False` for causal LM evaluation
- This is important if your model wasn't trained with special tokens

### Batch Size Arguments
- `--ppl_batch_size`: Batch size for perplexity evaluation (default: 1)
- `--acc_batch_size`: Batch size for accuracy evaluation (default: 32)
- Reduce if running out of memory, increase for faster evaluation

### Progress and Logging
- Timestamps added to evaluation prints: `[2025-11-22 14:30:45] Currently evaluating ACC task: mmlu`
- Each task prints result immediately: `mmlu ACC: 0.4521 (time: 120.56s)`
- Evaluator now logs each subtask: `Evaluating: mmlu_abstract_algebra (100 samples)`

### Multiple Choice Evaluation
- For tasks like arc_easy, mmlu, each question requires 4 forward passes (one per choice A/B/C/D)
- Log-likelihood scoring: model predicts which answer completion is most likely
- With batching, these 4 requests can be batched together for efficiency

### Numerical Precision Note
- Different batch sizes give slightly different results (~0.1% difference)
- This is expected due to floating point precision (operations are not associative)
- Same batch size = deterministic results
- Different batch size = mathematically equivalent but numerically different due to different reduction order in softmax/matmul
