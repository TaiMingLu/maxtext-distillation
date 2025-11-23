# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This directory contains scripts for generating sequence-level knowledge distillation (KD) data using a teacher model. The scripts spawn a MaxEngine inference server and use it to generate continuations from a teacher model, saving the results for student model training.

## Scripts

### get_sequence_kd_data.sh

Entry point that runs via multihost_runner. Launches the remote runner across TPU hosts.

**Required Environment Variables:**
- `TPU_PREFIX`: TPU VM name prefix
- `BUCKET_NAME`: GCS bucket for output data
- `HF_ACCESS_TOKEN`: HuggingFace token for dataset access

**Usage:**
```bash
export TPU_PREFIX=my-tpu
export BUCKET_NAME=my-bucket
export HF_ACCESS_TOKEN=hf_xxx
bash get_sequence_kd_data.sh
```

### remote_sequence_kd_runner.sh

Runs on TPU worker 0. Starts a MaxEngine server with the teacher model, then runs `MaxText.sequence_KD_data` to generate distillation data.

**Configuration (hardcoded in script):**
- Teacher model: `llama3.1-1b`
- Dataset: `HuggingFaceFW/fineweb-edu` (sample-350BT split)
- Max prefill length: 256 tokens
- Max target length: 4096 tokens
- Generation batch size: 512
- Sampling: greedy

**Key Paths:**
- Progress tracking: `/home/terry/gcs-bucket/sequence_kd_progress/{run_name}.json`
- Server log: `/tmp/sequence-kd/server.log`
- Output: `gs://{BUCKET_NAME}/sequence_kd_data/{run_name}/`

### full_loop_single_v6eu.sh

**Single-worker script for TPU v6e-8** (europe-west4). No multihost coordination needed.

**TPU v6e Environment Setup:**
```bash
export PJRT_DEVICE=TPU
unset JAX_COORDINATOR_ADDRESS
export JAX_PROCESS_COUNT=1
export JAX_LOCAL_DEVICE_COUNT=8
```

**Configuration:**
- Teacher model: `llama3.1-1b`
- Dataset: Local parquet files at `/mnt/ramdisk400/finewebedu/sample/100BT`
- Max prefill length: 1024 tokens
- Max target length: 4096 tokens
- Generation batch size: 128
- Server per-device batch: 16
- ICI parallelism: 1×4×2×1=8 (matches v6e-8 devices)

**Output:**
- JSONL files saved to `/home/terry/gcs-bucket/sequence_kd_data/finewebedu/sample-100BT/T50BS42/`

**Usage:**
```bash
export HF_ACCESS_TOKEN=hf_xxx
bash full_loop_single_v6eu.sh
```

### sequence_kd_parquet.py

**Parquet-based data generator** with preemption resilience and distributed processing support.

**Key Features:**
1. Processes parquet files one at a time
2. Saves progress in row-based chunks (batch-size independent)
3. Uses pyarrow metadata for instant row counting (no full parquet load)
4. Supports multiple TPU instances working in parallel without overlap
5. Automatically skips completed chunks

**Output Format (JSONL):**
```json
{"parquet_file": "006_00009.parquet", "row_idx": 0, "prefix": "...", "generated": "..."}
```

**Chunk Naming Scheme:**
```
{parquet_name}_rows_{start:07d}_{end:07d}.jsonl
Example: 006_00009_rows_0000000_0005120.jsonl
```

**Command Line Arguments:**
```bash
python3 -m MaxText.sequence_kd_parquet \
  --input-dir /path/to/parquets \
  --output-dir /tmp/output \
  --tokenizer-path /path/to/tokenizer \
  --gcs-bucket-path /path/to/bucket/output \
  --batch-size 512 \
  --save-every-n-batches 10  # Chunk size = 512 × 10 = 5120 rows
```

**Distributed Processing:**
- Each parquet file is shuffled randomly
- Missing row ranges are also shuffled
- Before processing a chunk, re-checks if another instance completed it
- Multiple instances can process the same parquet file without overlap
- Uses fixed chunk boundaries (0, chunk_size, 2*chunk_size, ...)
- If batch size changes between runs, partial chunks may be less efficient (fewer requests per batch), but no data loss or redundant processing

## Data Generation Pipeline

1. `get_sequence_kd_data.sh` invokes `multihost_runner_orig.py` to distribute work
2. On worker 0, `remote_sequence_kd_runner.sh`:
   - Starts `MaxText.maxengine_server` with teacher model checkpoint
   - Waits for server to be ready (up to 15 minutes)
   - Runs `MaxText.sequence_KD_data` to:
     - Stream dataset from HuggingFace
     - Generate teacher completions via JetStream
     - Upload array records to GCS

## Customization

To modify data generation settings, edit the variables at the top of `remote_sequence_kd_runner.sh`:
- `TEACHER_MODEL_NAME` / `TEACHER_PARAMETERS_PATH`: Change teacher model
- `MAX_PREFILL_LENGTH` / `MAX_TARGET_LENGTH`: Adjust sequence lengths
- `GEN_BATCH_SIZE`: Generation throughput (memory trade-off)
- `DECODE_SAMPLING_STRATEGY`: "greedy" or sampling parameters

## ICI Parallelism Configuration

The `ENGINE_PARALLEL_FLAGS` must have a product equal to the number of devices per slice:

| TPU Type | Chips | Devices per Slice | Required Product |
|----------|-------|-------------------|------------------|
| v6e-8    | 8     | 8                 | 8                |
| v4-64    | 32    | 32                | 32               |
| v4-128   | 64    | 64                | 64               |

**Configuration for v6e-8:**
```bash
ENGINE_PARALLEL_FLAGS=(ici_data_parallelism=1 ici_tensor_parallelism=4 ici_fsdp_parallelism=2 ici_autoregressive_parallelism=1)
# Product: 1 × 4 × 2 × 1 = 8
```

**Configuration for v4-64:**
```bash
ENGINE_PARALLEL_FLAGS=(ici_data_parallelism=2 ici_tensor_parallelism=4 ici_fsdp_parallelism=4 ici_autoregressive_parallelism=1)
# Product: 2 × 4 × 4 × 1 = 32
```

## Troubleshooting Log

### Attempt 1: Original configuration (FAILED)
**Configuration:**
```bash
ENGINE_PARALLEL_FLAGS=(ici_data_parallelism=4 ici_tensor_parallelism=4 ici_fsdp_parallelism=4 ici_autoregressive_parallelism=1)
```
**TPU:** v4-64 (32 devices per slice)
**Error:**
```
AssertionError: Number of devices per slice 32 does not match the product of the ICI parallelism 64
```
**Cause:** Product 4×4×4×1=64 doesn't match 32 devices

### Attempt 2: Adjusted ICI parallelism (FAILED)
**Configuration:**
```bash
ENGINE_PARALLEL_FLAGS=(ici_data_parallelism=2 ici_tensor_parallelism=4 ici_fsdp_parallelism=4 ici_autoregressive_parallelism=1)
```
**TPU:** v4-64 (32 devices per slice)
**Product:** 2×4×4×1=32 ✓
**Error:**
```
ValueError: NOT_FOUND: Error opening "zarr3" driver: Error reading "params.params.decoder.layers.pre_self_attention_layer_norm.scale/zarr.json" in OCDBT database
```
**Cause:** Checkpoint format mismatch - checkpoint was saved with `use_ocdbt=False use_zarr3=False` but loader defaults to `True`

### Attempt 3: Set checkpoint format flags to False (FAILED)
**Configuration:**
```bash
checkpoint_storage_use_ocdbt=False
checkpoint_storage_use_zarr3=False
```
**Error:** Same NOT_FOUND error - Orbax auto-detects checkpoint format from metadata and overrides flags
**Cause:** Checkpoint WAS saved with ocdbt/zarr3=True. The actual issue is file access, not format mismatch.

### Attempt 4: Reverted to True, investigate gcsfuse mount (FAILED)
**Configuration:**
```bash
checkpoint_storage_use_ocdbt=True
checkpoint_storage_use_zarr3=True
```
**Debugging result:** gcsfuse mount shows only:
```
_METADATA  _sharding  array_metadatas  commit_success.txt  manifest.ocdbt
```
The `/d/` subdirectory with ocdbt database files is NOT visible via gcsfuse, even though it exists in GCS.
**Cause:** gcsfuse limitation with ocdbt's nested directory structure

### Attempt 5: Use GCS path directly (PARTIAL SUCCESS)
**Configuration:** Changed `TEACHER_PARAMETERS_PATH` from gcsfuse mount path to direct GCS path:
```bash
TEACHER_PARAMETERS_PATH="gs://${BUCKET_NAME}/ckpts/pretrain_param_only/llama3.1-1b_finewebedu_pretrain_shuffled_lr_3e-4_seed_42/checkpoint_24999/0/items"
```
**Result:** Checkpoint loaded successfully!
**New Error:** Splash attention dimension mismatch:
```
TypeError: broadcast_in_dim operand dimension sizes must either be 1, or be equal to their corresponding dimensions in the target broadcast shape; got operand of shape (4096,), target broadcast shape (256, 128)
```
**Cause:** Splash attention (flash on TPU) has dimension requirements incompatible with prefill_length=256 and target_length=4096

### Attempt 6: Use dot_product attention kernel - wrong key name (FAILED)
**Configuration:**
```bash
attention_kernel=dot_product
```
**Error:** `ValueError: Key attention_kernel was passed at the command line but isn't in config.`

### Attempt 7: Use correct attention config key (PARTIAL SUCCESS)
**Configuration:** The correct key is `attention`, not `attention_kernel`:
```bash
attention=dot_product
```
**Result:** Server started and loaded weights successfully!
**New Error:** Tokenizer pad_token_id overflow:
```
OverflowError: out of range integral type conversion attempted
```
at `maxengine.py:1365` trying to set `pad_token_id = -1`
**Cause:** Llama 3.1 tokenizers have no pad_token or unk_token, so code fell back to -1 which is invalid

### Attempt 8: Fix tokenizer pad_token_id fallback (PENDING)
**Fix:** Modified `MaxText/maxengine.py:1360-1368` to use `eos_token_id` as fallback instead of -1:
```python
if tokenizer_model.tokenizer.pad_token_id is None:
    if tokenizer_model.tokenizer.unk_token_id is not None:
        tokenizer_model.tokenizer.pad_token_id = tokenizer_model.tokenizer.unk_token_id
    elif tokenizer_model.tokenizer.eos_token_id is not None:
        tokenizer_model.tokenizer.pad_token_id = tokenizer_model.tokenizer.eos_token_id
    else:
        tokenizer_model.tokenizer.pad_token_id = 0
```
**Note:** You need to push this change to your GitHub repo and re-clone on the TPU, or directly edit maxengine.py on the TPU.
**Outcome:** Awaiting test
