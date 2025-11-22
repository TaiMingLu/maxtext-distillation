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
| v4-64    | 32    | 32                | 32               |
| v4-128   | 64    | 64                | 64               |

**Current configuration (for v4-64):**
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

### Attempt 3: Added checkpoint format flags (PENDING)
**Configuration:**
```bash
checkpoint_storage_use_ocdbt=False
checkpoint_storage_use_zarr3=False
```
Added to maxengine_server command to match the format the checkpoint was saved with.
**Outcome:** Awaiting test
