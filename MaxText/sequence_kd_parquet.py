"""Simple parquet-based sequence KD data generator.

Processes parquet files one by one, generates completions, and saves to JSONL.
Each output line contains: {parquet_file, row_idx, prefix, generated}

Example:
  python3 -m MaxText.sequence_kd_parquet \
    --input-dir /path/to/parquets \
    --output-dir /path/to/output \
    --tokenizer-path /path/to/tokenizer \
    --jetstream-server-port 9000
"""

import argparse
import asyncio
import grpc
import json
import os
import glob
import shutil
import random
from dataclasses import dataclass
from typing import List, Set

import pandas as pd
import transformers

from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

_GRPC_KEEPALIVE_TIMEOUT_MS = 10000
_GRPC_MAX_ATTEMPTS = 5


@dataclass
class Request:
    parquet_file: str
    row_idx: int
    prefix_text: str
    prompt_token_ids: List[int]
    max_output_tokens: int


async def send_request(request, stub, tokenizer, progress_bar, config):
    """Sends request to JetStream server."""
    decode_request = jetstream_pb2.DecodeRequest(
        token_content=jetstream_pb2.DecodeRequest.TokenContent(token_ids=request.prompt_token_ids),
        max_tokens=request.max_output_tokens,
        num_samples=1,
        has_bos=True,
    )

    response = stub.Decode(decode_request)
    completion_tokens = []
    async for resp in response:
        for sample in resp.stream_content.samples:
            completion_tokens.extend(sample.token_ids)

    generated = tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()
    progress_bar.update(1)

    return {
        "parquet_file": os.path.basename(request.parquet_file),
        "row_idx": request.row_idx,
        "prefix": request.prefix_text,
        "generated": generated,
    }


async def run_inference(requests, tokenizer, config):
    """Runs async inference on JetStream server."""
    if not requests:
        return []

    progress_bar = async_tqdm(total=len(requests), desc="Generating")

    server_url = f"localhost:{config.jetstream_server_port}"
    options = [
        ("grpc.keepalive_timeout_ms", _GRPC_KEEPALIVE_TIMEOUT_MS),
        ("grpc.enable_retries", 1),
        ("grpc.service_config", json.dumps({
            "methodConfig": [{
                "name": [{}],
                "retryPolicy": {
                    "maxAttempts": _GRPC_MAX_ATTEMPTS,
                    "initialBackoff": "0.2s",
                    "maxBackoff": "1s",
                    "backoffMultiplier": 2,
                    "retryableStatusCodes": ["UNAVAILABLE"],
                },
            }]
        })),
    ]

    tasks = []
    async with grpc.aio.insecure_channel(server_url, options=options) as channel:
        stub = jetstream_pb2_grpc.OrchestratorStub(channel)
        for request in requests:
            tasks.append(
                asyncio.create_task(
                    send_request(request, stub, tokenizer, progress_bar, config)
                )
            )
        results = await asyncio.gather(*tasks)

    progress_bar.close()
    return results


def get_completed_from_bucket(gcs_bucket_path) -> Set[str]:
    """Check bucket for completed jsonl files."""
    if not gcs_bucket_path or not os.path.exists(gcs_bucket_path):
        return set()

    completed = set()
    for f in os.listdir(gcs_bucket_path):
        if f.endswith(".jsonl"):
            # Convert jsonl name back to parquet name
            parquet_name = f.replace(".jsonl", ".parquet")
            completed.add(parquet_name)
    return completed


def is_completed_in_bucket(gcs_bucket_path, parquet_filename) -> bool:
    """Check if specific file is completed in bucket."""
    if not gcs_bucket_path:
        return False
    jsonl_name = parquet_filename.replace(".parquet", ".jsonl")
    jsonl_path = os.path.join(gcs_bucket_path, jsonl_name)
    return os.path.exists(jsonl_path)


def process_parquet_file(parquet_path, tokenizer, config):
    """Process a single parquet file and return requests."""
    print(f"Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if config.text_column not in df.columns:
        raise ValueError(f"Column '{config.text_column}' not found. Available: {list(df.columns)}")

    requests = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building requests"):
        text = row[config.text_column]
        if not isinstance(text, str) or not text.strip():
            continue

        # Tokenize and truncate to prefix length
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            continue

        if len(tokens) > config.max_prefill_length:
            tokens = tokens[:config.max_prefill_length]

        max_output = config.max_target_length - len(tokens)
        if max_output <= 0:
            continue

        # Get the prefix text (decoded from truncated tokens)
        prefix_text = tokenizer.decode(tokens, skip_special_tokens=True)

        requests.append(Request(
            parquet_file=parquet_path,
            row_idx=idx,
            prefix_text=prefix_text,
            prompt_token_ids=tokens,
            max_output_tokens=max_output,
        ))

    print(f"Built {len(requests)} requests from {len(df)} rows")
    return requests


def main(config):
    # Find all parquet files
    parquet_files = sorted(glob.glob(os.path.join(config.input_dir, "*.parquet")))
    if not parquet_files:
        print(f"No parquet files found in {config.input_dir}")
        return

    print(f"Found {len(parquet_files)} parquet files")

    # Load tokenizer
    print(f"Loading tokenizer from {config.tokenizer_path}")
    if os.path.exists(config.tokenizer_path):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.tokenizer_path, local_files_only=True
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.tokenizer_path, token=config.hf_access_token
        )

    # Check bucket for completed files
    completed_files = get_completed_from_bucket(config.gcs_bucket_path)
    print(f"Already completed in bucket: {len(completed_files)} files")

    # Filter out completed files and shuffle remaining
    remaining_files = [f for f in parquet_files if os.path.basename(f) not in completed_files]
    random.shuffle(remaining_files)
    print(f"Remaining to process: {len(remaining_files)} files (shuffled)")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Process each parquet file in random order
    processed_count = 0
    for parquet_path in remaining_files:
        filename = os.path.basename(parquet_path)

        # Re-check bucket before starting (another instance might have completed it)
        if is_completed_in_bucket(config.gcs_bucket_path, filename):
            print(f"Skipping {filename} (completed by another instance)")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")

        # Build requests
        requests = process_parquet_file(parquet_path, tokenizer, config)

        if not requests:
            print(f"No valid requests for {filename}, marking as complete")
            completed_files.add(filename)
            save_progress(progress_path, completed_files)
            continue

        # Process in batches
        all_results = []
        for i in range(0, len(requests), config.batch_size):
            batch = requests[i:i + config.batch_size]
            print(f"\nBatch {i//config.batch_size + 1}: {len(batch)} requests")

            results = asyncio.run(run_inference(batch, tokenizer, config))
            all_results.extend(results)

        # Save to temp file first
        jsonl_name = filename.replace(".parquet", ".jsonl")
        temp_file = os.path.join(config.output_dir, jsonl_name)

        print(f"Saving {len(all_results)} results to: {temp_file}")
        with open(temp_file, "w") as f:
            for result in all_results:
                f.write(json.dumps(result) + "\n")

        # Copy to bucket (only after complete)
        if config.gcs_bucket_path:
            bucket_file = os.path.join(config.gcs_bucket_path, jsonl_name)
            print(f"Copying to bucket: {bucket_file}")
            shutil.copy(temp_file, bucket_file)

        # Mark as complete
        processed_count += 1
        print(f"Completed {filename} ({processed_count} processed this session)")

    print(f"\nAll done! Processed {processed_count} files this session")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing parquet files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save JSONL outputs")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--hf-access-token", type=str, default=None, help="HF token if needed")
    parser.add_argument("--text-column", type=str, default="text", help="Column name for text")
    parser.add_argument("--max-prefill-length", type=int, default=1024, help="Max prefix tokens")
    parser.add_argument("--max-target-length", type=int, default=4096, help="Max total tokens")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for inference")
    parser.add_argument("--jetstream-server-port", type=int, default=9000, help="JetStream port")
    parser.add_argument("--gcs-bucket-path", type=str, default=None, help="Path to mounted GCS bucket for final output")

    config = parser.parse_args()
    main(config)
