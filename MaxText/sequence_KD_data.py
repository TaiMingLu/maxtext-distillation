#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Sequence-level knowledge distillation data generator.

This variant targets pre-training style corpora rather than conversational
dialogues. It streams a text dataset, tokenizes each sample without adding
special tokens, feeds the resulting prefix into a running MaxText
`maxengine_server`, and captures the teacher's generated continuations. The
entire corpus is processed sequentially and progress is persisted so reruns
resume from the last completed chunk rather than starting from scratch.

Example command:

  python3 -m MaxText.sequence_KD_data \
    --dataset-path HuggingFaceFW/fineweb-edu --data-split sample-350BT --text-column text \
    --tokenizer-path /home/terry/gcs-bucket/HF_HOME/Llama-3.1-8B --hf-access-token <token> \
    --batch-size 1024 --max-prefill-length 256 --max-target-length 4096 \
    --progress-path sequence_kd_state.json --num-generations 1 \
    upload-to-gcs --gcs-bucket my-bucket --gcs-data-path distillation/

Make sure to start a `maxengine_server` process before running this script, for
example:

  python3 -m MaxText.maxengine_server MaxText/configs/base.yml \
    model_name=deepseek2-16b tokenizer_path=deepseek-ai/DeepSeek-V2-Lite-chat \
    tokenizer_type=huggingface load_parameters_path=<ckpt> \
    max_target_length=2048 max_prefill_predict_length=256 \
    per_device_batch_size=10 multi_sampling=True ici_tensor_parallelism=4 \
    decode_sampling_strategy=weighted scan_layers=False
"""

import argparse
import asyncio
import grpc
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import transformers

import datasets
from datasets import Dataset
from huggingface_hub import create_repo, get_full_repo_name, repo_exists, upload_file

from MaxText import max_logging
from MaxText.utils import gcs_utils

from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
from tqdm import tqdm as sync_tqdm
from tqdm.asyncio import tqdm as async_tqdm

_GRPC_KEEPALIVE_TIMEOUT_MS = 10000
_GRPC_MAX_ATTEMPTS = 5


@dataclass
class TextInputRequest:
  """Simple request wrapper for plain-text corpora."""

  text: str
  prompt_token_ids: List[int]
  max_output_tokens: int


async def get_request(input_requests):
  input_requests = iter(input_requests)
  for request in input_requests:
    yield request


async def send_request(config, request, stub, tokenizer, progress_bar):  # pylint: disable=redefined-outer-name
  """Sends the request to JetStream server."""
  decode_request = jetstream_pb2.DecodeRequest(
      token_content=jetstream_pb2.DecodeRequest.TokenContent(token_ids=request.prompt_token_ids),
      max_tokens=request.max_output_tokens,
      num_samples=config.num_generations,  # number of responses to generate for each request
      has_bos=True,
  )

  response = stub.Decode(decode_request)
  completion_tokens = [[] for _ in range(config.num_generations)]
  async for resp in response:
    for idx, sample in enumerate(resp.stream_content.samples):
      resp_tokens = sample.token_ids
      completion_tokens[idx].extend(resp_tokens)

  outputs = []
  for tokens in completion_tokens:
    completion = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    outputs.append(
        {
            "source_text": request.text,
            "teacher_completion": completion,
        }
    )
  progress_bar.update(1)
  return outputs


async def run_inference(config, requests, tokenizer):  # pylint: disable=redefined-outer-name
  """Asynchronously runs inference on JetStream server."""
  progress_bar = async_tqdm(total=len(requests))
  progress_bar.set_description(f"Running inference on {len(requests)} prompts")

  server_url = f"localhost:{config.jetstream_server_port}"
  options = []
  options.append(("grpc.keepalive_timeout_ms", _GRPC_KEEPALIVE_TIMEOUT_MS))
  options.append(("grpc.enable_retries", 1))
  service_config_json = json.dumps(
      {
          "methodConfig": [
              {
                  "name": [{}],
                  "retryPolicy": {
                      "maxAttempts": _GRPC_MAX_ATTEMPTS,
                      "initialBackoff": "0.2s",
                      "maxBackoff": "1s",
                      "backoffMultiplier": 2,
                      "retryableStatusCodes": ["UNAVAILABLE"],
                  },
              }
          ]
      }
  )
  options.append(("grpc.service_config", service_config_json))
  tasks = []
  async with grpc.aio.insecure_channel(server_url, options=options) as channel:
    stub = jetstream_pb2_grpc.OrchestratorStub(channel)
    async for request in get_request(requests):
      tasks.append(
          asyncio.create_task(
              send_request(
                  config=config,
                  request=request,
                  stub=stub,
                  tokenizer=tokenizer,
                  progress_bar=progress_bar,
              )
          )
      )
    outputs = await asyncio.gather(*tasks)
  progress_bar.close()
  return outputs


def generate_completions(config, requests, tokenizer):  # pylint: disable=redefined-outer-name
  """Generates num_generations of completions for each prompt in request."""
  if not requests:
    return []
  outputs = asyncio.run(
      run_inference(
          config=config,
          requests=requests,
          tokenizer=tokenizer,
      ),
  )
  return [output for output_per_prompt_list in outputs for output in output_per_prompt_list]


def upload_data_to_hf(config, parquet_file_name, batch_num):  # pylint: disable=redefined-outer-name
  """Upload dataset to Hugging Face."""
  full_repo_name = get_full_repo_name(model_id=config.hf_repo_id, token=config.hf_access_token)
  if not repo_exists(repo_id=full_repo_name, repo_type="dataset", token=config.hf_access_token):
    max_logging.log("Repository doesn't exist on Hugging Face, creating a new one.")
    try:
      repo_url = create_repo(repo_id=config.hf_repo_id, repo_type="dataset", private=True, token=config.hf_access_token)
      max_logging.log(f"Successfully created repository on Hugging Face: {repo_url}.")
    except Exception as e:  # pylint: disable=broad-except
      max_logging.log(f"Error in creating repository on Hugging Face: {e}")
      raise e

  max_logging.log(f"Pushing dataset to Hugging Face: https://huggingface.co/datasets/{full_repo_name}")
  try:
    upload_file(
        repo_id=full_repo_name,
        repo_type="dataset",
        path_or_fileobj=parquet_file_name,
        path_in_repo=f"data/{parquet_file_name}",
        commit_message=f"Uploading dataset batch number {batch_num}",
        token=config.hf_access_token,
    )
    max_logging.log(f"Successfully pushed dataset to Hugging Face: https://huggingface.co/datasets/{full_repo_name}")
  except Exception as e:  # pylint: disable=broad-except
    max_logging.log(f"Error in pushing dataset to Hugging Face: {e}")
    raise e


def upload_data_to_gcs(config, source_file_name):  # pylint: disable=redefined-outer-name
  """Uploads dataset to Google Cloud Storage bucket."""
  data_path = gcs_utils.add_trailing_slash(config.gcs_data_path)
  destination_name = f"gs://{config.gcs_bucket}/{data_path}{source_file_name}"
  max_logging.log(f"Pushing dataset to GCS: {destination_name}")
  try:
    gcs_utils.upload_blob(destination_name, source_file_name)
    max_logging.log(f"Successfully pushed dataset to GCS: {destination_name}")
  except FileNotFoundError as e:
    max_logging.log(f"Error in pushing dataset to GCS: '{source_file_name}' not found during upload attempt.")
    raise e
  except Exception as e:
    max_logging.log(f"Error in pushing dataset to GCS: {e}")
    raise e


def upload_data(config, data, batch_num):  # pylint: disable=redefined-outer-name
  """Uploads dataset to Google Cloud Storage or Hugging Face."""
  distillation_dataset = Dataset.from_list(data)
  parquet_file_name = f"sequence-distillation-data-{batch_num}.parquet"
  distillation_dataset.to_parquet(parquet_file_name)
  if config.upload == "upload-to-hf":
    upload_data_to_hf(config, parquet_file_name, batch_num)
  elif config.upload == "upload-to-gcs":
    upload_data_to_gcs(config, parquet_file_name)
  # remove local dataset files after upload
  if config.remove_local_dataset_files and os.path.exists(parquet_file_name):
    try:
      os.remove(parquet_file_name)
    except OSError as e:
      max_logging.log(f"Unable to remove local dataset file {parquet_file_name}: {e}")


def load_text_dataset(config):
  """Loads a Hugging Face text dataset or local parquet files."""
  import os
  import glob

  # Check if dataset_path is a local directory or file pattern
  if os.path.isdir(config.dataset_path):
    # Load all parquet files from directory
    parquet_files = sorted(glob.glob(os.path.join(config.dataset_path, "*.parquet")))
    if not parquet_files:
      raise ValueError(f"No parquet files found in {config.dataset_path}")
    max_logging.log(f"Loading {len(parquet_files)} parquet files from {config.dataset_path}")
    dataset = datasets.load_dataset("parquet", data_files=parquet_files, split="train")
  elif config.dataset_path.endswith(".parquet") or "*" in config.dataset_path:
    # Load from file pattern
    parquet_files = sorted(glob.glob(config.dataset_path))
    if not parquet_files:
      raise ValueError(f"No parquet files found matching {config.dataset_path}")
    max_logging.log(f"Loading {len(parquet_files)} parquet files from pattern {config.dataset_path}")
    dataset = datasets.load_dataset("parquet", data_files=parquet_files, split="train")
  else:
    # Load from Hugging Face Hub
    assert config.dataset_type == "huggingface", "Only Hugging Face datasets are supported."
    dataset = datasets.load_dataset(
        config.dataset_path,
        split=config.data_split,
        token=config.hf_access_token,
    )

  if config.text_column not in dataset.column_names:
    raise ValueError(
        f"Column '{config.text_column}' not found in dataset. Available columns: {dataset.column_names}"
    )
  return dataset


def build_requests(dataset_slice: Dataset, tokenizer, config) -> List[TextInputRequest]:  # pylint: disable=redefined-outer-name
  """Converts dataset rows into TextInputRequest objects."""
  requests = []
  for example in dataset_slice:
    text = example[config.text_column]
    if not isinstance(text, str) or not text.strip():
      continue
    prompt_token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not prompt_token_ids:
      continue
    if len(prompt_token_ids) > config.max_prefill_length:
      prompt_token_ids = prompt_token_ids[: config.max_prefill_length]
    if not prompt_token_ids:
      continue
    max_output_tokens = config.max_target_length - len(prompt_token_ids)
    if max_output_tokens <= 0:
      continue
    requests.append(
        TextInputRequest(
            text=text,
            prompt_token_ids=prompt_token_ids,
            max_output_tokens=max_output_tokens,
        )
    )
  if len(requests) < len(dataset_slice):
    max_logging.log(
        f"Filtered {len(dataset_slice) - len(requests)} samples due to length or empty content."
    )
  return requests


def load_progress(progress_path: str) -> Tuple[int, int, dict]:
  """Returns (next_index, next_batch_num, metadata)."""
  if not os.path.exists(progress_path):
    return 0, 0, {}
  with open(progress_path, "r", encoding="utf-8") as f:
    state = json.load(f)
  return int(state.get("next_index", 0)), int(state.get("next_batch_num", 0)), state


def save_progress(progress_path: str, next_index: int, next_batch_num: int, **metadata) -> None:
  state = {
      "next_index": next_index,
      "next_batch_num": next_batch_num,
  }
  state.update(metadata)
  progress_dir = os.path.dirname(progress_path)
  if progress_dir:
    os.makedirs(progress_dir, exist_ok=True)
  with open(progress_path, "w", encoding="utf-8") as f:
    json.dump(state, f)


def load_tokenizer(config):
  """Loads a tokenizer from a local path or Hugging Face Hub."""
  tokenizer_path = config.tokenizer_path
  if os.path.exists(tokenizer_path):
    max_logging.log(f"Loading tokenizer from local path: {tokenizer_path}")
    return transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )
  max_logging.log(f"Loading tokenizer from Hugging Face Hub: {tokenizer_path}")
  return transformers.AutoTokenizer.from_pretrained(
    tokenizer_path,
    token=config.hf_access_token,
  )


def _now_timestamp() -> str:
  return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_data(config):  # pylint: disable=redefined-outer-name
  """Generates teacher completions for every example in the dataset."""
  dataset = load_text_dataset(config)
  tokenizer = load_tokenizer(config)
  tokenizer.padding_side = "left"

  total_examples = len(dataset)
  if config.max_examples is not None and config.max_examples < total_examples:
    total_examples = config.max_examples
    max_logging.log(f"Limiting processing to {total_examples} examples (--max-examples).")
  start_index, batch_num, progress_state = load_progress(config.progress_path)
  if progress_state:
    last_ts = progress_state.get("last_update_ts", "unknown")
    max_logging.log(
        f"Resuming from index {start_index} (batch {batch_num}); last update: {last_ts}"
    )
  if start_index >= total_examples:
    max_logging.log("Dataset already fully processed; refreshing progress file.")
    save_progress(
        config.progress_path,
        total_examples,
        batch_num,
        status="completed",
        total_examples=total_examples,
        completed_examples=total_examples,
        last_update_ts=_now_timestamp(),
    )
    return

  progress_bar = sync_tqdm(
      total=total_examples,
      initial=start_index,
      unit="sample",
      desc="Sequence KD data",
  )

  while start_index < total_examples:
    batch_start = start_index
    end_index = min(start_index + config.batch_size, total_examples)
    max_logging.log(
        f"Processing samples [{batch_start}, {end_index}) out of {total_examples}."
    )
    dataset_slice = dataset.select(range(batch_start, end_index))
    requests = build_requests(dataset_slice, tokenizer, config)
    distillation_data = generate_completions(config, requests, tokenizer)
    if distillation_data:
      upload_data(config, distillation_data, batch_num)
    else:
      max_logging.log("No valid requests in this batch; skipping upload.")
    processed = end_index - batch_start
    progress_bar.update(processed)
    progress_bar.set_postfix(batch=batch_num, uploads=len(distillation_data))
    start_index = end_index
    batch_num += 1
    save_progress(
        config.progress_path,
        start_index,
        batch_num,
        status="running",
        total_examples=total_examples,
        completed_examples=start_index,
        last_update_ts=_now_timestamp(),
    )

  progress_bar.close()
  save_progress(
      config.progress_path,
      total_examples,
      batch_num,
      status="completed",
      total_examples=total_examples,
      completed_examples=total_examples,
      last_update_ts=_now_timestamp(),
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--jetstream-server-port", type=str, default=9000, help="JetStream server port.")
  parser.add_argument("--dataset-type", type=str, default="huggingface", help="Type of dataset.")
  parser.add_argument(
      "--dataset-path",
      type=str,
      default="HuggingFaceFW/fineweb-edu",
      help="Path to Hugging Face dataset.",
  )
  parser.add_argument(
      "--data-split",
      type=str,
      default="sample-350BT",
      help="Subset of data to load, eg. train or test.",
  )
  parser.add_argument(
      "--hf-access-token", type=str, required=True, help="Access token used to load a tokenizer from Hugging Face."
  )
  parser.add_argument(
      "--tokenizer-path",
      type=str,
      default="/home/terry/gcs-bucket/HF_HOME/Llama-3.1-8B",
      help="Path to tokenizer or checkpoint directory.",
  )
  parser.add_argument(
      "--text-column",
      type=str,
      default="text",
      help="Name of the column that stores raw text sequences.",
  )
  parser.add_argument("--max-prefill-length", type=int, default=256, help="The maximum prompt length.")
  parser.add_argument(
      "--max-target-length", type=int, default=4096, help="The maximum prompt length plus the output completion length."
  )
  parser.add_argument(
      "--num-generations", type=int, required=False, default=1, help="Number of samples to generate per prompt."
  )
  parser.add_argument("--batch-size", type=int, required=True, help="Number of prompts to process in a batch.")
  parser.add_argument(
      "--max-examples",
      type=int,
      default=None,
      help="Maximum number of examples to process. If not set, processes entire dataset.",
  )
  parser.add_argument(
      "--remove-local-dataset-files", action="store_true", help="Set to remove local dataset files after upload."
  )
  parser.add_argument(
      "--progress-path",
      type=str,
      default="sequence_kd_progress.json",
      help="File used to persist the next dataset index and batch number for resume.",
  )

  # Subparser for available upload commands (upload to GCS, upload to Hugging Face)
  subparsers = parser.add_subparsers(dest="upload", title="Available upload commands", required=True)

  # Subparser to upload dataset to Google Cloud Storage
  upload_to_gcs_parser = subparsers.add_parser("upload-to-gcs", help="Upload dataset to Google Cloud Storage.")
  upload_to_gcs_parser.add_argument(
      "--gcs-bucket", type=str, required=True, help="Name of GCS bucket to upload generated dataset."
  )
  upload_to_gcs_parser.add_argument("--gcs-data-path", type=str, required=True, help="Path to store dataset in GCS bucket.")

  # Subparser to upload dataset to Hugging Face
  upload_to_hf_parser = subparsers.add_parser(
      "upload-to-hf",
      help="Upload dataset to Hugging Face.",
  )
  upload_to_hf_parser.add_argument(
      "--hf-repo-id", type=str, required=True, help="Name of Hugging Face repository to upload generated dataset."
  )

  config = parser.parse_args()

  assert (
      config.max_prefill_length < config.max_target_length
  ), "Maximum length of prompt should be less than maximum target length."
  generate_data(config)
