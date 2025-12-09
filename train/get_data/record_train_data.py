"""
Record training data samples from MaxText Grain pipeline.

This script replicates the exact data sampling order used during training
by using the same configuration parameters and random seeds.

It samples data with a configurable probability (default 0.1%) and saves
the token sequences to a pickle file for later analysis.
"""

import os
import sys
import glob
import pickle
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import time

from tqdm import tqdm

# Add MaxText to path
maxtext_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, maxtext_path)

import numpy as np
import grain.python as grain
import tensorflow as tf

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


@dataclass
class DataConfig:
    """Configuration for data recording - mirrors training config."""
    grain_train_files: str
    grain_file_type: str = "arrayrecord"
    enable_data_shuffling: bool = True
    data_shuffle_seed: int = 43
    num_epoch: int = 1
    max_target_length: int = 8192
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    packing: bool = True
    tokenize_train_data: bool = False  # Data is pre-tokenized
    train_data_columns: tuple = ("text",)
    # For single host setup
    dataloading_host_index: int = 0
    dataloading_host_count: int = 1
    num_devices: int = 1  # Will be overridden based on actual TPU/GPU setup


def find_data_files(data_file_pattern: str) -> list:
    """Find all data files matching the pattern."""
    data_files = glob.glob(str(Path(data_file_pattern).expanduser().resolve()))
    assert len(data_files) > 0, f"No file found with pattern {data_file_pattern}."
    print(f"Found {len(data_files)} files for training data")
    return sorted(data_files)  # Sort for reproducibility


@dataclass
class ParseFeatures(grain.MapTransform):
    """Parse serialized example from array record."""
    data_columns: tuple

    def map(self, element):
        parsed = tf.io.parse_example(
            element,
            {col: tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True)
             for col in self.data_columns},
        )
        return parsed


@dataclass
class NormalizeFeatures(grain.MapTransform):
    """Normalize features to numpy arrays."""
    column_names: tuple

    def map(self, element):
        return {col: element[col].numpy() for col in self.column_names}


@dataclass
class Rekey(grain.MapTransform):
    """Rename keys according to mapping."""
    mapping_dict: dict

    def map(self, element):
        for new_key, old_key in self.mapping_dict.items():
            element[new_key] = element[old_key]
        for old_key in set(self.mapping_dict.values()):
            if old_key not in self.mapping_dict:
                del element[old_key]
        return element


@dataclass
class RekeySimple(grain.MapTransform):
    """Simplified rekey that creates inputs/targets from text."""

    def map(self, element):
        return {
            "inputs": element["text"],
            "targets": element["text"]
        }


@dataclass
class PadToMaxLength(grain.MapTransform):
    """Pad sequences to max length."""
    max_length: int
    pad_id: int = 0

    def map(self, element):
        def _pad(x, max_length, pad_id):
            pad_amount = max(max_length - x.shape[0], 0)
            return np.pad(x, (0, pad_amount), constant_values=pad_id)[:max_length]

        for key in list(element.keys()):
            if key in ("inputs", "targets"):
                element[f"{key}_segmentation"] = (element[key] != self.pad_id).astype(np.int32)
                element[f"{key}_position"] = np.arange(len(element[key]), dtype=np.int32)
                element[key] = _pad(element[key], self.max_length, self.pad_id)
                element[f"{key}_segmentation"] = _pad(element[f"{key}_segmentation"], self.max_length, 0)
                element[f"{key}_position"] = _pad(element[f"{key}_position"], self.max_length, 0)
        return element


def create_data_iterator(config: DataConfig):
    """
    Create data iterator matching the training pipeline.

    This replicates the exact sampling order from training.
    """
    print(f"Creating data iterator with:")
    print(f"  - Files: {config.grain_train_files}")
    print(f"  - Shuffle seed: {config.data_shuffle_seed}")
    print(f"  - Shuffling enabled: {config.enable_data_shuffling}")
    print(f"  - Max target length: {config.max_target_length}")
    print(f"  - Packing: {config.packing}")

    # Find and load data files
    data_files = find_data_files(config.grain_train_files)

    # Create the dataset source
    dataset = grain.MapDataset.source(grain.ArrayRecordDataSource(data_files))

    # Apply shuffling with the same seed as training
    if config.enable_data_shuffling:
        dataset = dataset.shuffle(seed=config.data_shuffle_seed)
        print(f"Applied shuffle with seed {config.data_shuffle_seed}")

    # Repeat for num_epoch
    dataset = dataset.repeat(config.num_epoch)

    # Shard across hosts (for single host, this is identity)
    dataset = dataset[config.dataloading_host_index::config.dataloading_host_count]

    # Convert to iterable
    dataset = dataset.to_iter_dataset()

    # Parse and normalize features
    dataset = dataset.map(ParseFeatures(config.train_data_columns))
    dataset = dataset.map(NormalizeFeatures(config.train_data_columns))

    # Rekey to inputs/targets
    dataset = dataset.map(RekeySimple())

    # Apply packing or padding
    if config.packing:
        length_struct = {"inputs": config.max_target_length, "targets": config.max_target_length}
        dataset = grain.experimental.FirstFitPackIterDataset(dataset, length_struct=length_struct, num_packing_bins=30)

        # Rekey packing outputs
        rekey_dict = {
            "targets_segmentation": "targets_segment_ids",
            "inputs_segmentation": "inputs_segment_ids",
            "targets_position": "targets_positions",
            "inputs_position": "inputs_positions",
        }
        dataset = dataset.map(Rekey(rekey_dict))
    else:
        dataset = dataset.map(PadToMaxLength(config.max_target_length, pad_id=0))

    # Compute batch size
    # In actual training: global_batch_size_to_load = num_devices * per_device_batch_size * grad_accum
    # Then batched by: global_batch_size_to_load // process_count
    # For single-host single-process: batch_size = num_devices * per_device_batch_size * grad_accum
    batch_size = config.num_devices * config.per_device_batch_size * config.gradient_accumulation_steps
    print(f"Batch size: {batch_size}")

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)

    return iter(dataset)


def record_training_data(
    config: DataConfig,
    num_steps: int,
    sample_prob: float = 0.001,
    output_file: str = "train_data_samples.pkl",
    save_interval: int = 1000,
):
    """
    Record training data samples.

    Args:
        config: Data configuration
        num_steps: Number of training steps to simulate
        sample_prob: Probability of recording each sequence (0.001 = 0.1%)
        output_file: Output pickle file path
        save_interval: Save progress every N steps
    """
    print(f"\n{'='*60}")
    print(f"Recording training data for {num_steps} steps")
    print(f"Sample probability: {sample_prob*100:.2f}%")
    print(f"Output file: {output_file}")
    print(f"{'='*60}\n")

    # Create iterator
    data_iter = create_data_iterator(config)

    # Use a fixed random seed for sampling decisions (separate from data shuffle seed)
    # This ensures reproducibility of which samples we record
    sample_rng = random.Random(12345)

    # Storage for samples
    samples = {
        "config": {
            "grain_train_files": config.grain_train_files,
            "data_shuffle_seed": config.data_shuffle_seed,
            "max_target_length": config.max_target_length,
            "per_device_batch_size": config.per_device_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "packing": config.packing,
            "num_steps": num_steps,
            "sample_prob": sample_prob,
        },
        "samples": []  # List of (step, batch_idx, sequence) tuples
    }

    total_sequences = 0
    recorded_sequences = 0
    start_time = time.time()

    pbar = tqdm(range(num_steps), desc="Recording", unit="step")

    try:
        for step in pbar:
            batch = next(data_iter)
            batch_size = batch["inputs"].shape[0]

            for batch_idx in range(batch_size):
                total_sequences += 1

                # Sample with given probability
                if sample_rng.random() < sample_prob:
                    # Record this sequence
                    sample_data = {
                        "step": step,
                        "batch_idx": batch_idx,
                        "inputs": batch["inputs"][batch_idx].copy(),
                        "targets": batch["targets"][batch_idx].copy(),
                    }

                    # Include segmentation if available (for packed data)
                    if "inputs_segmentation" in batch:
                        sample_data["inputs_segmentation"] = batch["inputs_segmentation"][batch_idx].copy()
                        sample_data["targets_segmentation"] = batch["targets_segmentation"][batch_idx].copy()

                    samples["samples"].append(sample_data)
                    recorded_sequences += 1

            # Update progress bar
            pbar.set_postfix({
                "seqs": total_sequences,
                "recorded": recorded_sequences,
            })

            # Periodic save
            if (step + 1) % save_interval == 0:
                temp_file = output_file + ".tmp"
                with open(temp_file, 'wb') as f:
                    pickle.dump(samples, f)
                os.rename(temp_file, output_file)
                tqdm.write(f"  -> Saved checkpoint at step {step + 1}")

    except StopIteration:
        print(f"Data exhausted at step {step}")

    pbar.close()

    # Final save
    samples["metadata"] = {
        "total_sequences": total_sequences,
        "recorded_sequences": recorded_sequences,
        "actual_sample_rate": recorded_sequences / total_sequences if total_sequences > 0 else 0,
        "completed_steps": step + 1 if 'step' in dir() else 0,
        "elapsed_time": time.time() - start_time,
    }

    with open(output_file, 'wb') as f:
        pickle.dump(samples, f)

    print(f"\n{'='*60}")
    print(f"Recording complete!")
    print(f"Total sequences seen: {total_sequences}")
    print(f"Sequences recorded: {recorded_sequences} ({100*recorded_sequences/total_sequences:.2f}%)")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Record training data samples from MaxText")

    # Data configuration - match training script
    parser.add_argument("--grain_train_files", type=str, required=True,
                        help="Path pattern for training data files")
    parser.add_argument("--data_shuffle_seed", type=int, default=43,
                        help="Shuffle seed (must match training)")
    parser.add_argument("--max_target_length", type=int, default=8192,
                        help="Max sequence length")
    parser.add_argument("--per_device_batch_size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--num_devices", type=int, default=1,
                        help="Number of devices (TPU chips / GPUs)")
    parser.add_argument("--packing", type=str, default="True",
                        help="Whether packing is enabled (True/False)")
    parser.add_argument("--enable_data_shuffling", type=str, default="True",
                        help="Whether data shuffling is enabled (True/False)")

    # Recording configuration
    parser.add_argument("--num_steps", type=int, required=True,
                        help="Number of training steps to simulate")
    parser.add_argument("--sample_prob", type=float, default=0.001,
                        help="Probability of recording each sequence (0.001 = 0.1%)")
    parser.add_argument("--output_file", type=str, default="train_data_samples.pkl",
                        help="Output pickle file path")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Save checkpoint every N steps")

    # Host configuration for multi-host
    parser.add_argument("--dataloading_host_index", type=int, default=0,
                        help="Index of this host for data loading")
    parser.add_argument("--dataloading_host_count", type=int, default=1,
                        help="Total number of data loading hosts")

    args = parser.parse_args()

    # Convert string booleans
    def str_to_bool(s):
        return s.lower() in ('true', '1', 'yes')

    # Create config
    config = DataConfig(
        grain_train_files=args.grain_train_files,
        enable_data_shuffling=str_to_bool(args.enable_data_shuffling),
        data_shuffle_seed=args.data_shuffle_seed,
        max_target_length=args.max_target_length,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        packing=str_to_bool(args.packing),
        num_devices=args.num_devices,
        dataloading_host_index=args.dataloading_host_index,
        dataloading_host_count=args.dataloading_host_count,
    )

    # Record data
    record_training_data(
        config=config,
        num_steps=args.num_steps,
        sample_prob=args.sample_prob,
        output_file=args.output_file,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
