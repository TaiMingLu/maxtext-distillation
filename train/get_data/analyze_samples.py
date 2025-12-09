"""
Utility script to analyze and decode recorded training samples.

Usage:
    python analyze_samples.py --samples_file train_data_samples.pkl --tokenizer_path /path/to/tokenizer
"""

import argparse
import pickle
from pathlib import Path


def load_samples(samples_file: str) -> dict:
    """Load samples from pickle file."""
    with open(samples_file, 'rb') as f:
        return pickle.load(f)


def print_summary(data: dict):
    """Print summary of recorded samples."""
    print("\n" + "="*60)
    print("SAMPLE RECORDING SUMMARY")
    print("="*60)

    config = data.get("config", {})
    metadata = data.get("metadata", {})
    samples = data.get("samples", [])

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print("\nMetadata:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")

    print(f"\nTotal samples recorded: {len(samples)}")

    if samples:
        print("\nSample distribution by step:")
        steps = [s["step"] for s in samples]
        min_step, max_step = min(steps), max(steps)
        print(f"  Steps covered: {min_step} to {max_step}")

        # Bin by 1000 steps
        bins = {}
        for step in steps:
            bin_key = (step // 1000) * 1000
            bins[bin_key] = bins.get(bin_key, 0) + 1

        print("  Samples per 1000 steps:")
        for k in sorted(bins.keys()):
            print(f"    Steps {k}-{k+999}: {bins[k]} samples")


def decode_sample(sample: dict, tokenizer) -> dict:
    """Decode a single sample using the tokenizer."""
    inputs = sample["inputs"]
    # Remove padding (zeros)
    non_pad = inputs != 0
    if hasattr(non_pad, 'numpy'):
        non_pad = non_pad.numpy()
    actual_tokens = inputs[non_pad]

    decoded = {
        "step": sample["step"],
        "batch_idx": sample["batch_idx"],
        "num_tokens": len(actual_tokens),
        "text": tokenizer.decode(actual_tokens.tolist()) if tokenizer else None,
    }

    # If packed, show segment info
    if "inputs_segmentation" in sample:
        seg = sample["inputs_segmentation"]
        unique_segments = set(seg[seg > 0].tolist())
        decoded["num_packed_examples"] = len(unique_segments)

    return decoded


def main():
    parser = argparse.ArgumentParser(description="Analyze recorded training samples")
    parser.add_argument("--samples_file", type=str, required=True,
                        help="Path to samples pickle file")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer for decoding")
    parser.add_argument("--show_samples", type=int, default=5,
                        help="Number of samples to decode and display")
    parser.add_argument("--output_text", type=str, default=None,
                        help="Output file to save decoded samples")

    args = parser.parse_args()

    # Load samples
    print(f"Loading samples from {args.samples_file}...")
    data = load_samples(args.samples_file)

    # Print summary
    print_summary(data)

    samples = data.get("samples", [])
    if not samples:
        print("No samples to analyze.")
        return

    # Load tokenizer if provided
    tokenizer = None
    if args.tokenizer_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
            print(f"\nLoaded tokenizer from {args.tokenizer_path}")
        except Exception as e:
            print(f"\nWarning: Could not load tokenizer: {e}")

    # Decode and display samples
    if args.show_samples > 0:
        print(f"\n{'='*60}")
        print(f"SAMPLE PREVIEW (first {args.show_samples} samples)")
        print("="*60)

        for i, sample in enumerate(samples[:args.show_samples]):
            decoded = decode_sample(sample, tokenizer)
            print(f"\n--- Sample {i+1} ---")
            print(f"Step: {decoded['step']}, Batch idx: {decoded['batch_idx']}")
            print(f"Tokens: {decoded['num_tokens']}")
            if "num_packed_examples" in decoded:
                print(f"Packed examples: {decoded['num_packed_examples']}")
            if decoded["text"]:
                # Show first 500 chars
                text_preview = decoded["text"][:500]
                if len(decoded["text"]) > 500:
                    text_preview += "..."
                print(f"Text preview:\n{text_preview}")

    # Save decoded samples if requested
    if args.output_text and tokenizer:
        print(f"\nSaving decoded samples to {args.output_text}...")
        with open(args.output_text, 'w') as f:
            for i, sample in enumerate(samples):
                decoded = decode_sample(sample, tokenizer)
                f.write(f"=== Sample {i+1} (Step {decoded['step']}, Batch {decoded['batch_idx']}) ===\n")
                f.write(f"Tokens: {decoded['num_tokens']}\n")
                if decoded["text"]:
                    f.write(f"Text:\n{decoded['text']}\n")
                f.write("\n")
        print(f"Saved {len(samples)} decoded samples.")


def merge_host_samples(host_files: list, output_file: str):
    """Merge samples from multiple hosts into a single file."""
    all_samples = []
    config = None
    total_sequences = 0
    recorded_sequences = 0

    for hf in host_files:
        data = load_samples(hf)
        if config is None:
            config = data.get("config", {})

        all_samples.extend(data.get("samples", []))

        meta = data.get("metadata", {})
        total_sequences += meta.get("total_sequences", 0)
        recorded_sequences += meta.get("recorded_sequences", 0)

    # Sort by step, then batch_idx
    all_samples.sort(key=lambda x: (x["step"], x.get("host_index", 0), x["batch_idx"]))

    merged = {
        "config": config,
        "samples": all_samples,
        "metadata": {
            "total_sequences": total_sequences,
            "recorded_sequences": recorded_sequences,
            "actual_sample_rate": recorded_sequences / total_sequences if total_sequences > 0 else 0,
            "merged_from": host_files,
        }
    }

    with open(output_file, 'wb') as f:
        pickle.dump(merged, f)

    print(f"Merged {len(all_samples)} samples from {len(host_files)} hosts to {output_file}")


if __name__ == "__main__":
    main()
