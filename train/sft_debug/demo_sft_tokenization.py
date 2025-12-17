#!/usr/bin/env python3
"""
Demo script showing how SFT tokenization works.

Format: Plain completion (no special tokens)
  {user}\n{assistant}\n

This preserves completion-style evaluation performance on benchmarks like MMLU/ARC.

Usage:
    python demo_sft_tokenization.py --tokenizer_path /path/to/tokenizer
"""

import argparse
from transformers import AutoTokenizer


# =============================================================================
# SFT Template: Plain completion format (no special tokens)
# =============================================================================

TEMPLATE = {
    "user_start": "",
    "user_end": "\n",
    "assistant_start": "",
    "assistant_end": "\n",
}


def format_conversation(messages):
    """
    Format a conversation using plain completion format (no special tokens).

    Returns list of (text, is_prompt) tuples for each segment.
    """
    segments = []

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            text = TEMPLATE["user_start"] + content + TEMPLATE["user_end"]
            segments.append((text, True))  # True = is_prompt (masked)

        elif role == "assistant":
            text = TEMPLATE["assistant_start"] + content + TEMPLATE["assistant_end"]
            segments.append((text, False))  # False = is_completion (loss computed)

    return segments


def show_tokenization(tokenizer, messages):
    """Process messages and show tokenization details."""

    print("=" * 80)
    print("INPUT MESSAGES:")
    print("=" * 80)
    for msg in messages:
        print(f"  [{msg['role']}]: {msg['content']}")
    print()

    # Format conversation
    segments = format_conversation(messages)

    print("=" * 80)
    print("FORMATTED SEGMENTS (repr with \\n visible):")
    print("=" * 80)
    for i, (text, is_prompt) in enumerate(segments):
        label = "PROMPT (masked)" if is_prompt else "COMPLETION (loss)"
        print(f"[{i}] {label}:")
        print(f"    {repr(text)}")
    print()

    # Tokenize each segment
    tokenized_segments = []
    for text, is_prompt in segments:
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        tokenized_segments.append((tokens, is_prompt))

    print("=" * 80)
    print("TOKENIZED SEGMENTS:")
    print("=" * 80)
    for i, (tokens, is_prompt) in enumerate(tokenized_segments):
        label = "PROMPT" if is_prompt else "COMPLETION"
        print(f"[{i}] {label}: {len(tokens)} tokens")
        print(f"    IDs: {tokens}")
    print()

    # Concatenate
    all_input_ids = []
    for tokens, _ in tokenized_segments:
        all_input_ids.extend(tokens)

    # Create targets with prompt masking (use -1 as mask, not 0!)
    MASK_ID = -1
    all_target_ids = []
    for tokens, is_prompt in tokenized_segments:
        if is_prompt:
            all_target_ids.extend([MASK_ID] * len(tokens))
        else:
            all_target_ids.extend(tokens)

    print("=" * 80)
    print("FINAL SEQUENCE (repr with \\n visible):")
    print("=" * 80)
    decoded = tokenizer.decode(all_input_ids)
    print(f"Total tokens: {len(all_input_ids)}")
    print(f"Repr: {repr(decoded)}")
    print()

    # Loss stats
    num_completion = sum(1 for t in all_target_ids if t != MASK_ID)
    num_prompt = len(all_target_ids) - num_completion

    print("=" * 80)
    print("LOSS COMPUTATION (train_on_completion_only=True):")
    print("=" * 80)
    print(f"Prompt tokens (masked):    {num_prompt}")
    print(f"Completion tokens (loss):  {num_completion}")
    print(f"Loss ratio: {num_completion}/{len(all_target_ids)} = {num_completion/len(all_target_ids):.1%}")
    print()

    # Token breakdown
    print("=" * 80)
    print("TOKEN-BY-TOKEN:")
    print("=" * 80)
    print(f"{'Pos':<4} {'ID':<8} {'Target':<8} {'Loss':<5} Token")
    print("-" * 60)
    for i, (inp, tgt) in enumerate(zip(all_input_ids, all_target_ids)):
        has_loss = "YES" if tgt != MASK_ID else "-"
        tgt_str = str(tgt) if tgt != MASK_ID else "MASK"
        token_str = repr(tokenizer.decode([inp]))
        print(f"{i:<4} {inp:<8} {tgt_str:<8} {has_loss:<5} {token_str}")

    return all_input_ids, all_target_ids


def main():
    parser = argparse.ArgumentParser(description="Demo SFT tokenization (plain completion format)")
    parser.add_argument(
        "--tokenizer_path",
        default="/home/terry/gcs-bucket/HF_HOME/Llama-3.2-1B-Instruct",
        help="Path to tokenizer"
    )
    parser.add_argument("--hf_token", default=None, help="HuggingFace access token")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        add_bos_token=False,
        add_eos_token=False,
        legacy=False,
        token=args.hf_token,
    )
    print(f"Vocab size: {tokenizer.vocab_size}")
    print()

    # Show the template format
    print("#" * 80)
    print("# TEMPLATE FORMAT: Plain completion (no special tokens)")
    print("#" * 80)
    print()
    print("Single-turn:")
    print("  {user}\\n{assistant}\\n")
    print()
    print("Multi-turn:")
    print("  {user1}\\n{asst1}\\n{user2}\\n{asst2}\\n")
    print()

    # Single-turn example
    print("\n" + "#" * 80)
    print("# EXAMPLE 1: Single-turn")
    print("#" * 80 + "\n")

    single_turn = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
    ]
    show_tokenization(tokenizer, single_turn)

    # Multi-turn example
    print("\n" + "#" * 80)
    print("# EXAMPLE 2: Multi-turn")
    print("#" * 80 + "\n")

    multi_turn = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]
    show_tokenization(tokenizer, multi_turn)


if __name__ == "__main__":
    main()
