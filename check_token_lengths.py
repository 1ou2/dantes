"""
Check token lengths in the training dataset to verify max_seq_length requirements

Usage:
    python check_token_lengths.py [--dataset path/to/dataset.jsonl] [--max_length 2048]
"""

import argparse
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
import numpy as np

def main(args):
    print("=" * 60)
    print("🔍 Analyzing dataset token lengths")
    print("=" * 60)

    # Load tokenizer (lightweight, no model needed)
    print("\n📥 Loading tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Setup chat template (same as training script)
    print("📝 Setting up ChatML template...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        map_eos_token=True,
    )

    # Load dataset
    print(f"\n📚 Loading dataset from {args.dataset}...")
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    print(f"   Dataset size: {len(dataset)} conversations")

    # Tokenize all conversations
    print("\n🔢 Tokenizing conversations...")
    token_lengths = []
    long_conversations = []

    for idx, example in enumerate(dataset):
        # Apply chat template
        text = tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize
        tokens = tokenizer.encode(text)
        length = len(tokens)
        token_lengths.append(length)

        # Track conversations exceeding max_length
        if length > args.max_length:
            long_conversations.append({
                "index": idx,
                "length": length,
                "text_preview": text[:200] + "..."
            })

    # Calculate statistics
    token_lengths = np.array(token_lengths)

    print("\n" + "=" * 60)
    print("📊 Token Length Statistics")
    print("=" * 60)
    print(f"Min length:     {token_lengths.min():>6} tokens")
    print(f"Max length:     {token_lengths.max():>6} tokens")
    print(f"Mean length:    {token_lengths.mean():>6.1f} tokens")
    print(f"Median length:  {np.median(token_lengths):>6.1f} tokens")
    print(f"Std deviation:  {token_lengths.std():>6.1f} tokens")
    print()
    print(f"25th percentile: {np.percentile(token_lengths, 25):>6.1f} tokens")
    print(f"50th percentile: {np.percentile(token_lengths, 50):>6.1f} tokens")
    print(f"75th percentile: {np.percentile(token_lengths, 75):>6.1f} tokens")
    print(f"95th percentile: {np.percentile(token_lengths, 95):>6.1f} tokens")
    print(f"99th percentile: {np.percentile(token_lengths, 99):>6.1f} tokens")

    # Check against max_length
    print("\n" + "=" * 60)
    print(f"⚠️  Conversations exceeding {args.max_length} tokens")
    print("=" * 60)

    if long_conversations:
        print(f"\n❌ Found {len(long_conversations)} conversations exceeding max_length:")
        print(f"   ({len(long_conversations) / len(dataset) * 100:.2f}% of dataset)")
        print("\nLongest conversations:")
        for conv in sorted(long_conversations, key=lambda x: x["length"], reverse=True)[:10]:
            print(f"\n   Index {conv['index']}: {conv['length']} tokens")
            print(f"   Preview: {conv['text_preview']}")

        print(f"\n⚠️  RECOMMENDATION: Increase max_seq_length to at least {token_lengths.max()}")
    else:
        print(f"\n✅ All conversations fit within {args.max_length} tokens!")
        print(f"   Current max_seq_length ({args.max_length}) is sufficient.")

        # Suggest optimization if there's a lot of headroom
        if token_lengths.max() < args.max_length * 0.7:
            print(f"\n💡 OPTIMIZATION: Consider reducing max_seq_length to {int(np.percentile(token_lengths, 99))}")
            print(f"   This would speed up training without truncating conversations.")

    # Packing efficiency estimate
    print("\n" + "=" * 60)
    print("📦 Packing Efficiency Estimate")
    print("=" * 60)

    avg_length = token_lengths.mean()
    packing_ratio = args.max_length / avg_length

    print(f"Average conversation length: {avg_length:.1f} tokens")
    print(f"Max sequence length: {args.max_length} tokens")
    print(f"Estimated packing ratio: {packing_ratio:.1f}x")
    print(f"\n💡 With packing enabled, you can fit ~{packing_ratio:.1f} conversations per sequence")
    print(f"   This gives approximately {packing_ratio:.1f}x speedup compared to no packing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check token lengths in training dataset")

    parser.add_argument("--dataset", type=str, default="data/dataset/dantes_conversations.jsonl",
                        help="Path to the JSONL dataset")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length to check against")

    args = parser.parse_args()
    main(args)
