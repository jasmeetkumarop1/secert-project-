"""Prepare raw X posts for causal language-model training."""

from __future__ import annotations

import argparse
from pathlib import Path

from x_agent.utils import clean_text, read_jsonl, write_jsonl


def prepare_dataset(input_path: Path, output_path: Path, min_chars: int = 20) -> int:
    """Convert raw tweet JSONL to training JSONL with a single text field."""
    raw_records = read_jsonl(input_path)
    cleaned = [clean_text(r.get("text", "")) for r in raw_records]
    prepared = [
        {"text": f"<x_post>{text}</x_post>"}
        for text in cleaned
        if len(text) >= min_chars
    ]
    write_jsonl(prepared, output_path)
    return len(prepared)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare X posts for model training")
    parser.add_argument("--input", default="data/raw_tweets.jsonl", help="Raw JSONL path")
    parser.add_argument("--output", default="data/train.jsonl", help="Prepared JSONL path")
    parser.add_argument("--min-chars", type=int, default=20, help="Skip posts shorter than this")
    args = parser.parse_args()

    count = prepare_dataset(Path(args.input), Path(args.output), args.min_chars)
    print(f"Prepared {count} training examples at {args.output}")


if __name__ == "__main__":
    main()
