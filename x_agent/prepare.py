"""Prepare raw X posts for causal language-model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def clean_text(text: str) -> str:
    """Normalize tweet text while preserving author style."""
    return " ".join(text.replace("\n", " ").split())


def prepare_dataset(input_path: Path, output_path: Path, min_chars: int = 20) -> int:
    """Convert raw tweet JSONL to training JSONL with a single text field."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = clean_text(record.get("text", ""))
            if len(text) < min_chars:
                continue
            target.write(json.dumps({"text": f"<x_post>{text}</x_post>"}, ensure_ascii=False) + "\n")
            count += 1

    return count


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
