"""Prepare raw X posts for causal language-model training."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Normalize tweet text while preserving author style."""
    return " ".join(text.replace("\n", " ").split())


def prepare_dataset(input_path: Path, output_path: Path, min_chars: int = 20) -> int:
    """Convert raw tweet JSONL to training JSONL with a single text field."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line_num, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d: %s", line_num, exc)
                skipped += 1
                continue

            if "text" not in record:
                logger.warning("Skipping line %d: missing 'text' field", line_num)
                skipped += 1
                continue

            text = clean_text(record["text"])
            if len(text) < min_chars:
                continue
            target.write(json.dumps({"text": f"<x_post>{text}</x_post>"}, ensure_ascii=False) + "\n")
            count += 1

    if skipped:
        logger.warning("Skipped %d problematic lines during preparation", skipped)

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare X posts for model training")
    parser.add_argument("--input", default="data/raw_tweets.jsonl", help="Raw JSONL path")
    parser.add_argument("--output", default="data/train.jsonl", help="Prepared JSONL path")
    parser.add_argument("--min-chars", type=int, default=20, help="Skip posts shorter than this")
    args = parser.parse_args()

    try:
        count = prepare_dataset(Path(args.input), Path(args.output), args.min_chars)
    except FileNotFoundError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    except OSError as exc:
        raise SystemExit(f"I/O error during preparation: {exc}") from exc

    if count == 0:
        logger.warning("No training examples produced; check the input data")
    print(f"Prepared {count} training examples at {args.output}")


if __name__ == "__main__":
    main()
