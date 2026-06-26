"""Shared I/O and text utilities for the x_agent package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines, and return parsed records."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records as newline-delimited JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    """Normalize text by collapsing whitespace and removing newlines."""
    return " ".join(text.replace("\n", " ").split())
