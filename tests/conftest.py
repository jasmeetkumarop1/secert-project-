"""Shared fixtures for the x_agent test suite."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for test file I/O."""
    return tmp_path


@pytest.fixture
def sample_posts() -> list[dict]:
    """Return sample tweet data as would be returned by X API."""
    return [
        {
            "id": "1001",
            "text": "This is a sample tweet for testing purposes with enough characters.",
            "created_at": "2024-01-15T10:00:00.000Z",
            "author_id": "12345",
            "lang": "en",
            "public_metrics": {"like_count": 5, "retweet_count": 2},
        },
        {
            "id": "1002",
            "text": "Short",
            "created_at": "2024-01-15T11:00:00.000Z",
            "author_id": "12345",
            "lang": "en",
            "public_metrics": {"like_count": 1, "retweet_count": 0},
        },
        {
            "id": "1003",
            "text": "Another longer tweet\nwith newlines and   extra   spaces for cleaning.",
            "created_at": "2024-01-15T12:00:00.000Z",
            "author_id": "12345",
            "lang": "en",
            "public_metrics": {"like_count": 10, "retweet_count": 3},
        },
    ]


@pytest.fixture
def raw_jsonl_file(tmp_dir: Path, sample_posts: list[dict]) -> Path:
    """Create a temporary raw JSONL file from sample posts."""
    path = tmp_dir / "raw_tweets.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for post in sample_posts:
            f.write(json.dumps(post) + "\n")
    return path
