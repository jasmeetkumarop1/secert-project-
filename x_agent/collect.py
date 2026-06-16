"""Collect posts from X API v2 into JSONL.

This module intentionally uses the official X API instead of scraping. Only collect
and train on content you own or are licensed to use.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"


def fetch_recent_posts(query: str, bearer_token: str, max_results: int = 100) -> list[dict[str, Any]]:
    """Fetch recent X posts matching a query using the official API."""
    if not 10 <= max_results <= 100:
        raise ValueError("max_results must be between 10 and 100 for one recent-search request")

    response = requests.get(
        SEARCH_URL,
        headers={"Authorization": f"Bearer {bearer_token}"},
        params={
            "query": query,
            "max_results": max_results,
            "tweet.fields": "created_at,author_id,lang,public_metrics",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("data", [])


def write_jsonl(posts: list[dict[str, Any]], output: Path) -> None:
    """Write posts as newline-delimited JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for post in posts:
            handle.write(json.dumps(post, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect recent X posts into JSONL")
    parser.add_argument("--query", required=True, help="X API recent-search query")
    parser.add_argument("--output", default="data/raw_tweets.jsonl", help="Output JSONL path")
    parser.add_argument("--max-results", type=int, default=100, help="Number of posts to fetch, 10-100")
    args = parser.parse_args()

    load_dotenv()
    bearer_token = os.getenv("X_BEARER_TOKEN")
    if not bearer_token:
        raise SystemExit("Set X_BEARER_TOKEN in your environment or .env file")

    posts = fetch_recent_posts(args.query, bearer_token, args.max_results)
    write_jsonl(posts, Path(args.output))
    print(f"Wrote {len(posts)} posts to {args.output}")


if __name__ == "__main__":
    main()
