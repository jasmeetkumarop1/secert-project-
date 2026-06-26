"""Collect posts from X API v2 into JSONL.

This module intentionally uses the official X API instead of scraping. Only collect
and train on content you own or are licensed to use.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"


def fetch_recent_posts(query: str, bearer_token: str, max_results: int = 100) -> list[dict[str, Any]]:
    """Fetch recent X posts matching a query using the official API."""
    if not 10 <= max_results <= 100:
        raise ValueError("max_results must be between 10 and 100 for one recent-search request")

    try:
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
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Failed to connect to X API: {exc}") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("X API request timed out after 30 seconds") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"X API request failed: {exc}") from exc

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"X API returned HTTP {response.status_code}: {response.text[:500]}"
        ) from exc

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"X API returned invalid JSON: {response.text[:500]}"
        ) from exc

    if "errors" in body:
        error_msgs = "; ".join(e.get("message", str(e)) for e in body["errors"])
        raise RuntimeError(f"X API returned errors: {error_msgs}")

    posts = body.get("data")
    if posts is None:
        logger.warning("X API response contained no 'data' field; returned 0 posts")
        return []

    return posts


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

    try:
        posts = fetch_recent_posts(args.query, bearer_token, args.max_results)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error fetching posts: {exc}") from exc

    write_jsonl(posts, Path(args.output))
    print(f"Wrote {len(posts)} posts to {args.output}")


if __name__ == "__main__":
    main()
