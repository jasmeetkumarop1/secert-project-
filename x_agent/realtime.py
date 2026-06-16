"""Near-real-time X polling that updates agent memory as often as every second."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any


from x_agent.collect import fetch_recent_posts
from x_agent.memory import AgentMemory


def _parse_x_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        try:
            return parsedate_to_datetime(value).timestamp()
        except (TypeError, ValueError):
            return None


def store_posts(memory: AgentMemory, posts: list[dict[str, Any]]) -> int:
    """Store X posts in memory and return the number of new posts."""
    inserted = 0
    for post in posts:
        text = post.get("text", "")
        external_id = str(post.get("id", "")) or None
        created_at = _parse_x_timestamp(post.get("created_at"))
        if memory.remember(
            text,
            source="x_post",
            external_id=external_id,
            created_at=created_at,
        ):
            inserted += 1
    return inserted


def poll_x_into_memory(query: str, bearer_token: str, memory: AgentMemory, interval: float = 1.0) -> None:
    """Continuously poll X and update local memory.

    X API rate limits may require a larger interval depending on your developer
    plan. The one-second interval is a best-effort local update cadence, not a
    guarantee that the remote API will allow one request per second forever.
    """
    print(f"Polling X every {interval:g}s. Press Ctrl+C to stop.")
    while True:
        started = time.monotonic()
        posts = fetch_recent_posts(query, bearer_token, max_results=10)
        inserted = store_posts(memory, posts)
        print(f"remembered {inserted} new posts; total recent checked={len(posts)}")
        elapsed = time.monotonic() - started
        time.sleep(max(0.0, interval - elapsed))


def main() -> None:
    parser = argparse.ArgumentParser(description="Update agent memory from X on a fast polling loop")
    parser.add_argument("--query", required=True, help="X API recent-search query")
    parser.add_argument("--memory", default="data/agent_memory.sqlite", help="SQLite memory path")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    bearer_token = os.getenv("X_BEARER_TOKEN")
    if not bearer_token:
        raise SystemExit("Set X_BEARER_TOKEN in your environment or .env file")

    memory = AgentMemory(args.memory)
    try:
        poll_x_into_memory(args.query, bearer_token, memory, args.interval)
    finally:
        memory.close()


if __name__ == "__main__":
    main()
