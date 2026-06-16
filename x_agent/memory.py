"""SQLite memory layer for fast social-context updates and recall.

The memory store is intentionally separate from model fine-tuning. Updating a
neural model every second is usually impractical on local hardware, but updating
retrieval memory every second is fast and gives the agent fresh context without
forgetting earlier important facts.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemoryItem:
    """A remembered item returned from the memory store."""

    id: int
    source: str
    text: str
    created_at: float
    pinned: bool = False


class AgentMemory:
    """Persistent memory with full-text search and pinned first-memory support."""

    def __init__(self, path: str | Path = "data/agent_memory.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._setup()

    def _setup(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                external_id TEXT,
                text TEXT NOT NULL,
                created_at REAL NOT NULL,
                pinned INTEGER NOT NULL DEFAULT 0,
                UNIQUE(source, external_id)
            )
            """
        )
        self.connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                text,
                source UNINDEXED,
                content='memories',
                content_rowid='id'
            )
            """
        )
        self.connection.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, text, source) VALUES (new.id, new.text, new.source);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, text, source)
                VALUES('delete', old.id, old.text, old.source);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, text, source)
                VALUES('delete', old.id, old.text, old.source);
                INSERT INTO memories_fts(rowid, text, source) VALUES (new.id, new.text, new.source);
            END;
            """
        )
        self.connection.commit()

    def remember(
        self,
        text: str,
        *,
        source: str = "user",
        external_id: str | None = None,
        pinned: bool = False,
        created_at: float | None = None,
    ) -> bool:
        """Store a memory item and return True when a new row was inserted."""
        clean_text = " ".join(text.split())
        if not clean_text:
            return False

        timestamp = created_at if created_at is not None else time.time()
        before = self.connection.total_changes
        self.connection.execute(
            """
            INSERT OR IGNORE INTO memories(source, external_id, text, created_at, pinned)
            VALUES (?, ?, ?, ?, ?)
            """,
            (source, external_id, clean_text, timestamp, int(pinned)),
        )
        self.connection.commit()
        return self.connection.total_changes > before

    def remember_first(self, text: str) -> bool:
        """Pin the first user-provided fact so the agent can always recall it."""
        existing = self.connection.execute(
            "SELECT id FROM memories WHERE source = 'first_user_memory' LIMIT 1"
        ).fetchone()
        if existing:
            return False
        return self.remember(text, source="first_user_memory", external_id="first", pinned=True)

    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        """Return pinned memories plus full-text matches for the query."""
        pinned_rows = self.connection.execute(
            "SELECT id, source, text, created_at, pinned FROM memories WHERE pinned = 1 ORDER BY created_at ASC"
        ).fetchall()

        matched_rows: list[sqlite3.Row] = []
        clean_query = " ".join(query.split())
        if clean_query:
            # Quote the query so arbitrary user prompts with punctuation do not
            # become invalid FTS syntax. This performs phrase-style retrieval.
            fts_query = f'"{clean_query.replace(chr(34), chr(34) + chr(34))}"'
            matched_rows = self.connection.execute(
                """
                SELECT m.id, m.source, m.text, m.created_at, m.pinned
                FROM memories_fts f
                JOIN memories m ON m.id = f.rowid
                WHERE memories_fts MATCH ?
                ORDER BY bm25(memories_fts)
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()

        seen: set[int] = set()
        items: list[MemoryItem] = []
        for row in [*pinned_rows, *matched_rows]:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            items.append(
                MemoryItem(
                    id=row["id"],
                    source=row["source"],
                    text=row["text"],
                    created_at=row["created_at"],
                    pinned=bool(row["pinned"]),
                )
            )
        return items[:limit]

    def recent(self, limit: int = 10) -> list[MemoryItem]:
        """Return the most recent memory items."""
        rows = self.connection.execute(
            """
            SELECT id, source, text, created_at, pinned
            FROM memories
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            MemoryItem(row["id"], row["source"], row["text"], row["created_at"], bool(row["pinned"]))
            for row in rows
        ]

    def close(self) -> None:
        """Close the SQLite connection."""
        self.connection.close()
