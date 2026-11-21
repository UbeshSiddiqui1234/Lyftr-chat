"""SQLite persistence for chat sessions."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages (session_id, id);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    system_prompt TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, path: str) -> None:
        self.path = path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def init_schema(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)

    def ping(self) -> None:
        with self._connect() as connection:
            connection.execute("SELECT 1 FROM messages LIMIT 1").fetchone()

    def load_context(self, session_id: str, limit: int) -> tuple[str | None, list[dict[str, str]]]:
        """Return the session's system prompt and its most recent messages."""
        with self._connect() as connection:
            session = connection.execute(
                "SELECT system_prompt FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            rows = connection.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit - limit % 2),
            ).fetchall()
        history = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
        return (session["system_prompt"] if session else None), history

    def save_turn(
        self,
        session_id: str,
        user_message: str,
        reply: str,
        metadata: dict,
        received_at: str,
        system_prompt: str | None,
        update_system_prompt: bool,
    ) -> None:
        """Atomically store a completed exchange."""
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (session_id, system_prompt, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (session_id) DO UPDATE SET
                    system_prompt = CASE WHEN ? THEN excluded.system_prompt
                                         ELSE sessions.system_prompt END,
                    updated_at = excluded.updated_at
                """,
                (session_id, system_prompt, received_at, now, update_system_prompt),
            )
            connection.executemany(
                """
                INSERT INTO messages (session_id, role, content, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (session_id, "user", user_message, None, received_at),
                    (session_id, "assistant", reply, json.dumps(metadata), now),
                ],
            )

    def get_messages(self, session_id: str) -> list[dict] | None:
        """Return a session's full history, oldest first, or None if unknown."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role, content, metadata FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        if not rows:
            return None
        messages = []
        for row in rows:
            message = {"role": row["role"], "content": row["content"]}
            if row["role"] == "assistant" and row["metadata"]:
                message["metadata"] = json.loads(row["metadata"])
            messages.append(message)
        return messages
