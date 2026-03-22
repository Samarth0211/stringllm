"""SQLite-backed persistent conversation memory using aiosqlite."""

from __future__ import annotations

import os
from pathlib import Path

import aiosqlite

from stringllm.memory.base import BaseMemory


class SQLiteMemory(BaseMemory):
    """Persistent conversation memory stored in an SQLite database.

    Each conversation is isolated by *conversation_id*, allowing multiple
    independent conversations within the same database file.

    Args:
        db_path: Path to the SQLite database file.  Defaults to
            ``~/.stringllm/memory.db``.  Parent directories are created
            automatically.
        conversation_id: Identifier for the conversation.  Defaults to
            ``"default"``.
    """

    _CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """

    _CREATE_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_messages_conversation
        ON messages (conversation_id)
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        conversation_id: str = "default",
    ) -> None:
        if db_path is None:
            db_path = Path.home() / ".stringllm" / "memory.db"
        self._db_path = Path(db_path)
        self._conversation_id = conversation_id
        self._initialised = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_initialised(self) -> None:
        """Create the database directory, file, and tables if necessary."""
        if self._initialised:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(self._CREATE_TABLE_SQL)
            await db.execute(self._CREATE_INDEX_SQL)
            await db.commit()

        self._initialised = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add(self, role: str, content: str) -> None:
        """Persist a message to the SQLite database."""
        await self._ensure_initialised()

        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                (self._conversation_id, role, content),
            )
            await db.commit()

    async def get_history(self) -> list[dict[str, str]]:
        """Retrieve the full conversation history, ordered chronologically."""
        await self._ensure_initialised()

        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT role, content FROM messages "
                "WHERE conversation_id = ? ORDER BY id ASC",
                (self._conversation_id,),
            )
            rows = await cursor.fetchall()

        return [{"role": row["role"], "content": row["content"]} for row in rows]

    async def clear(self) -> None:
        """Delete all messages for the current conversation."""
        await self._ensure_initialised()

        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (self._conversation_id,),
            )
            await db.commit()
