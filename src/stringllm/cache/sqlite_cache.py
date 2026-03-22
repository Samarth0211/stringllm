"""SQLite-backed response cache with TTL support."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import aiosqlite


class SQLiteCache:
    """Async cache that stores LLM responses in an SQLite database.

    Cache keys are SHA-256 hashes derived from the combination of
    provider name, prompt text, temperature, and max tokens.  Entries
    expire after *ttl* seconds and are pruned lazily on :meth:`get`.

    Args:
        db_path: Path to the SQLite database file.  Defaults to
            ``~/.stringllm/cache.db``.
        ttl: Time-to-live for cache entries in seconds.  Defaults to
            ``86400`` (24 hours).
    """

    _CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        ttl: int = 86400,
    ) -> None:
        if db_path is None:
            db_path = Path.home() / ".stringllm" / "cache.db"
        self._db_path = Path(db_path)
        self._ttl = ttl
        self._initialised = False

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(
        provider_name: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Build a deterministic cache key by hashing request parameters.

        Args:
            provider_name: Name of the LLM provider (e.g. ``"openai"``).
            prompt: The full prompt text.
            temperature: Sampling temperature.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            A hex-encoded SHA-256 digest.
        """
        raw = f"{provider_name}|{prompt}|{temperature}|{max_tokens}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_initialised(self) -> None:
        if self._initialised:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(self._CREATE_TABLE_SQL)
            await db.commit()
        self._initialised = True

    async def _prune_expired(self, db: aiosqlite.Connection) -> None:
        """Delete entries older than the configured TTL."""
        cutoff = time.time() - self._ttl
        await db.execute("DELETE FROM cache WHERE created_at < ?", (cutoff,))
        await db.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, key: str) -> str | None:
        """Retrieve a cached value by *key*, returning ``None`` on miss or expiry.

        Expired entries are pruned automatically before the lookup.
        """
        await self._ensure_initialised()

        async with aiosqlite.connect(str(self._db_path)) as db:
            await self._prune_expired(db)

            cursor = await db.execute(
                "SELECT value FROM cache WHERE key = ?", (key,)
            )
            row = await cursor.fetchone()

        return row[0] if row else None

    async def set(self, key: str, value: str) -> None:
        """Store a value in the cache, replacing any existing entry for *key*."""
        await self._ensure_initialised()

        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
                (key, value, time.time()),
            )
            await db.commit()

    async def clear(self) -> None:
        """Remove all entries from the cache."""
        await self._ensure_initialised()

        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute("DELETE FROM cache")
            await db.commit()
