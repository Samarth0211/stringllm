"""Tests for SQLiteCache."""

from __future__ import annotations

from pathlib import Path

import pytest

from stringllm.cache.sqlite_cache import SQLiteCache


@pytest.mark.asyncio
async def test_set_and_get(temp_db: Path):
    """Setting a value and getting it back should return the stored value."""
    cache = SQLiteCache(db_path=temp_db, ttl=3600)

    await cache.set("test_key", "test_value")
    result = await cache.get("test_key")

    assert result == "test_value"


@pytest.mark.asyncio
async def test_cache_miss_returns_none(temp_db: Path):
    """Getting a non-existent key should return None."""
    cache = SQLiteCache(db_path=temp_db, ttl=3600)

    result = await cache.get("nonexistent_key")

    assert result is None


@pytest.mark.asyncio
async def test_expired_entry_returns_none(temp_db: Path):
    """An entry with TTL=0 should be expired immediately on the next get."""
    cache = SQLiteCache(db_path=temp_db, ttl=0)

    await cache.set("expiring_key", "expiring_value")
    # With TTL=0, the entry is already expired by the time we call get
    result = await cache.get("expiring_key")

    assert result is None


@pytest.mark.asyncio
async def test_clear(temp_db: Path):
    """Clearing the cache should remove all entries."""
    cache = SQLiteCache(db_path=temp_db, ttl=3600)

    await cache.set("key1", "value1")
    await cache.set("key2", "value2")

    await cache.clear()

    assert await cache.get("key1") is None
    assert await cache.get("key2") is None


@pytest.mark.asyncio
async def test_overwrite_existing_key(temp_db: Path):
    """Setting the same key twice should overwrite the previous value."""
    cache = SQLiteCache(db_path=temp_db, ttl=3600)

    await cache.set("key", "original")
    await cache.set("key", "updated")

    result = await cache.get("key")
    assert result == "updated"


@pytest.mark.asyncio
async def test_make_key_deterministic():
    """SQLiteCache.make_key should produce consistent hashes for the same inputs."""
    key1 = SQLiteCache.make_key("gemini", "Hello world", 0.7, 1024)
    key2 = SQLiteCache.make_key("gemini", "Hello world", 0.7, 1024)
    key3 = SQLiteCache.make_key("groq", "Hello world", 0.7, 1024)

    assert key1 == key2
    assert key1 != key3
