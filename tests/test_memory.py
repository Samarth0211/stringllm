"""Tests for BufferMemory and SQLiteMemory."""

from __future__ import annotations

from pathlib import Path

import pytest

from stringllm.memory.buffer import BufferMemory
from stringllm.memory.sqlite import SQLiteMemory


# ---------------------------------------------------------------------------
# BufferMemory Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_buffer_add_and_get():
    """Adding messages and retrieving history should work correctly."""
    mem = BufferMemory(max_size=10)

    await mem.add("user", "Hello")
    await mem.add("assistant", "Hi there!")
    await mem.add("user", "How are you?")

    history = await mem.get_history()

    assert len(history) == 3
    assert history[0] == {"role": "user", "content": "Hello"}
    assert history[1] == {"role": "assistant", "content": "Hi there!"}
    assert history[2] == {"role": "user", "content": "How are you?"}


@pytest.mark.asyncio
async def test_buffer_max_size_eviction():
    """When the buffer exceeds max_size, the oldest messages should be evicted."""
    mem = BufferMemory(max_size=3)

    await mem.add("user", "Message 1")
    await mem.add("assistant", "Reply 1")
    await mem.add("user", "Message 2")
    await mem.add("assistant", "Reply 2")  # This should evict "Message 1"

    history = await mem.get_history()

    assert len(history) == 3
    # "Message 1" should have been evicted
    assert history[0] == {"role": "assistant", "content": "Reply 1"}
    assert history[1] == {"role": "user", "content": "Message 2"}
    assert history[2] == {"role": "assistant", "content": "Reply 2"}


@pytest.mark.asyncio
async def test_buffer_clear():
    """Clearing the buffer should remove all messages."""
    mem = BufferMemory()

    await mem.add("user", "Hello")
    await mem.add("assistant", "Hi")

    await mem.clear()
    history = await mem.get_history()

    assert history == []


# ---------------------------------------------------------------------------
# SQLiteMemory Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sqlite_add_and_get(temp_db: Path):
    """Adding messages and retrieving history from SQLite should work."""
    mem = SQLiteMemory(db_path=temp_db, conversation_id="test_conv")

    await mem.add("user", "Hello from SQLite")
    await mem.add("assistant", "SQLite says hi!")

    history = await mem.get_history()

    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hello from SQLite"}
    assert history[1] == {"role": "assistant", "content": "SQLite says hi!"}


@pytest.mark.asyncio
async def test_sqlite_conversation_isolation(temp_db: Path):
    """Messages in different conversations should be isolated."""
    conv_a = SQLiteMemory(db_path=temp_db, conversation_id="conv_a")
    conv_b = SQLiteMemory(db_path=temp_db, conversation_id="conv_b")

    await conv_a.add("user", "Message for A")
    await conv_b.add("user", "Message for B")
    await conv_a.add("assistant", "Reply in A")

    history_a = await conv_a.get_history()
    history_b = await conv_b.get_history()

    assert len(history_a) == 2
    assert len(history_b) == 1
    assert history_a[0]["content"] == "Message for A"
    assert history_a[1]["content"] == "Reply in A"
    assert history_b[0]["content"] == "Message for B"


@pytest.mark.asyncio
async def test_sqlite_clear(temp_db: Path):
    """Clearing SQLiteMemory should remove all messages for that conversation."""
    mem = SQLiteMemory(db_path=temp_db, conversation_id="clear_test")

    await mem.add("user", "Will be cleared")
    await mem.add("assistant", "Also cleared")

    await mem.clear()
    history = await mem.get_history()

    assert history == []
