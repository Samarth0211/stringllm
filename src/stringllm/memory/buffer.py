"""In-memory conversation buffer with configurable max size."""

from __future__ import annotations

from collections import deque

from stringllm.memory.base import BaseMemory


class BufferMemory(BaseMemory):
    """A simple in-memory conversation store backed by a :class:`collections.deque`.

    When the buffer reaches *max_size* messages the oldest entries are
    automatically discarded.

    Args:
        max_size: Maximum number of messages to retain.  Defaults to ``20``.
    """

    def __init__(self, max_size: int = 20) -> None:
        self._max_size = max_size
        self._messages: deque[dict[str, str]] = deque(maxlen=max_size)

    async def add(self, role: str, content: str) -> None:
        """Append a message, dropping the oldest if the buffer is full."""
        self._messages.append({"role": role, "content": content})

    async def get_history(self) -> list[dict[str, str]]:
        """Return a snapshot of the current conversation history."""
        return list(self._messages)

    async def clear(self) -> None:
        """Remove all messages from the buffer."""
        self._messages.clear()
