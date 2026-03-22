"""Abstract base class for all memory backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseMemory(ABC):
    """Interface that every memory backend must implement.

    All methods are async to allow for I/O-bound implementations (e.g. database
    or network-backed stores) without blocking the event loop.
    """

    @abstractmethod
    async def add(self, role: str, content: str) -> None:
        """Append a message to the conversation history.

        Args:
            role: The speaker role (e.g. ``"user"``, ``"assistant"``, ``"system"``).
            content: The message text.
        """

    @abstractmethod
    async def get_history(self) -> list[dict[str, str]]:
        """Return the full conversation history.

        Returns:
            A list of dicts, each containing ``"role"`` and ``"content"`` keys.
        """

    @abstractmethod
    async def clear(self) -> None:
        """Remove all messages from the conversation history."""
