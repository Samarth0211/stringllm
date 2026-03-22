"""Base LLM provider interface for StringLLM."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderResponse:
    """Standard response returned by all LLM providers."""

    text: str
    tokens_used: int
    model: str
    provider: str


class BaseLLMProvider(ABC):
    """Abstract base class that every LLM provider must implement."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        """Generate a completion for the given prompt."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and operational."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique human-readable name for this provider."""
        ...
