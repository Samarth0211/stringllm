"""Shared test fixtures for the StringLLM test suite."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from stringllm.providers.base import BaseLLMProvider, ProviderResponse


# ---------------------------------------------------------------------------
# MockProvider - a deterministic in-memory provider for unit tests
# ---------------------------------------------------------------------------

class MockProvider(BaseLLMProvider):
    """A deterministic LLM provider that returns predictable responses.

    The generated text always follows the pattern:
        "Mock response for: <first 50 chars of prompt>"

    This allows tests to assert on outputs without calling any real API.
    """

    def __init__(self, *, name: str = "mock", healthy: bool = True) -> None:
        self._name = name
        self._healthy = healthy

    @property
    def name(self) -> str:
        return self._name

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        text = f"Mock response for: {prompt[:50]}"
        return ProviderResponse(
            text=text,
            tokens_used=len(prompt.split()) + len(text.split()),
            model="mock-model",
            provider=self._name,
        )

    async def health_check(self) -> bool:
        return self._healthy


class FailingProvider(BaseLLMProvider):
    """A provider that always raises an exception on generate."""

    def __init__(self, *, name: str = "failing") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        raise RuntimeError(f"Provider {self._name} always fails")

    async def health_check(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_provider() -> MockProvider:
    """Return a fresh MockProvider instance."""
    return MockProvider()


@pytest.fixture
def failing_provider() -> FailingProvider:
    """Return a fresh FailingProvider instance."""
    return FailingProvider()


@pytest_asyncio.fixture
async def temp_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite database path and clean up after the test."""
    db_path = tmp_path / "test_stringllm.db"
    yield db_path
    # Cleanup: remove the DB file if it still exists
    if db_path.exists():
        db_path.unlink()
