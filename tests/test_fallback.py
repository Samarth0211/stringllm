"""Tests for FallbackProvider resilience logic."""

from __future__ import annotations

import pytest

from stringllm.providers.fallback import FallbackProvider

from tests.conftest import FailingProvider, MockProvider


@pytest.mark.asyncio
async def test_uses_first_healthy_provider():
    """FallbackProvider should use the first provider when it succeeds."""
    first = MockProvider(name="first")
    second = MockProvider(name="second")

    fallback = FallbackProvider(providers=[first, second])
    result = await fallback.generate(prompt="Hello")

    assert "fallback(first)" == result.provider
    assert result.text.startswith("Mock response for:")


@pytest.mark.asyncio
async def test_falls_back_on_failure():
    """When the first provider fails, the fallback should try the second."""
    first = FailingProvider(name="broken")
    second = MockProvider(name="backup")

    fallback = FallbackProvider(providers=[first, second])
    result = await fallback.generate(prompt="Hello")

    assert "fallback(backup)" == result.provider
    assert result.text.startswith("Mock response for:")


@pytest.mark.asyncio
async def test_all_providers_fail():
    """When all providers fail, FallbackProvider should raise RuntimeError."""
    first = FailingProvider(name="broken1")
    second = FailingProvider(name="broken2")

    fallback = FallbackProvider(providers=[first, second])

    with pytest.raises(RuntimeError, match="All providers failed"):
        await fallback.generate(prompt="Hello")


@pytest.mark.asyncio
async def test_health_check_any_healthy():
    """health_check returns True if at least one underlying provider is healthy."""
    healthy = MockProvider(name="healthy", healthy=True)
    unhealthy = MockProvider(name="unhealthy", healthy=False)

    fallback = FallbackProvider(providers=[unhealthy, healthy])
    result = await fallback.health_check()

    assert result is True


@pytest.mark.asyncio
async def test_health_check_all_unhealthy():
    """health_check returns False when every underlying provider is unhealthy."""
    p1 = MockProvider(name="p1", healthy=False)
    p2 = MockProvider(name="p2", healthy=False)

    fallback = FallbackProvider(providers=[p1, p2])
    result = await fallback.health_check()

    assert result is False


@pytest.mark.asyncio
async def test_fallback_requires_at_least_one_provider():
    """FallbackProvider must raise ValueError if no providers are given."""
    with pytest.raises(ValueError, match="At least one provider"):
        FallbackProvider(providers=[])
