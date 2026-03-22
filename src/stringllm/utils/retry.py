"""Async retry decorator with exponential backoff and jitter."""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from typing import Any, Callable, Tuple, Type, TypeVar

import aiohttp

logger = logging.getLogger("stringllm")

F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: Tuple[Type[BaseException], ...] = (
        aiohttp.ClientError,
        asyncio.TimeoutError,
    ),
) -> Callable[[F], F]:
    """Decorator that retries an async function on specified exceptions.

    Uses exponential backoff with full jitter::

        delay = random.uniform(0, min(max_delay, base_delay * 2 ** attempt))

    Args:
        max_retries: Maximum number of retry attempts (in addition to the
            initial call).
        base_delay: Base delay in seconds for the backoff calculation.
        max_delay: Upper bound for the computed delay.
        retry_on: Tuple of exception types that should trigger a retry.

    Returns:
        A decorator that wraps an async callable with retry logic.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: BaseException | None = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_on as exc:
                    last_exception = exc

                    if attempt == max_retries:
                        logger.error(
                            "All %d retries exhausted for %s: %s",
                            max_retries,
                            func.__qualname__,
                            exc,
                        )
                        raise

                    delay = random.uniform(
                        0, min(max_delay, base_delay * (2 ** attempt))
                    )
                    logger.warning(
                        "Retry %d/%d for %s after %.2fs — %s: %s",
                        attempt + 1,
                        max_retries,
                        func.__qualname__,
                        delay,
                        type(exc).__name__,
                        exc,
                    )
                    await asyncio.sleep(delay)

            # Should be unreachable, but keeps type checkers happy.
            raise last_exception  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
