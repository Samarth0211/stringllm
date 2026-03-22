"""StringChain - sequential execution of StringNode instances."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from stringllm.core.result import ChainResult, StepResult

if TYPE_CHECKING:
    from stringllm.core.node import StringNode
    from stringllm.providers.base import BaseLLMProvider


# ---------------------------------------------------------------------------
# Lightweight protocols so chain.py doesn't depend on concrete memory / cache
# ---------------------------------------------------------------------------

@runtime_checkable
class MemoryProtocol(Protocol):
    """Minimal interface that any memory backend must satisfy."""

    async def add(self, role: str, content: str) -> None: ...
    async def get_history(self) -> list[dict[str, str]]: ...


@runtime_checkable
class CacheProtocol(Protocol):
    """Minimal interface that any cache backend must satisfy."""

    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, **kwargs: Any) -> None: ...


def _cache_key(prompt: str, provider_name: str, temperature: float, max_tokens: int) -> str:
    """Deterministic cache key for a specific prompt + generation params."""
    blob = json.dumps(
        {"prompt": prompt, "provider": provider_name, "temperature": temperature, "max_tokens": max_tokens},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class StringChain:
    """Execute an ordered sequence of :class:`StringNode` instances.

    Each node's output is stored under its ``output_key`` and becomes
    available as an input variable for subsequent nodes.

    Parameters
    ----------
    nodes:
        Ordered list of nodes to execute.
    provider:
        Default LLM provider used when a node does not specify its own.
    memory:
        Optional memory backend. When supplied, each user prompt and
        assistant response are stored after every node execution.
    cache:
        Optional cache backend. When supplied, the rendered prompt is
        hashed and looked up before calling the LLM; hits skip the
        provider entirely.
    """

    def __init__(
        self,
        *,
        nodes: list[StringNode],
        provider: BaseLLMProvider,
        memory: MemoryProtocol | None = None,
        cache: CacheProtocol | None = None,
    ) -> None:
        self._nodes = nodes
        self._provider = provider
        self._memory = memory
        self._cache = cache

    async def run(self, **kwargs: str) -> ChainResult:
        """Run every node in order and return an aggregate :class:`ChainResult`."""
        context: dict[str, str] = dict(kwargs)
        steps: list[StepResult] = []
        outputs: dict[str, str] = {}
        total_tokens = 0
        total_time_ms = 0.0
        last_provider = self._provider.name

        for node in self._nodes:
            effective_provider = node.provider if node.provider is not None else self._provider
            rendered_prompt = node.prompt.format(**context)

            # --- cache lookup ------------------------------------------------
            cached_text: str | None = None
            cache_k: str | None = None
            if self._cache is not None:
                cache_k = _cache_key(
                    rendered_prompt,
                    effective_provider.name,
                    node.temperature,
                    node.max_tokens,
                )
                cached_text = await self._cache.get(cache_k)

            if cached_text is not None:
                step = StepResult(
                    node_name=node.name,
                    output=cached_text,
                    tokens=0,
                    time_ms=0.0,
                    provider=f"{effective_provider.name} (cached)",
                )
            else:
                step = await node.run(inputs=context, provider=self._provider)

                # store in cache
                if self._cache is not None and cache_k is not None:
                    await self._cache.set(cache_k, step.output)

            # --- memory ------------------------------------------------------
            if self._memory is not None:
                await self._memory.add("user", rendered_prompt)
                await self._memory.add("assistant", step.output)

            # --- accumulate --------------------------------------------------
            context[node.output_key] = step.output
            outputs[node.output_key] = step.output
            steps.append(step)
            total_tokens += step.tokens
            total_time_ms += step.time_ms
            last_provider = step.provider

        return ChainResult(
            outputs=outputs,
            total_tokens=total_tokens,
            total_time_ms=total_time_ms,
            provider_used=last_provider,
            steps=steps,
        )
