"""StringNode - a single step in a StringChain."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from stringllm.core.result import StepResult

if TYPE_CHECKING:
    from stringllm.providers.base import BaseLLMProvider


class StringNode:
    """Represents a single LLM call within a chain.

    The *prompt* is a Python format-style template string containing
    ``{variable}`` placeholders that are resolved at run time from the
    accumulated chain context.
    """

    def __init__(
        self,
        *,
        name: str,
        prompt: str,
        output_key: str,
        provider: BaseLLMProvider | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        self._name = name
        self._prompt = prompt
        self._output_key = output_key
        self._provider = provider
        self._temperature = temperature
        self._max_tokens = max_tokens

    # -- public read-only properties ------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def output_key(self) -> str:
        return self._output_key

    @property
    def provider(self) -> BaseLLMProvider | None:
        return self._provider

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    # -- execution ------------------------------------------------------------

    async def run(
        self,
        inputs: dict[str, str],
        provider: BaseLLMProvider,
    ) -> StepResult:
        """Render the prompt template with *inputs* and call the provider.

        If the node was constructed with its own ``provider`` override, that
        provider is used instead of the one passed in.

        Returns a :class:`StepResult` with timing and token information.
        """
        effective_provider = self._provider if self._provider is not None else provider

        rendered_prompt = self._prompt.format(**inputs)

        start = time.perf_counter()
        response = await effective_provider.generate(
            prompt=rendered_prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return StepResult(
            node_name=self._name,
            output=response.text,
            tokens=response.tokens_used,
            time_ms=elapsed_ms,
            provider=effective_provider.name,
        )
