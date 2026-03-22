"""FallbackProvider - resilient multi-provider wrapper for StringLLM."""

from __future__ import annotations

import time

from stringllm.providers.base import BaseLLMProvider, ProviderResponse


class FallbackProvider(BaseLLMProvider):
    """Wraps an ordered list of providers and tries each until one succeeds.

    Providers that fail are temporarily deprioritised for a configurable
    cooldown period (default 60 seconds).  After the cooldown expires the
    provider is retried in its original priority position.

    Parameters
    ----------
    providers:
        Ordered list of providers to attempt (highest priority first).
    cooldown_seconds:
        How long (in seconds) to deprioritise a provider after a failure.
    """

    def __init__(
        self,
        *,
        providers: list[BaseLLMProvider],
        cooldown_seconds: float = 60.0,
    ) -> None:
        if not providers:
            raise ValueError("At least one provider is required.")
        self._providers = providers
        self._cooldown_seconds = cooldown_seconds
        # Maps provider name -> timestamp of last failure
        self._failures: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "fallback"

    def _sorted_providers(self) -> list[BaseLLMProvider]:
        """Return providers sorted so that those in cooldown come last."""
        now = time.monotonic()

        def _sort_key(p: BaseLLMProvider) -> int:
            failed_at = self._failures.get(p.name)
            if failed_at is not None and (now - failed_at) < self._cooldown_seconds:
                return 1  # deprioritised
            return 0  # healthy / cooldown expired

        return sorted(self._providers, key=_sort_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        errors: list[tuple[str, Exception]] = []

        for provider in self._sorted_providers():
            try:
                response = await provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # Clear failure record on success
                self._failures.pop(provider.name, None)
                return ProviderResponse(
                    text=response.text,
                    tokens_used=response.tokens_used,
                    model=response.model,
                    provider=f"fallback({provider.name})",
                )
            except Exception as exc:  # noqa: BLE001
                self._failures[provider.name] = time.monotonic()
                errors.append((provider.name, exc))

        # Every provider failed
        details = "; ".join(f"{name}: {exc}" for name, exc in errors)
        raise RuntimeError(f"All providers failed. Details: {details}")

    async def health_check(self) -> bool:
        """Return True if *any* underlying provider passes its health check."""
        for provider in self._providers:
            try:
                if await provider.health_check():
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False
