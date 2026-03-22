"""Groq LLM provider for StringLLM."""

from __future__ import annotations

import os

import aiohttp

from stringllm.providers.base import BaseLLMProvider, ProviderResponse


class GroqProvider(BaseLLMProvider):
    """Provider that calls the Groq OpenAI-compatible chat completions API.

    Requires the ``GROQ_API_KEY`` environment variable.
    """

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Groq API key is required. Pass api_key= or set the GROQ_API_KEY env var."
            )
        self._model = model

    @property
    def name(self) -> str:
        return "groq"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        url = f"{self.BASE_URL}/chat/completions"

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()

        choice = data["choices"][0]
        text = choice["message"]["content"]

        usage = data.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)

        return ProviderResponse(
            text=text,
            tokens_used=tokens_used,
            model=self._model,
            provider=self.name,
        )

    async def health_check(self) -> bool:
        url = f"{self.BASE_URL}/models"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    return resp.status == 200
        except Exception:
            return False
