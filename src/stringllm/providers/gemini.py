"""Google Gemini LLM provider for StringLLM."""

from __future__ import annotations

import os

import aiohttp

from stringllm.providers.base import BaseLLMProvider, ProviderResponse


class GeminiProvider(BaseLLMProvider):
    """Provider that calls the Google Gemini REST API.

    Requires the ``GEMINI_API_KEY`` environment variable.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
    ) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Gemini API key is required. Pass api_key= or set the GEMINI_API_KEY env var."
            )
        self._model = model

    @property
    def name(self) -> str:
        return "gemini"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        url = f"{self.BASE_URL}/models/{self._model}:generateContent?key={self._api_key}"

        contents: list[dict] = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": system_prompt}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()

        # Parse Gemini response format
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no candidates.")

        text = candidates[0]["content"]["parts"][0]["text"]

        # Token usage metadata
        usage = data.get("usageMetadata", {})
        tokens_used = usage.get("totalTokenCount", 0)

        return ProviderResponse(
            text=text,
            tokens_used=tokens_used,
            model=self._model,
            provider=self.name,
        )

    async def health_check(self) -> bool:
        url = f"{self.BASE_URL}/models/{self._model}?key={self._api_key}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    return resp.status == 200
        except Exception:
            return False
