"""Hugging Face Inference API provider for StringLLM."""

from __future__ import annotations

import os

import aiohttp

from stringllm.providers.base import BaseLLMProvider, ProviderResponse


class HuggingFaceProvider(BaseLLMProvider):
    """Provider that calls the Hugging Face Inference API.

    Requires the ``HF_API_KEY`` environment variable.
    """

    BASE_URL = "https://api-inference.huggingface.co/models"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    ) -> None:
        self._api_key = api_key or os.environ.get("HF_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Hugging Face API key is required. Pass api_key= or set the HF_API_KEY env var."
            )
        self._model = model

    @property
    def name(self) -> str:
        return "huggingface"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        url = f"{self.BASE_URL}/{self._model}"

        # Build an instruction-style prompt when a system prompt is provided
        if system_prompt:
            full_prompt = f"[INST] {system_prompt}\n\n{prompt} [/INST]"
        else:
            full_prompt = f"[INST] {prompt} [/INST]"

        payload = {
            "inputs": full_prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_full_text": False,
            },
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()

        # The HF Inference API returns a list of generated results
        if isinstance(data, list) and len(data) > 0:
            text = data[0].get("generated_text", "")
        elif isinstance(data, dict):
            text = data.get("generated_text", "")
        else:
            raise RuntimeError(f"Unexpected HuggingFace response format: {data!r}")

        # HF Inference API does not always return token counts;
        # estimate from response length when unavailable.
        tokens_used = len(full_prompt.split()) + len(text.split())

        return ProviderResponse(
            text=text,
            tokens_used=tokens_used,
            model=self._model,
            provider=self.name,
        )

    async def health_check(self) -> bool:
        url = f"{self.BASE_URL}/{self._model}"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    return resp.status == 200
        except Exception:
            return False
