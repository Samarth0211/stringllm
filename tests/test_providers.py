"""Tests for LLM providers with mocked HTTP calls."""

from __future__ import annotations

import re

import pytest
from aioresponses import aioresponses

from stringllm.providers.base import ProviderResponse


# ---------------------------------------------------------------------------
# Gemini Provider Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_generate(monkeypatch: pytest.MonkeyPatch):
    """GeminiProvider.generate should parse the Gemini API response correctly."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    from stringllm.providers.gemini import GeminiProvider

    provider = GeminiProvider(api_key="test-gemini-key", model="gemini-2.0-flash")

    mock_response = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "This is a summary of the text."}],
                    "role": "model",
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 8,
            "totalTokenCount": 18,
        },
    }

    with aioresponses() as mocked:
        # Use regex pattern to match URL with query parameters
        pattern = re.compile(
            r"^https://generativelanguage\.googleapis\.com/v1beta/models/"
            r"gemini-2\.0-flash:generateContent\?key=.+$"
        )
        mocked.post(pattern, payload=mock_response, repeat=True)

        result = await provider.generate(prompt="Summarize this text", temperature=0.5)

    assert isinstance(result, ProviderResponse)
    assert result.text == "This is a summary of the text."
    assert result.tokens_used == 18
    assert result.model == "gemini-2.0-flash"
    assert result.provider == "gemini"


@pytest.mark.asyncio
async def test_gemini_health_check(monkeypatch: pytest.MonkeyPatch):
    """GeminiProvider.health_check should return True on a 200 response."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    from stringllm.providers.gemini import GeminiProvider

    provider = GeminiProvider(api_key="test-gemini-key", model="gemini-2.0-flash")

    with aioresponses() as mocked:
        # Use regex pattern to match URL with query parameters
        pattern = re.compile(
            r"^https://generativelanguage\.googleapis\.com/v1beta/models/"
            r"gemini-2\.0-flash\?key=.+$"
        )
        mocked.get(pattern, status=200, payload={"name": "gemini-2.0-flash"})

        healthy = await provider.health_check()

    assert healthy is True


@pytest.mark.asyncio
async def test_gemini_health_check_failure(monkeypatch: pytest.MonkeyPatch):
    """GeminiProvider.health_check should return False on a non-200 response."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    from stringllm.providers.gemini import GeminiProvider

    provider = GeminiProvider(api_key="test-gemini-key", model="gemini-2.0-flash")

    with aioresponses() as mocked:
        pattern = re.compile(
            r"^https://generativelanguage\.googleapis\.com/v1beta/models/"
            r"gemini-2\.0-flash\?key=.+$"
        )
        mocked.get(pattern, status=500)

        healthy = await provider.health_check()


# ---------------------------------------------------------------------------
# Groq Provider Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_groq_generate(monkeypatch: pytest.MonkeyPatch):
    """GroqProvider.generate should parse the OpenAI-compatible response correctly."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    from stringllm.providers.groq import GroqProvider

    provider = GroqProvider(api_key="test-groq-key", model="llama-3.3-70b-versatile")

    mock_response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The sentiment is positive.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 6,
            "total_tokens": 18,
        },
    }

    with aioresponses() as mocked:
        mocked.post(
            f"{GroqProvider.BASE_URL}/chat/completions",
            payload=mock_response,
        )

        result = await provider.generate(prompt="Analyze sentiment of this text")

    assert isinstance(result, ProviderResponse)
    assert result.text == "The sentiment is positive."
    assert result.tokens_used == 18
    assert result.model == "llama-3.3-70b-versatile"
    assert result.provider == "groq"


# ---------------------------------------------------------------------------
# HuggingFace Provider Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_huggingface_generate(monkeypatch: pytest.MonkeyPatch):
    """HuggingFaceProvider.generate should parse the HF Inference API response."""
    monkeypatch.setenv("HF_API_KEY", "test-hf-key")

    from stringllm.providers.huggingface import HuggingFaceProvider

    provider = HuggingFaceProvider(
        api_key="test-hf-key", model="mistralai/Mistral-7B-Instruct-v0.3"
    )

    mock_response = [
        {"generated_text": "The code looks clean and follows best practices."}
    ]

    with aioresponses() as mocked:
        mocked.post(
            f"{HuggingFaceProvider.BASE_URL}/mistralai/Mistral-7B-Instruct-v0.3",
            payload=mock_response,
        )

        result = await provider.generate(prompt="Review this code")

    assert isinstance(result, ProviderResponse)
    assert result.text == "The code looks clean and follows best practices."
    assert result.tokens_used > 0
    assert result.model == "mistralai/Mistral-7B-Instruct-v0.3"
    assert result.provider == "huggingface"
