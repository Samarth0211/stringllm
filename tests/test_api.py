"""Tests for FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from stringllm.providers.base import ProviderResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_provider_response() -> ProviderResponse:
    return ProviderResponse(
        text="Mocked LLM output",
        tokens_used=42,
        model="mock-model",
        provider="gemini",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_chain_endpoint(monkeypatch: pytest.MonkeyPatch):
    """POST /api/chain/run should execute a chain and return results."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    from server.app import app

    mock_generate = AsyncMock(return_value=_mock_provider_response())
    mock_health = AsyncMock(return_value=True)

    with (
        patch("stringllm.providers.gemini.GeminiProvider.generate", mock_generate),
        patch("stringllm.providers.gemini.GeminiProvider.health_check", mock_health),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/chain/run",
                json={
                    "nodes": [
                        {
                            "name": "step1",
                            "prompt": "Summarize: {text}",
                            "output_key": "summary",
                        }
                    ],
                    "input": {"text": "Some long text to summarize."},
                    "provider": "gemini",
                },
            )

    assert response.status_code == 200
    data = response.json()
    assert "outputs" in data
    assert "summary" in data["outputs"]
    assert data["total_tokens"] >= 0
    assert len(data["steps"]) == 1


@pytest.mark.asyncio
async def test_providers_status_endpoint(monkeypatch: pytest.MonkeyPatch):
    """GET /api/providers/status should return provider availability info."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)

    from server.app import app

    mock_health = AsyncMock(return_value=True)

    with patch("stringllm.providers.gemini.GeminiProvider.health_check", mock_health):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/providers/status")

    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    providers = {p["name"]: p for p in data["providers"]}
    assert "gemini" in providers
    assert "groq" in providers
    assert "huggingface" in providers
    assert "fallback" in providers
    assert providers["gemini"]["available"] is True


@pytest.mark.asyncio
async def test_templates_endpoint():
    """GET /api/templates should return available prompt templates."""
    from server.app import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/templates")

    assert response.status_code == 200
    data = response.json()
    assert "templates" in data
    template_names = [t["name"] for t in data["templates"]]
    assert "summarize" in template_names
    assert "translate" in template_names
    # Each template should have variables
    for tpl in data["templates"]:
        assert "variables" in tpl
        assert isinstance(tpl["variables"], list)


@pytest.mark.asyncio
async def test_root_serves_html():
    """GET / should serve the playground HTML page."""
    from server.app import app

    # Create a minimal index.html for the test
    import tempfile
    from pathlib import Path
    from unittest.mock import patch as sync_patch

    static_dir = Path(tempfile.mkdtemp())
    index_html = static_dir / "index.html"
    index_html.write_text(
        "<!DOCTYPE html><html><body><h1>StringLLM Playground</h1></body></html>"
    )

    with sync_patch("server.routes.playground.STATIC_DIR", static_dir):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "StringLLM" in response.text

    # Cleanup
    index_html.unlink()
    static_dir.rmdir()
