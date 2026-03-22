"""Provider status API routes."""

from __future__ import annotations

import os

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/providers", tags=["providers"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ProviderStatus(BaseModel):
    name: str
    available: bool
    healthy: bool


class ProviderStatusList(BaseModel):
    providers: list[ProviderStatus]


# ---------------------------------------------------------------------------
# Provider key / health mapping
# ---------------------------------------------------------------------------

_PROVIDER_ENV_KEYS: dict[str, str] = {
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
}


async def _check_provider_health(name: str) -> bool:
    """Try to instantiate the provider and call its health_check method.

    Returns False on any error so the status endpoint never crashes.
    """
    try:
        if name == "gemini":
            from stringllm.providers.gemini import GeminiProvider

            provider = GeminiProvider(api_key=os.environ["GOOGLE_API_KEY"])
        elif name == "groq":
            from stringllm.providers.groq import GroqProvider

            provider = GroqProvider(api_key=os.environ["GROQ_API_KEY"])
        elif name == "huggingface":
            from stringllm.providers.huggingface import HuggingFaceProvider

            provider = HuggingFaceProvider(api_key=os.environ["HUGGINGFACE_API_KEY"])
        else:
            return False

        return await provider.health_check()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/status", response_model=ProviderStatusList)
async def get_provider_status() -> ProviderStatusList:
    """Return availability and health for every known provider.

    A provider is *available* when its API key environment variable is set.
    A provider is *healthy* when it is available **and** its ``health_check``
    method returns ``True``.
    """
    statuses: list[ProviderStatus] = []

    for name, env_key in _PROVIDER_ENV_KEYS.items():
        api_key = os.getenv(env_key, "").strip()
        available = bool(api_key)

        healthy = False
        if available:
            healthy = await _check_provider_health(name)

        statuses.append(ProviderStatus(name=name, available=available, healthy=healthy))

    # Fallback is available when at least one underlying provider has a key
    any_available = any(s.available for s in statuses)
    any_healthy = any(s.healthy for s in statuses)
    statuses.append(
        ProviderStatus(name="fallback", available=any_available, healthy=any_healthy)
    )

    return ProviderStatusList(providers=statuses)
