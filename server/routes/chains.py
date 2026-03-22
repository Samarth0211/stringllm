"""Chain execution API routes."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from stringllm.core.chain import StringChain
from stringllm.core.node import StringNode
from stringllm.core.result import ChainResult

router = APIRouter(prefix="/api/chain", tags=["chain"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class NodeSpec(BaseModel):
    """Specification for a single node within a chain."""

    name: str
    prompt: str
    output_key: str


class RunChainRequest(BaseModel):
    """Body for POST /api/chain/run."""

    nodes: list[NodeSpec] = Field(..., min_length=1)
    input: dict[str, str] = Field(default_factory=dict)
    provider: str = "gemini"


class StepResultResponse(BaseModel):
    node_name: str
    output: str
    tokens: int
    time_ms: float
    provider: str


class ChainResultResponse(BaseModel):
    outputs: dict[str, str]
    total_tokens: int
    total_time_ms: float
    provider_used: str
    steps: list[StepResultResponse]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_provider(name: str) -> Any:
    """Instantiate the requested LLM provider by name.

    Imports are deferred so the server module does not hard-depend on every
    provider SDK at import time.
    """
    name_lower = name.lower().strip()

    if name_lower == "gemini":
        from stringllm.providers.gemini import GeminiProvider

        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="GOOGLE_API_KEY environment variable is not set.",
            )
        return GeminiProvider(api_key=api_key)

    if name_lower == "groq":
        from stringllm.providers.groq import GroqProvider

        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="GROQ_API_KEY environment variable is not set.",
            )
        return GroqProvider(api_key=api_key)

    if name_lower == "huggingface":
        from stringllm.providers.huggingface import HuggingFaceProvider

        api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="HUGGINGFACE_API_KEY environment variable is not set.",
            )
        return HuggingFaceProvider(api_key=api_key)

    if name_lower == "fallback":
        from stringllm.providers.fallback import FallbackProvider

        providers = []
        # Build fallback list from whichever keys are available
        if os.getenv("GOOGLE_API_KEY"):
            from stringllm.providers.gemini import GeminiProvider

            providers.append(GeminiProvider(api_key=os.environ["GOOGLE_API_KEY"]))
        if os.getenv("GROQ_API_KEY"):
            from stringllm.providers.groq import GroqProvider

            providers.append(GroqProvider(api_key=os.environ["GROQ_API_KEY"]))
        if os.getenv("HUGGINGFACE_API_KEY"):
            from stringllm.providers.huggingface import HuggingFaceProvider

            providers.append(HuggingFaceProvider(api_key=os.environ["HUGGINGFACE_API_KEY"]))

        if not providers:
            raise HTTPException(
                status_code=400,
                detail="No API keys are set for any provider. Set at least one of "
                "GOOGLE_API_KEY, GROQ_API_KEY, or HUGGINGFACE_API_KEY.",
            )
        return FallbackProvider(providers=providers)

    raise HTTPException(
        status_code=400,
        detail=f"Unknown provider '{name}'. Choose from: gemini, groq, huggingface, fallback.",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/run", response_model=ChainResultResponse)
async def run_chain(body: RunChainRequest) -> ChainResultResponse:
    """Build a StringChain from the request body, execute it, and return results."""
    try:
        provider = _build_provider(body.provider)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to initialize provider: {exc}") from exc

    nodes = [
        StringNode(
            name=n.name,
            prompt=n.prompt,
            output_key=n.output_key,
        )
        for n in body.nodes
    ]

    chain = StringChain(nodes=nodes, provider=provider)

    try:
        result: ChainResult = await chain.run(**body.input)
    except KeyError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Missing input variable: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chain execution failed: {exc}",
        ) from exc

    return ChainResultResponse(
        outputs=result.outputs,
        total_tokens=result.total_tokens,
        total_time_ms=result.total_time_ms,
        provider_used=result.provider_used,
        steps=[
            StepResultResponse(
                node_name=s.node_name,
                output=s.output,
                tokens=s.tokens,
                time_ms=s.time_ms,
                provider=s.provider,
            )
            for s in result.steps
        ],
    )
