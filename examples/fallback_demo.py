"""Demonstrate provider fallback across multiple LLM backends.

This example sets up a FallbackProvider that tries Gemini first, then Groq,
then HuggingFace. If one provider is down or misconfigured, the chain
automatically falls back to the next available provider.

Usage:
    # Set at least one API key (ideally multiple to see fallback in action):
    export GEMINI_API_KEY="your-gemini-key"
    export GROQ_API_KEY="your-groq-key"
    export HF_API_KEY="your-hf-key"
    python examples/fallback_demo.py
"""

import asyncio
import os

from stringllm import FallbackProvider, StringChain, StringNode
from stringllm.providers.base import BaseLLMProvider


def build_fallback_provider() -> FallbackProvider:
    """Build a FallbackProvider from whichever API keys are available."""
    providers: list[BaseLLMProvider] = []

    if os.getenv("GEMINI_API_KEY"):
        from stringllm import GeminiProvider

        providers.append(GeminiProvider())
        print("[+] Gemini provider added to fallback chain")

    if os.getenv("GROQ_API_KEY"):
        from stringllm import GroqProvider

        providers.append(GroqProvider())
        print("[+] Groq provider added to fallback chain")

    if os.getenv("HF_API_KEY"):
        from stringllm import HuggingFaceProvider

        providers.append(HuggingFaceProvider())
        print("[+] HuggingFace provider added to fallback chain")

    if not providers:
        raise RuntimeError(
            "No API keys found. Set at least one of: "
            "GEMINI_API_KEY, GROQ_API_KEY, HF_API_KEY"
        )

    return FallbackProvider(providers=providers, cooldown_seconds=30.0)


async def main() -> None:
    fallback = build_fallback_provider()

    # Check which providers are healthy
    print("\nChecking provider health...")
    is_healthy = await fallback.health_check()
    print(f"At least one provider healthy: {is_healthy}")
    print()

    nodes = [
        StringNode(
            name="generate",
            prompt=(
                "Write a short haiku (3 lines, 5-7-5 syllables) about: {topic}"
            ),
            output_key="haiku",
        ),
        StringNode(
            name="explain",
            prompt=(
                "Explain the imagery and meaning behind this haiku in "
                "2-3 sentences:\n\n{haiku}"
            ),
            output_key="explanation",
        ),
    ]

    chain = StringChain(nodes=nodes, provider=fallback)
    result = await chain.run(topic="resilience in software engineering")

    print("=" * 60)
    print("HAIKU")
    print("=" * 60)
    print(result.outputs["haiku"])
    print()
    print("=" * 60)
    print("EXPLANATION")
    print("=" * 60)
    print(result.outputs["explanation"])
    print()
    print(f"Provider used: {result.provider_used}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Total time: {result.total_time_ms:.0f}ms")

    # Show which provider handled each step
    for step in result.steps:
        print(f"  {step.node_name} -> {step.provider} ({step.time_ms:.0f}ms)")


if __name__ == "__main__":
    asyncio.run(main())
