"""Simple two-step chain: summarize then analyze sentiment.

Usage:
    export GEMINI_API_KEY="your-api-key"
    python examples/simple_chain.py
"""

import asyncio

from stringllm import GeminiProvider, StringChain, StringNode


async def main() -> None:
    provider = GeminiProvider()

    nodes = [
        StringNode(
            name="summarize",
            prompt=(
                "Summarize the following text into 3 concise bullet points:\n\n{text}"
            ),
            output_key="summary",
        ),
        StringNode(
            name="sentiment",
            prompt=(
                "Analyze the sentiment of the following summary. "
                "Classify it as positive, negative, or neutral and explain why:\n\n{summary}"
            ),
            output_key="sentiment_analysis",
        ),
    ]

    chain = StringChain(nodes=nodes, provider=provider)

    result = await chain.run(
        text=(
            "Python 3.12 introduces several exciting features including improved error "
            "messages, a new type parameter syntax, and significant performance improvements "
            "through the specializing adaptive interpreter. The community has responded "
            "enthusiastically, with many developers praising the continued focus on developer "
            "experience and runtime speed. However, some library maintainers have expressed "
            "concern about the pace of deprecations in the standard library."
        )
    )

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(result.outputs["summary"])
    print()
    print("=" * 60)
    print("SENTIMENT ANALYSIS")
    print("=" * 60)
    print(result.outputs["sentiment_analysis"])
    print()
    print(f"Total tokens used: {result.total_tokens}")
    print(f"Total time: {result.total_time_ms:.0f}ms")
    print(f"Provider: {result.provider_used}")
    print(f"Steps: {len(result.steps)}")


if __name__ == "__main__":
    asyncio.run(main())
