"""Summarize text and then translate the summary to Spanish.

Usage:
    export GEMINI_API_KEY="your-api-key"
    python examples/summarize_and_translate.py
"""

import asyncio

from stringllm import GeminiProvider, StringChain, StringNode


async def main() -> None:
    provider = GeminiProvider()

    nodes = [
        StringNode(
            name="summarize",
            prompt=(
                "Summarize the following article into a short paragraph "
                "(3-4 sentences maximum):\n\n{article}"
            ),
            output_key="english_summary",
        ),
        StringNode(
            name="translate",
            prompt=(
                "Translate the following English text into Spanish. "
                "Provide only the translation, no explanations:\n\n{english_summary}"
            ),
            output_key="spanish_summary",
        ),
    ]

    chain = StringChain(nodes=nodes, provider=provider)

    article = (
        "Artificial intelligence has transformed the healthcare industry in recent years. "
        "Machine learning models can now detect diseases from medical images with accuracy "
        "rivaling human specialists. Natural language processing enables automated analysis "
        "of electronic health records, helping doctors identify patterns across thousands "
        "of patient histories. Robotic surgery systems powered by AI assist surgeons in "
        "performing minimally invasive procedures with unprecedented precision. Despite "
        "these advances, challenges remain around data privacy, algorithmic bias, and the "
        "need for regulatory frameworks that keep pace with technological innovation."
    )

    result = await chain.run(article=article)

    print("=" * 60)
    print("ORIGINAL ARTICLE")
    print("=" * 60)
    print(article)
    print()
    print("=" * 60)
    print("ENGLISH SUMMARY")
    print("=" * 60)
    print(result.outputs["english_summary"])
    print()
    print("=" * 60)
    print("SPANISH TRANSLATION")
    print("=" * 60)
    print(result.outputs["spanish_summary"])
    print()
    print(f"Total tokens: {result.total_tokens}")
    print(f"Total time: {result.total_time_ms:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
