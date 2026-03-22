# StringLLM

**Chain free LLMs together like strings to solve complex tasks.**

StringLLM is a lightweight, async-first Python framework for chaining LLM calls into multi-step pipelines. Each step is a **StringNode** that takes a prompt template, calls an LLM provider, and passes its output to the next node in the **StringChain**.

## Key Features

- **Provider agnostic** -- built-in support for Gemini, Groq, and Hugging Face with a simple base class for adding your own.
- **Automatic fallback** -- `FallbackProvider` tries providers in order and gracefully recovers from failures.
- **Prompt templates** -- reusable `PromptTemplate` objects with variable substitution and a `PromptLibrary` of common tasks.
- **Memory backends** -- in-memory buffer and SQLite-backed conversation history.
- **Response caching** -- SQLite cache with TTL to avoid redundant LLM calls.
- **Fully async** -- built on `aiohttp` and `aiosqlite` for non-blocking I/O throughout.
- **Playground UI** -- a FastAPI server with a web interface for building and testing chains interactively.

## Quick Example

```python
import asyncio
from stringllm import GeminiProvider, StringChain, StringNode

async def main():
    provider = GeminiProvider()  # uses GEMINI_API_KEY env var

    chain = StringChain(
        nodes=[
            StringNode(
                name="summarize",
                prompt="Summarize this text:\n\n{text}",
                output_key="summary",
            ),
            StringNode(
                name="sentiment",
                prompt="What is the sentiment of:\n\n{summary}",
                output_key="sentiment",
            ),
        ],
        provider=provider,
    )

    result = await chain.run(text="Python 3.12 brings exciting new features...")
    print(result.outputs["summary"])
    print(result.outputs["sentiment"])

asyncio.run(main())
```

## Installation

```bash
pip install stringllm
```

Or install from source for development:

```bash
git clone https://github.com/samarthbhamare/stringllm.git
cd stringllm
pip install -e ".[dev]"
```

## License

MIT License. See [LICENSE](https://github.com/samarthbhamare/stringllm/blob/main/LICENSE) for details.
