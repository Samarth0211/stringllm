# Quickstart

Get up and running with StringLLM in under five minutes.

## 1. Install

```bash
pip install stringllm
```

Or from source:

```bash
git clone https://github.com/samarthbhamare/stringllm.git
cd stringllm
pip install -e ".[dev]"
```

## 2. Set your API key

StringLLM supports multiple providers. Set the environment variable for the one you want to use:

=== "Gemini"

    ```bash
    export GEMINI_API_KEY="your-gemini-api-key"
    ```

=== "Groq"

    ```bash
    export GROQ_API_KEY="your-groq-api-key"
    ```

=== "Hugging Face"

    ```bash
    export HF_API_KEY="your-huggingface-api-key"
    ```

## 3. Build your first chain

```python
import asyncio
from stringllm import GeminiProvider, StringChain, StringNode

async def main():
    provider = GeminiProvider()

    chain = StringChain(
        nodes=[
            StringNode(
                name="summarize",
                prompt="Summarize into 3 bullet points:\n\n{text}",
                output_key="summary",
            ),
            StringNode(
                name="translate",
                prompt="Translate to French:\n\n{summary}",
                output_key="french_summary",
            ),
        ],
        provider=provider,
    )

    result = await chain.run(
        text="StringLLM is a Python framework for chaining LLM calls."
    )

    print(result.outputs["summary"])
    print(result.outputs["french_summary"])
    print(f"Total tokens: {result.total_tokens}")
    print(f"Time: {result.total_time_ms:.0f}ms")

asyncio.run(main())
```

## 4. Use prompt templates

Instead of writing prompts inline, use the built-in `PromptLibrary`:

```python
from stringllm import PromptLibrary

tpl = PromptLibrary.summarize()
rendered = tpl.render(text="Your long article here...")
print(rendered)
# "Summarize the following text into 3 bullet points:\n\nYour long article here..."
```

## 5. Add memory

Keep conversation history across chain runs:

```python
from stringllm import StringChain, StringNode, GeminiProvider
from stringllm.memory import BufferMemory

memory = BufferMemory(max_size=50)
chain = StringChain(
    nodes=[
        StringNode(name="chat", prompt="{user_message}", output_key="reply"),
    ],
    provider=GeminiProvider(),
    memory=memory,
)
```

## 6. Enable caching

Avoid redundant LLM calls with the SQLite cache:

```python
from stringllm.cache import SQLiteCache

cache = SQLiteCache(ttl=3600)  # cache responses for 1 hour
chain = StringChain(
    nodes=[...],
    provider=provider,
    cache=cache,
)
```

## 7. Run the playground

StringLLM ships with a web UI for interactive chain building:

```bash
stringllm
# or: python -m server.app
```

Open `http://127.0.0.1:8000` in your browser to start experimenting.

## Next steps

- [Providers](providers.md) -- learn about all supported LLM backends
- [Chains](chains.md) -- deep dive into chain construction and execution
- [Memory](memory.md) -- conversation history and caching
- [Examples](examples.md) -- complete working examples
