```
   _____ _        _             _     _     __  __
  / ____| |      (_)           | |   | |   |  \/  |
 | (___ | |_ _ __ _ _ __   __ _| |   | |   | \  / |
  \___ \| __| '__| | '_ \ / _` | |   | |   | |\/| |
  ____) | |_| |  | | | | | (_| | |___| |___| |  | |
 |_____/ \__|_|  |_|_| |_|\__, |_____|_____|_|  |_|
                            __/ |
                           |___/
```

# StringLLM

### Chain free LLMs together like strings to solve complex tasks

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/Samarth0211/stringllm/ci.yml?style=for-the-badge&label=CI)](https://github.com/Samarth0211/stringllm/actions)
[![Tests](https://img.shields.io/badge/Tests-43_passing-brightgreen?style=for-the-badge)](tests/)

**[Live Playground](https://stringllm.vercel.app)** | **[Documentation](https://samarth0211.github.io/stringllm/)**

---

A lightweight Python framework for chaining multiple free LLM API calls together into multi-step AI pipelines. Summarize, analyze, translate, review — all in sequence, with automatic fallback across providers, built-in caching, conversation memory, and a live web playground.

**Zero paid resources.** Uses only free-tier APIs from Google Gemini, Groq, and HuggingFace.

---

## Why StringLLM?

- **Multi-provider fallback** — If Gemini hits a rate limit, Groq picks up. If Groq fails, HuggingFace takes over. Automatically.
- **Chain any LLM calls** — Build sequential pipelines where each step feeds into the next. Summarize → Analyze → Translate in 10 lines of code.
- **Production patterns** — Async everywhere, retry with exponential backoff, response caching, structured logging.
- **Built-in prompt library** — Pre-made templates for summarization, sentiment analysis, translation, code review, and more.
- **Web playground** — Dark-theme UI to visually build and run chains without writing code.
- **Publishable to PyPI** — Proper `pyproject.toml`, CI/CD, documentation, and 85%+ test coverage.

---

## Quick Start

### Install

```bash
git clone https://github.com/Samarth0211/stringllm.git
cd stringllm
pip install -e ".[dev]"
```

### Set at least one API key

```bash
export GEMINI_API_KEY="your_key"     # https://aistudio.google.com/apikey
# Optional:
export GROQ_API_KEY="your_key"       # https://console.groq.com/keys
export HF_API_KEY="your_key"         # https://huggingface.co/settings/tokens
```

### Run your first chain

```python
import asyncio
from stringllm import StringChain, StringNode, GeminiProvider

chain = StringChain(
    nodes=[
        StringNode(
            name="summarize",
            prompt="Summarize this in 3 bullet points:\n\n{input_text}",
            output_key="summary"
        ),
        StringNode(
            name="sentiment",
            prompt="What is the sentiment of this text?\n\n{summary}",
            output_key="analysis"
        ),
    ],
    provider=GeminiProvider()
)

result = asyncio.run(chain.run(input_text="Your long article here..."))
print(result.outputs["summary"])
print(result.outputs["analysis"])
print(f"Total tokens: {result.total_tokens}")
print(f"Time: {result.total_time_ms:.0f}ms")
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      StringLLM                          │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │  Node 1  │───>│  Node 2  │───>│  Node 3  │          │
│  │ Summarize│    │ Analyze  │    │ Translate │          │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘          │
│       │               │               │                 │
│       └───────────────┼───────────────┘                 │
│                       │                                 │
│              ┌────────┴────────┐                        │
│              │ FallbackProvider│                        │
│              └────────┬────────┘                        │
│       ┌───────────────┼───────────────┐                 │
│       │               │               │                 │
│  ┌────┴────┐    ┌────┴────┐    ┌────┴────┐             │
│  │ Gemini  │    │  Groq   │    │HuggingFace│            │
│  │ (free)  │    │ (free)  │    │  (free)  │             │
│  └─────────┘    └─────────┘    └─────────┘             │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │  Cache   │  │  Memory  │  │ Prompts  │             │
│  │ (SQLite) │  │(Buf/SQL) │  │ Library  │             │
│  └──────────┘  └──────────┘  └──────────┘             │
└─────────────────────────────────────────────────────────┘
```

---

## Providers

| Provider | Model | Free Tier Limits | Speed |
|----------|-------|-----------------|-------|
| **Gemini** | gemini-2.5-flash | 15 RPM, 1M tokens/day | Fast |
| **Groq** | llama-3.3-70b-versatile | 30 RPM, 14.4K tokens/min | Very Fast |
| **HuggingFace** | Mistral-7B-Instruct-v0.3 | Rate limited | Moderate |
| **Fallback** | Tries all in order | Combined limits | Resilient |

### Using Fallback

```python
from stringllm import FallbackProvider, GeminiProvider, GroqProvider

provider = FallbackProvider(providers=[
    GeminiProvider(),    # Try Gemini first
    GroqProvider(),      # Fall back to Groq
])

chain = StringChain(nodes=[...], provider=provider)
```

---

## Features

### Prompt Templates

```python
from stringllm import PromptLibrary

template = PromptLibrary.summarize()  # Pre-built template
rendered = template.render(text="Long article...", num_points="5")
```

Built-in templates: `summarize`, `analyze_sentiment`, `translate`, `extract_keywords`, `rewrite`, `code_review`, `explain_code`

### Conversation Memory

```python
from stringllm import StringChain, BufferMemory, SQLiteMemory

# In-memory (lost on restart)
chain = StringChain(nodes=[...], provider=p, memory=BufferMemory(max_size=20))

# Persistent SQLite
chain = StringChain(nodes=[...], provider=p, memory=SQLiteMemory(conversation_id="session-1"))
```

### Response Caching

```python
from stringllm import StringChain, SQLiteCache

# Cache identical prompts for 24 hours (avoids duplicate API calls)
chain = StringChain(nodes=[...], provider=p, cache=SQLiteCache(ttl=86400))
```

### Web Playground

```bash
stringllm  # or: python -m server.app
```

Opens a dark-theme web UI at `http://localhost:8000` where you can visually build and run chains.

---

## Project Structure

```
stringllm/
├── src/stringllm/
│   ├── core/               # StringChain, StringNode, ChainResult
│   ├── providers/          # Gemini, Groq, HuggingFace, Fallback
│   ├── prompts/            # PromptTemplate, PromptLibrary
│   ├── memory/             # BufferMemory, SQLiteMemory
│   ├── cache/              # SQLiteCache
│   └── utils/              # retry, token estimation, logging
├── server/
│   ├── app.py              # FastAPI application
│   ├── routes/             # API endpoints
│   └── static/             # Playground HTML/CSS/JS
├── tests/                  # 43 tests (all passing)
├── examples/               # 4 runnable example scripts
├── docs/                   # MkDocs documentation
├── .github/workflows/      # CI + docs deployment
├── pyproject.toml
└── mkdocs.yml
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest --cov=stringllm
```

All 43 tests use mocked HTTP — no real API calls needed.

---

## Examples

| Example | Description |
|---------|-------------|
| [simple_chain.py](examples/simple_chain.py) | Basic 2-step summarize + analyze |
| [summarize_and_translate.py](examples/summarize_and_translate.py) | Summarize then translate to Spanish |
| [code_review_chain.py](examples/code_review_chain.py) | Multi-step code review pipeline |
| [fallback_demo.py](examples/fallback_demo.py) | Provider fallback behavior demo |

---

## Environment Variables

```bash
GEMINI_API_KEY=           # Google Gemini API key (free)
GROQ_API_KEY=             # Groq API key (free)
HF_API_KEY=               # HuggingFace API key (free)
STRINGLLM_LOG_LEVEL=      # DEBUG, INFO, WARNING, ERROR (default: INFO)
```

At least one provider API key is required. The fallback provider works best with multiple keys set.

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest`)
4. Run linter (`ruff check .`)
5. Commit and push
6. Open a Pull Request

---

## Author

**Samarth Bhamare** — AI/ML Engineer

- [GitHub](https://github.com/Samarth0211)
- [LinkedIn](https://linkedin.com/in/samarth-bhamare)
- [Portfolio](https://samarthbhamare.vercel.app)

---

## License

MIT — see [LICENSE](LICENSE) for details.
