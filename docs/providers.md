# Providers

StringLLM supports multiple LLM backends through a unified provider interface. Every provider implements `BaseLLMProvider` with two methods: `generate()` and `health_check()`.

## Built-in Providers

### GeminiProvider

Google's Gemini API (REST-based).

```python
from stringllm import GeminiProvider

provider = GeminiProvider(
    api_key="...",          # or set GEMINI_API_KEY env var
    model="gemini-2.5-flash",  # default model
)
```

**Environment variable:** `GEMINI_API_KEY`

### GroqProvider

Groq's OpenAI-compatible chat completions API.

```python
from stringllm import GroqProvider

provider = GroqProvider(
    api_key="...",                      # or set GROQ_API_KEY env var
    model="llama-3.3-70b-versatile",    # default model
)
```

**Environment variable:** `GROQ_API_KEY`

### HuggingFaceProvider

Hugging Face Inference API.

```python
from stringllm import HuggingFaceProvider

provider = HuggingFaceProvider(
    api_key="...",                                  # or set HF_API_KEY env var
    model="mistralai/Mistral-7B-Instruct-v0.3",    # default model
)
```

**Environment variable:** `HF_API_KEY`

### FallbackProvider

Wraps multiple providers and tries each in order until one succeeds. Providers that fail are temporarily deprioritized for a configurable cooldown period.

```python
from stringllm import FallbackProvider, GeminiProvider, GroqProvider

fallback = FallbackProvider(
    providers=[GeminiProvider(), GroqProvider()],
    cooldown_seconds=60.0,
)
```

## Custom Providers

Implement `BaseLLMProvider` to add your own backend:

```python
from stringllm.providers.base import BaseLLMProvider, ProviderResponse

class MyProvider(BaseLLMProvider):
    @property
    def name(self) -> str:
        return "my-provider"

    async def generate(
        self, prompt, system_prompt=None, temperature=0.7, max_tokens=1024
    ) -> ProviderResponse:
        text = await call_my_api(prompt)
        return ProviderResponse(
            text=text, tokens_used=100, model="my-model", provider=self.name
        )

    async def health_check(self) -> bool:
        return True
```

## ProviderResponse

All providers return a `ProviderResponse` dataclass:

| Field         | Type  | Description                    |
|---------------|-------|--------------------------------|
| `text`        | `str` | Generated text                 |
| `tokens_used` | `int` | Total tokens consumed          |
| `model`       | `str` | Model identifier               |
| `provider`    | `str` | Provider name                  |
