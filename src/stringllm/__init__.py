"""StringLLM - a lightweight async LLM chaining framework."""

from stringllm.core.chain import StringChain
from stringllm.core.node import StringNode
from stringllm.core.result import ChainResult, StepResult
from stringllm.providers.base import BaseLLMProvider, ProviderResponse
from stringllm.providers.fallback import FallbackProvider
from stringllm.providers.gemini import GeminiProvider
from stringllm.providers.groq import GroqProvider
from stringllm.providers.huggingface import HuggingFaceProvider

# The following imports are declared in the public API but their concrete
# implementations live in sibling packages that may be added later.
# We use lazy imports so the top-level package doesn't break if those
# modules aren't present yet.

def __getattr__(name: str):
    _lazy = {
        "PromptTemplate": "stringllm.prompts.template",
        "PromptLibrary": "stringllm.prompts.library",
        "BufferMemory": "stringllm.memory.buffer",
        "SQLiteMemory": "stringllm.memory.sqlite",
        "SQLiteCache": "stringllm.cache.sqlite",
    }
    if name in _lazy:
        import importlib
        module = importlib.import_module(_lazy[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core
    "StringChain",
    "StringNode",
    "ChainResult",
    "StepResult",
    # Providers
    "BaseLLMProvider",
    "ProviderResponse",
    "GeminiProvider",
    "GroqProvider",
    "HuggingFaceProvider",
    "FallbackProvider",
    # Prompts (lazy)
    "PromptTemplate",
    "PromptLibrary",
    # Memory (lazy)
    "BufferMemory",
    "SQLiteMemory",
    # Cache (lazy)
    "SQLiteCache",
]
