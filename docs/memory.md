# Memory and Caching

StringLLM provides conversation memory and response caching to make chains stateful and efficient.

## Memory Backends

All memory backends implement the `BaseMemory` interface with three async methods: `add()`, `get_history()`, and `clear()`.

### BufferMemory

An in-memory conversation buffer backed by a `collections.deque`. When the buffer reaches its maximum size, the oldest messages are automatically evicted.

```python
from stringllm.memory import BufferMemory

memory = BufferMemory(max_size=20)  # keep last 20 messages

await memory.add("user", "Hello!")
await memory.add("assistant", "Hi there!")

history = await memory.get_history()
# [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]

await memory.clear()
```

### SQLiteMemory

Persistent conversation memory stored in an SQLite database via `aiosqlite`. Conversations are isolated by a `conversation_id`, allowing multiple independent sessions within the same database.

```python
from stringllm.memory import SQLiteMemory

memory = SQLiteMemory(
    db_path="~/.stringllm/memory.db",  # default path
    conversation_id="session-42",
)

await memory.add("user", "Remember this for later")
history = await memory.get_history()
await memory.clear()
```

### Using Memory with Chains

Pass a memory backend to `StringChain` and each node's prompt and response will be automatically stored:

```python
from stringllm import StringChain, StringNode, GeminiProvider
from stringllm.memory import BufferMemory

chain = StringChain(
    nodes=[StringNode(name="chat", prompt="{message}", output_key="reply")],
    provider=GeminiProvider(),
    memory=BufferMemory(max_size=50),
)
```

## Response Cache

### SQLiteCache

An async cache that stores LLM responses in SQLite with configurable TTL (time-to-live). Cache keys are SHA-256 hashes of the provider name, prompt, temperature, and max_tokens.

```python
from stringllm.cache import SQLiteCache

cache = SQLiteCache(
    db_path="~/.stringllm/cache.db",  # default path
    ttl=86400,                         # 24 hours (default)
)

await cache.set("key", "cached response")
value = await cache.get("key")  # returns "cached response"

await cache.clear()
```

### Using Cache with Chains

```python
from stringllm import StringChain, StringNode, GeminiProvider
from stringllm.cache import SQLiteCache

chain = StringChain(
    nodes=[...],
    provider=GeminiProvider(),
    cache=SQLiteCache(ttl=3600),  # cache for 1 hour
)
```

When the cache is enabled, identical prompts with the same parameters will return the cached response without calling the LLM provider. Cached steps show `(cached)` in the provider name.
