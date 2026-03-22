# Chains

A **StringChain** executes an ordered sequence of **StringNode** instances, passing each node's output to the next as a template variable.

## StringNode

A node represents a single LLM call within a chain.

```python
from stringllm import StringNode

node = StringNode(
    name="summarize",                           # human-readable name
    prompt="Summarize:\n\n{text}",              # prompt template with {variables}
    output_key="summary",                       # key to store the output under
    temperature=0.7,                            # sampling temperature (default 0.7)
    max_tokens=1024,                            # max output tokens (default 1024)
    provider=None,                              # optional per-node provider override
)
```

### Prompt Templates

Node prompts use Python format-string syntax. Variables in `{braces}` are resolved from the chain context at runtime:

- Initial inputs passed to `chain.run(**kwargs)` are available to the first node.
- Each node's output is stored under its `output_key` and becomes available to subsequent nodes.

### Provider Override

If a node has its own `provider`, that provider is used instead of the chain-level default. This lets you mix providers within a single chain.

## StringChain

```python
from stringllm import StringChain, StringNode, GeminiProvider

chain = StringChain(
    nodes=[node_a, node_b, node_c],
    provider=GeminiProvider(),
    memory=None,    # optional memory backend
    cache=None,     # optional cache backend
)

result = await chain.run(text="input text here")
```

### ChainResult

The `run()` method returns a `ChainResult`:

| Field           | Type               | Description                          |
|-----------------|--------------------|--------------------------------------|
| `outputs`       | `dict[str, str]`   | Map of output_key to generated text  |
| `total_tokens`  | `int`              | Sum of tokens across all steps       |
| `total_time_ms` | `float`            | Total wall-clock time in milliseconds|
| `provider_used` | `str`              | Name of the last provider used       |
| `steps`         | `list[StepResult]` | Per-node execution details           |

### StepResult

Each entry in `steps` contains:

| Field       | Type    | Description               |
|-------------|---------|---------------------------|
| `node_name` | `str`   | Name of the node          |
| `output`    | `str`   | Generated text            |
| `tokens`    | `int`   | Tokens used by this step  |
| `time_ms`   | `float` | Execution time in ms      |
| `provider`  | `str`   | Provider that handled it  |

## Example: Three-Step Pipeline

```python
import asyncio
from stringllm import GeminiProvider, StringChain, StringNode

async def main():
    chain = StringChain(
        nodes=[
            StringNode(name="draft", prompt="Draft an email about: {topic}", output_key="draft"),
            StringNode(name="review", prompt="Review this email for tone:\n{draft}", output_key="review"),
            StringNode(name="final", prompt="Apply these edits:\n{review}\n\nTo:\n{draft}", output_key="final_email"),
        ],
        provider=GeminiProvider(),
    )
    result = await chain.run(topic="project deadline extension")
    print(result.outputs["final_email"])

asyncio.run(main())
```
