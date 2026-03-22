# Examples

Complete working examples are in the `examples/` directory. Each can be run directly after setting the appropriate API key.

## Simple Chain

Summarize text and then analyze the sentiment of the summary.

```bash
export GEMINI_API_KEY="your-key"
python examples/simple_chain.py
```

```python
nodes = [
    StringNode(
        name="summarize",
        prompt="Summarize into 3 bullet points:\n\n{text}",
        output_key="summary",
    ),
    StringNode(
        name="sentiment",
        prompt="Analyze the sentiment of:\n\n{summary}",
        output_key="sentiment_analysis",
    ),
]
```

## Summarize and Translate

Summarize an article into English, then translate the summary to Spanish.

```bash
python examples/summarize_and_translate.py
```

## Code Review Pipeline

A three-step chain that reviews code for bugs, suggests improvements, and produces a summary report.

```bash
python examples/code_review_chain.py
```

Steps:

1. **Bug review** -- find bugs, security issues, and logic errors
2. **Improvements** -- suggest naming, design, and performance improvements
3. **Summary** -- generate a final quality rating and top recommendations

## Fallback Demo

Demonstrates automatic failover across multiple LLM providers. Set multiple API keys to see the fallback in action:

```bash
export GEMINI_API_KEY="your-gemini-key"
export GROQ_API_KEY="your-groq-key"
export HF_API_KEY="your-hf-key"
python examples/fallback_demo.py
```

If the first provider fails, the chain transparently retries with the next available provider.
