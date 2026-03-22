"""Multi-step code review pipeline.

This chain performs three stages:
1. Review the code for bugs and issues
2. Suggest concrete improvements
3. Generate a final summary report

Usage:
    export GEMINI_API_KEY="your-api-key"
    python examples/code_review_chain.py
"""

import asyncio

from stringllm import GeminiProvider, StringChain, StringNode


async def main() -> None:
    provider = GeminiProvider()

    nodes = [
        StringNode(
            name="bug_review",
            prompt=(
                "You are an expert code reviewer. Analyze the following code for bugs, "
                "security vulnerabilities, and logic errors. List each issue with its "
                "severity (critical, warning, info):\n\n```\n{code}\n```"
            ),
            output_key="bug_report",
        ),
        StringNode(
            name="improvements",
            prompt=(
                "Based on the following code and its bug report, suggest concrete "
                "improvements including better naming, design patterns, performance "
                "optimizations, and test suggestions.\n\n"
                "Original code:\n```\n{code}\n```\n\n"
                "Bug report:\n{bug_report}"
            ),
            output_key="improvement_suggestions",
        ),
        StringNode(
            name="summary",
            prompt=(
                "Generate a concise code review summary report that includes:\n"
                "1. Overall code quality rating (1-10)\n"
                "2. Top 3 issues found\n"
                "3. Top 3 recommended improvements\n"
                "4. One sentence overall assessment\n\n"
                "Bug report:\n{bug_report}\n\n"
                "Improvement suggestions:\n{improvement_suggestions}"
            ),
            output_key="review_summary",
        ),
    ]

    chain = StringChain(nodes=nodes, provider=provider)

    sample_code = '''\
import sqlite3
import os

def get_user(username):
    conn = sqlite3.connect("users.db")
    query = f"SELECT * FROM users WHERE username = '{username}'"
    result = conn.execute(query).fetchone()
    conn.close()
    return result

def save_file(filename, content):
    path = "/uploads/" + filename
    with open(path, "w") as f:
        f.write(content)
    return path

def process_data(items):
    result = []
    for i in range(len(items)):
        if items[i] != None:
            result.append(items[i] * 2)
    return result

PASSWORD = "admin123"
'''

    result = await chain.run(code=sample_code)

    for step in result.steps:
        print("=" * 60)
        print(f"STEP: {step.node_name} ({step.time_ms:.0f}ms, {step.tokens} tokens)")
        print("=" * 60)
        print(step.output)
        print()

    print("=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total tokens: {result.total_tokens}")
    print(f"Total time: {result.total_time_ms:.0f}ms")
    print(f"Steps completed: {len(result.steps)}")


if __name__ == "__main__":
    asyncio.run(main())
