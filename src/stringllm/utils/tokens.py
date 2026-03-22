"""Lightweight token estimation utility."""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in *text*.

    Uses a simple heuristic: split by whitespace and multiply the word
    count by 1.3 to approximate sub-word tokenisation.  This is intentionally
    fast and dependency-free; for exact counts use a proper tokeniser.

    Args:
        text: The input string.

    Returns:
        Estimated token count (rounded to the nearest integer).
    """
    word_count = len(text.split())
    return round(word_count * 1.3)
