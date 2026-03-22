"""Tests for PromptTemplate and PromptLibrary."""

from __future__ import annotations

import pytest

from stringllm.prompts.template import PromptTemplate
from stringllm.prompts.library import PromptLibrary


# ---------------------------------------------------------------------------
# PromptTemplate Tests
# ---------------------------------------------------------------------------

def test_render_with_variables():
    """Template variables should be replaced with provided values."""
    tpl = PromptTemplate("Translate {text} to {language}")
    result = tpl.render(text="Hello", language="Spanish")

    assert result == "Translate Hello to Spanish"


def test_render_with_defaults():
    """Variables with defaults should use the default when not provided."""
    tpl = PromptTemplate(
        "Summarize into {num_points} points: {text}",
        defaults={"num_points": "3"},
    )
    result = tpl.render(text="Some long text here")

    assert result == "Summarize into 3 points: Some long text here"


def test_render_defaults_can_be_overridden():
    """Explicitly provided values should override defaults."""
    tpl = PromptTemplate(
        "Extract {num_keywords} keywords from: {text}",
        defaults={"num_keywords": "5"},
    )
    result = tpl.render(num_keywords="10", text="Some text")

    assert result == "Extract 10 keywords from: Some text"


def test_missing_variable_raises():
    """Rendering with a missing required variable should raise ValueError."""
    tpl = PromptTemplate("Translate {text} to {language}")

    with pytest.raises(ValueError, match="Missing required template variable"):
        tpl.render(text="Hello")  # missing "language"


def test_variables_returns_all_placeholders():
    """The variables() method should return all unique variable names."""
    tpl = PromptTemplate("{a} and {b} and {a} again")
    variables = tpl.variables()

    assert variables == ["a", "b"]


# ---------------------------------------------------------------------------
# PromptLibrary Tests
# ---------------------------------------------------------------------------

def test_library_summarize_template():
    """PromptLibrary.summarize() should return a working template."""
    tpl = PromptLibrary.summarize()
    result = tpl.render(text="The quick brown fox jumps over the lazy dog.")

    assert "3" in result  # default num_points
    assert "The quick brown fox" in result


def test_library_all_templates_render():
    """Every template in PromptLibrary should render without errors
    when all required variables are supplied."""
    templates_and_vars = {
        "summarize": {"text": "Sample text"},
        "analyze_sentiment": {"text": "I love this product"},
        "translate": {"target_language": "French", "text": "Hello world"},
        "extract_keywords": {"text": "Machine learning is a subset of AI"},
        "rewrite": {"tone": "formal", "text": "Hey what's up"},
        "code_review": {"code": "def foo(): pass"},
        "explain_code": {"code": "print('hello')"},
    }

    for method_name, variables in templates_and_vars.items():
        method = getattr(PromptLibrary, method_name)
        tpl = method()
        result = tpl.render(**variables)
        assert isinstance(result, str)
        assert len(result) > 0, f"Template {method_name} rendered to empty string"
