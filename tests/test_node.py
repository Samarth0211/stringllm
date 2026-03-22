"""Tests for StringNode."""

from __future__ import annotations

import pytest

from stringllm.core.node import StringNode
from stringllm.core.result import StepResult

from tests.conftest import MockProvider


@pytest.mark.asyncio
async def test_node_renders_prompt():
    """Variables in the prompt template should be substituted with input values."""
    provider = MockProvider()
    node = StringNode(
        name="renderer",
        prompt="Translate to {language}: {text}",
        output_key="translated",
    )

    result = await node.run(
        inputs={"language": "Spanish", "text": "Hello world"},
        provider=provider,
    )

    # The mock provider echoes back "Mock response for: <prompt[:50]>"
    # so the rendered prompt should contain the substituted values
    assert "Translate to Spanish: Hello world" in result.output


@pytest.mark.asyncio
async def test_node_returns_step_result():
    """Running a node should return a fully populated StepResult."""
    provider = MockProvider()
    node = StringNode(
        name="test_node",
        prompt="Analyze: {data}",
        output_key="analysis",
    )

    result = await node.run(inputs={"data": "some data"}, provider=provider)

    assert isinstance(result, StepResult)
    assert result.node_name == "test_node"
    assert isinstance(result.output, str)
    assert len(result.output) > 0
    assert result.tokens > 0
    assert result.time_ms >= 0
    assert result.provider == "mock"


@pytest.mark.asyncio
async def test_node_with_provider_override():
    """A node-level provider should take precedence over the chain-level provider."""
    chain_provider = MockProvider(name="chain_default")
    node_provider = MockProvider(name="node_override")

    node = StringNode(
        name="override_test",
        prompt="Process: {input}",
        output_key="output",
        provider=node_provider,
    )

    result = await node.run(inputs={"input": "test"}, provider=chain_provider)

    # The node should have used its own provider, not the chain-level one
    assert result.provider == "node_override"


@pytest.mark.asyncio
async def test_node_without_provider_override():
    """When no node-level provider is set, the chain-level provider is used."""
    chain_provider = MockProvider(name="chain_default")

    node = StringNode(
        name="no_override",
        prompt="Process: {input}",
        output_key="output",
    )

    result = await node.run(inputs={"input": "test"}, provider=chain_provider)

    assert result.provider == "chain_default"
