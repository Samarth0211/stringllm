"""Chain and step result dataclasses for StringLLM."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StepResult:
    """Result of a single node execution within a chain."""

    node_name: str
    output: str
    tokens: int
    time_ms: float
    provider: str


@dataclass(frozen=True)
class ChainResult:
    """Aggregate result of an entire chain execution."""

    outputs: dict[str, str] = field(default_factory=dict)
    total_tokens: int = 0
    total_time_ms: float = 0.0
    provider_used: str = ""
    steps: list[StepResult] = field(default_factory=list)
