"""PromptTemplate for variable substitution in prompt strings."""

from __future__ import annotations

import re


class PromptTemplate:
    """A template string with named variables that can be rendered with provided values.

    Variables are denoted by ``{variable_name}`` in the template string.  Default
    values can be supplied at construction time so callers only need to provide the
    variables they wish to override.

    Raises:
        ValueError: If required variables (those without defaults) are not supplied
            when :meth:`render` is called.
    """

    _VARIABLE_PATTERN: re.Pattern[str] = re.compile(r"\{(\w+)\}")

    def __init__(self, template: str, defaults: dict[str, str] | None = None) -> None:
        self._template = template
        self._defaults: dict[str, str] = defaults or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def variables(self) -> list[str]:
        """Return an ordered, deduplicated list of variable names found in the template."""
        seen: set[str] = set()
        result: list[str] = []
        for match in self._VARIABLE_PATTERN.finditer(self._template):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result

    def render(self, **kwargs: str) -> str:
        """Render the template by substituting variables.

        Variables present in *kwargs* take precedence over defaults.  Any variable
        that appears in the template but is missing from both *kwargs* and the
        defaults will cause a :class:`ValueError`.

        Args:
            **kwargs: Variable name / value pairs to substitute.

        Returns:
            The rendered prompt string.

        Raises:
            ValueError: When one or more required variables are missing.
        """
        merged: dict[str, str] = {**self._defaults, **kwargs}

        missing = [v for v in self.variables() if v not in merged]
        if missing:
            raise ValueError(
                f"Missing required template variable(s): {', '.join(missing)}"
            )

        result = self._template
        for name, value in merged.items():
            result = result.replace(f"{{{name}}}", str(value))
        return result

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PromptTemplate(template={self._template!r}, "
            f"defaults={self._defaults!r})"
        )
