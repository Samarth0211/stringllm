"""Structured JSON logging for StringLLM."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any


class _JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_entry["extra"] = record.extra_data  # type: ignore[attr-defined]

        return json.dumps(log_entry, default=str)


def setup_logger(name: str = "stringllm") -> logging.Logger:
    """Create (or retrieve) a logger that emits structured JSON to stderr.

    The log level is read from the ``STRINGLLM_LOG_LEVEL`` environment
    variable.  Accepted values are standard Python level names (``DEBUG``,
    ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``).  Defaults to ``INFO``.

    Calling this function multiple times with the same *name* is safe; handlers
    are only attached once.

    Args:
        name: Logger name.  Defaults to ``"stringllm"``.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called more than once.
    if logger.handlers:
        return logger

    level_name = os.environ.get("STRINGLLM_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter())
    logger.addHandler(handler)

    # Prevent logs from propagating to the root logger and being duplicated.
    logger.propagate = False

    return logger
