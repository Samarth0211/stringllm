"""Vercel serverless entry point for StringLLM playground."""

import sys
from pathlib import Path

# Add project root to path so imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from server.app import create_app  # noqa: E402

app = create_app()
