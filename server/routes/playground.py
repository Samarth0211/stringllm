"""Playground UI and template routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

from stringllm.prompts.library import PromptLibrary

router = APIRouter(tags=["playground"])

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class TemplateInfo(BaseModel):
    name: str
    template: str
    variables: list[str]


class TemplateListResponse(BaseModel):
    templates: list[TemplateInfo]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gather_templates() -> list[TemplateInfo]:
    """Introspect PromptLibrary to collect every static template method."""
    templates: list[TemplateInfo] = []

    # PromptLibrary uses @staticmethod for each template factory
    for attr_name in dir(PromptLibrary):
        if attr_name.startswith("_"):
            continue
        method = getattr(PromptLibrary, attr_name)
        if not callable(method):
            continue
        try:
            prompt_template = method()
            # Expect a PromptTemplate with ._template and .variables()
            templates.append(
                TemplateInfo(
                    name=attr_name,
                    template=prompt_template._template,
                    variables=prompt_template.variables(),
                )
            )
        except Exception:
            continue

    return templates


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/api/templates", response_model=TemplateListResponse)
async def list_templates() -> TemplateListResponse:
    """Return all available PromptLibrary templates."""
    return TemplateListResponse(templates=_gather_templates())


@router.get("/")
async def serve_playground() -> FileResponse:
    """Serve the single-page playground UI."""
    return FileResponse(str(STATIC_DIR / "index.html"), media_type="text/html")
