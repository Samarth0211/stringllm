"""FastAPI application for the StringLLM playground server."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from server.routes.chains import router as chains_router
from server.routes.providers import router as providers_router
from server.routes.playground import router as playground_router

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    """Factory function to create the FastAPI application."""
    application = FastAPI(
        title="StringLLM Playground",
        description="Interactive playground for building and running StringLLM chains.",
        version="0.1.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(chains_router)
    application.include_router(providers_router)
    application.include_router(playground_router)
    application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return application


app = create_app()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the StringLLM playground server via uvicorn."""
    import uvicorn

    host = os.getenv("STRINGLLM_HOST", "127.0.0.1")
    port = int(os.getenv("STRINGLLM_PORT", "8000"))
    reload_flag = os.getenv("STRINGLLM_RELOAD", "false").lower() == "true"

    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        reload=reload_flag,
    )


if __name__ == "__main__":
    main()
