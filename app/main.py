"""FastAPI application entrypoint.

Run locally:
    uvicorn app.main:app --reload

Routes:
    GET  /              -> redirects to /ui
    GET  /ui            -> single-page dashboard (HTML)
    GET  /health        -> liveness
    GET  /health/detailed -> readiness + per-dependency snapshot
    POST /suggest-trip  -> the actual recommendation engine
    GET  /history       -> last N persisted queries (for the UI panel)
    GET  /docs          -> Swagger UI
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import routes_health, routes_trip
from app.clients.http_base import close_http_client
from app.config import get_settings
from app.db.database import init_db
from app.utils.logger import RequestIdMiddleware, configure_logging

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    db_label = settings.database_url.split("@")[-1]
    logger.info(
        "Starting Local Trip Suggester (db=%s, llm_mock=%s, model=%s)",
        db_label,
        settings.llm_mock,
        settings.bedrock_model_id,
    )
    init_db()
    logger.info("DB tables ready")
    yield
    await close_http_client()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Local Trip Suggester",
        version="0.3.0",
        description=(
            "AI-powered trip recommendations: weather-aware, preference-aware, "
            "distance + budget enriched, with AWS Bedrock reasoning and a "
            "Foursquare -> OpenStreetMap fallback chain."
        ),
        lifespan=_lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(routes_health.router)
    app.include_router(routes_trip.router)

    # Mount the bundled single-page UI. We expose three convenient paths:
    #   * /ui                -> serves index.html (the dashboard)
    #   * /how-it-works      -> serves the architecture / code-flow page
    #   * /static/<file>     -> JS/CSS/etc.
    if _STATIC_DIR.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(_STATIC_DIR)),
            name="static",
        )

        @app.get("/ui", include_in_schema=False)
        async def ui() -> FileResponse:
            return FileResponse(str(_STATIC_DIR / "index.html"))

        @app.get("/how-it-works", include_in_schema=False)
        async def how_it_works() -> FileResponse:
            return FileResponse(str(_STATIC_DIR / "how-it-works.html"))

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        # Send humans to the UI, machines to /docs via the API path.
        return RedirectResponse(url="/ui")

    @app.get("/api", tags=["meta"])
    async def api_root() -> dict[str, str]:
        return {
            "name": "local-trip-suggester",
            "version": "0.3.0",
            "ui": "/ui",
            "docs": "/docs",
            "health": "/health/detailed",
        }

    return app


app = create_app()
