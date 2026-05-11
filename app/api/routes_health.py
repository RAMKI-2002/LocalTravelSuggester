"""Health and observability endpoints.

We expose two endpoints:

* ``GET /health``
    Liveness probe. Cheap, no I/O, used by load balancers and the UI
    badge so the dashboard light goes green even if upstreams are flaky.

* ``GET /health/detailed``
    Readiness + dependency snapshot. Concurrently probes every upstream
    we depend on (OpenWeather, Foursquare, Overpass, Nominatim, Bedrock,
    DB) and reports each one's status. The frontend renders this as a
    colour-coded grid so an interviewer can see at a glance how every
    integration is doing.

Design notes worth saying out loud in an interview:
    * Two endpoints, two purposes (liveness vs readiness) - this is the
      Kubernetes-standard split.
    * Detailed checks run *concurrently* via ``asyncio.gather`` - one slow
      upstream cannot block the whole report.
    * Each probe has its own short timeout so the page stays responsive.
    * Probes never raise; every failure is captured into the response so
      the UI can render it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.clients.llm_client import BedrockLLMProvider, MockLLMProvider, get_llm_provider
from app.clients.overpass_client import ping_overpass
from app.clients.places_client import PlacesClient
from app.config import get_settings
from app.db.database import get_db
from app.db.models import PlaceCache, QueryHistory, WeatherLog
from app.utils.log_buffer import get_recent_logs

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])

# Probes get a tighter timeout than user requests - we don't want a slow
# upstream to make the health page itself feel slow.
_PROBE_TIMEOUT = 4.0


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe - process is up and responding."""
    return {"status": "ok"}


@router.get("/logs")
async def logs(
    limit: int = Query(default=200, ge=1, le=500),
    level: str | None = Query(default=None, description="Min level filter"),
    since_id: int | None = Query(default=None, description="Only return entries newer than this index"),
) -> dict[str, Any]:
    """Return the last ``limit`` log records from the in-memory ring buffer.

    The dashboard polls this endpoint every couple of seconds to render
    a live log feed. ``level`` lets the UI filter to WARNING+ when the
    "errors only" toggle is on.

    NOTE: this is a process-local buffer, not a persistent log store -
    it's a debugging convenience for demos. In production you'd ship
    these to CloudWatch / Loki / OpenSearch.
    """
    entries = get_recent_logs(limit=limit)
    if level:
        wanted = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
        floor = wanted.get(level.upper(), 0)
        entries = [
            e for e in entries
            if wanted.get(e.get("level", "INFO"), 0) >= floor
        ]
    return {"count": len(entries), "items": entries}


@router.get("/health/detailed")
async def health_detailed(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Readiness probe with per-dependency status.

    Returns ``200`` always (the *content* tells the truth). This way the
    UI can render the grid without having to interpret HTTP error codes.
    """
    settings = get_settings()
    started = time.perf_counter()

    db_check = _check_db(db)
    llm_check = _check_llm()
    weather_check = _probe(
        "openweather",
        "GET",
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "q": "Hyderabad",
            "appid": settings.openweather_api_key or "",
            "units": "metric",
        },
        configured=bool(settings.openweather_api_key),
    )
    # If Foursquare is disabled by config OR by the runtime circuit
    # breaker (account is out of credits), don't even hit the network -
    # report it as ``disabled`` so the dashboard tile renders amber, not
    # red. This keeps the health page honest AND fast.
    foursquare_check = _foursquare_probe()
    # Overpass health probe uses the same dedicated function the real
    # query path uses, so the dashboard reflects what the orchestrator
    # actually sees. We were previously sending an empty POST body which
    # made overpass-api.de return 406 forever - the dashboard was
    # always red, even though the real queries worked fine.
    overpass_check = _overpass_probe()
    nominatim_check = _probe(
        "nominatim",
        "GET",
        "https://nominatim.openstreetmap.org/search",
        params={"q": "Hyderabad", "format": "json", "limit": 1},
        headers={"User-Agent": settings.nominatim_user_agent},
        configured=True,
    )

    results = await asyncio.gather(
        db_check,
        llm_check,
        weather_check,
        foursquare_check,
        overpass_check,
        nominatim_check,
        return_exceptions=False,
    )
    checks = {r["name"]: r for r in results}

    # Overall status: degraded if any *configured* check is genuinely
    # down. ``disabled`` (amber) is not a failure - it's an explicit
    # "operator turned this off" / "circuit breaker open" signal.
    overall = "ok"
    for c in results:
        if c["status"] == "down" and c.get("configured", True):
            overall = "degraded"
            break

    return {
        "status": overall,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "checks": checks,
        "stats": _db_stats(db),
        "config": {
            "llm_provider": type(get_llm_provider()).__name__,
            "bedrock_model": settings.bedrock_model_id,
            "db_kind": "postgres" if settings.is_postgres else "sqlite",
            "llm_mock": settings.llm_mock,
        },
    }


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
async def _check_db(db: Session) -> dict[str, Any]:
    """Run a trivial SELECT 1-style query to confirm DB connectivity."""
    started = time.perf_counter()
    try:
        db.execute(select(1))
        return _ok("database", started, configured=True)
    except Exception as exc:
        logger.warning("DB health probe failed: %s", exc)
        return _down("database", started, str(exc), configured=True)


async def _check_llm() -> dict[str, Any]:
    """Confirm the LLM provider is wired up (does NOT make a billable call)."""
    started = time.perf_counter()
    settings = get_settings()
    provider = get_llm_provider()
    if isinstance(provider, MockLLMProvider):
        return _ok(
            "llm",
            started,
            configured=not settings.llm_mock,
            note="mock-provider (no Bedrock call made)",
        )
    if isinstance(provider, BedrockLLMProvider):
        # We deliberately do not invoke the model here - that would cost
        # money on every health refresh. The presence of an initialised
        # boto3 client is a strong-enough signal for the dashboard.
        return _ok(
            "llm",
            started,
            configured=True,
            note=f"bedrock client ready ({provider.model_id})",
        )
    return _ok("llm", started, configured=True, note="unknown provider")


async def _foursquare_probe() -> dict[str, Any]:
    """Fast, honest Foursquare probe.

    * If ``FOURSQUARE_ENABLED=false`` -> ``disabled`` (amber).
    * If no API key -> ``disabled`` (amber).
    * If the runtime breaker has tripped -> ``disabled`` with the
      reason the breaker opened.
    * Otherwise: do a real 1-row search. 200 OR 429 is fine (429 just
      means the account is out of quota; the fallback will handle it).
    """
    started = time.perf_counter()
    settings = get_settings()
    if not settings.foursquare_enabled:
        return _disabled(
            "foursquare", started, "disabled by FOURSQUARE_ENABLED=false"
        )
    if not settings.foursquare_api_key:
        return _disabled("foursquare", started, "no API key configured")
    if PlacesClient._billing_disabled:  # noqa: SLF001 - intentional read
        return _disabled(
            "foursquare", started, "circuit breaker open (out of API credits)"
        )
    return await _probe(
        "foursquare",
        "GET",
        "https://places-api.foursquare.com/places/search",
        params={"near": "Hyderabad", "query": "tourist attractions", "limit": 1},
        headers={
            "Authorization": f"Bearer {settings.foursquare_api_key}",
            "X-Places-Api-Version": "2025-06-17",
        },
        configured=True,
        ok_statuses=(200, 429),
    )


async def _overpass_probe() -> dict[str, Any]:
    """Probe Overpass via the same code path our real queries use, so
    the dashboard reflects reality. Tries each mirror in order."""
    started = time.perf_counter()
    try:
        ok, info = await ping_overpass(timeout_s=_PROBE_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - probes never raise
        return _down("overpass", started, str(exc)[:140], configured=True)
    if ok:
        return _ok("overpass", started, configured=True, note=f"reachable via {info}")
    return _down(
        "overpass", started, f"all mirrors failed: {info or 'unknown'}",
        configured=True,
    )


async def _probe(
    name: str,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    data: dict[str, Any] | None = None,
    configured: bool = True,
    ok_statuses: tuple[int, ...] = (200,),
) -> dict[str, Any]:
    """Generic upstream probe with isolated httpx client + timeout."""
    started = time.perf_counter()
    if not configured:
        return _down(name, started, "not configured", configured=False)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(_PROBE_TIMEOUT)) as c:
            resp = await c.request(method, url, params=params, headers=headers, data=data)
        if resp.status_code in ok_statuses:
            return _ok(
                name, started, configured=True, note=f"HTTP {resp.status_code}"
            )
        return _down(
            name,
            started,
            f"HTTP {resp.status_code}",
            configured=True,
        )
    except Exception as exc:  # noqa: BLE001 - probes never raise
        return _down(name, started, str(exc)[:140], configured=configured)


def _db_stats(db: Session) -> dict[str, Any]:
    """Lightweight cache + history counters for the dashboard."""
    try:
        history_count = db.query(QueryHistory).count()
        place_cache_count = db.query(PlaceCache).count()
        weather_log_count = db.query(WeatherLog).count()
    except Exception as exc:
        logger.warning("DB stats probe failed: %s", exc)
        return {"error": str(exc)}
    return {
        "history_rows": history_count,
        "place_cache_rows": place_cache_count,
        "weather_log_rows": weather_log_count,
    }


def _ok(
    name: str, started: float, *, configured: bool, note: str = "ok"
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "ok",
        "configured": configured,
        "note": note,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


def _down(
    name: str, started: float, error: str, *, configured: bool
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "down",
        "configured": configured,
        "note": error,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


def _disabled(name: str, started: float, reason: str) -> dict[str, Any]:
    """Intentionally turned off (config or runtime breaker). Renders
    amber on the dashboard - it is NOT a failure.
    """
    return {
        "name": name,
        "status": "disabled",
        "configured": False,
        "note": reason,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }
