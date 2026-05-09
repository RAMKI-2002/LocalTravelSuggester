"""Trip-suggestion + history endpoints.

These are the two routes that drive the dashboard:

* ``POST /suggest-trip`` - the main pipeline (weather + places + LLM
  reasoning) returning a fully-enriched response that the UI renders.
* ``GET  /history``      - last N persisted suggestions, used to show
  the "recent queries" panel.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import QueryHistory
from app.schemas.trip import TripRequest, TripResponse
from app.services.trip_service import suggest_trip

logger = logging.getLogger(__name__)
router = APIRouter(tags=["trip"])


@router.post("/suggest-trip", response_model=TripResponse)
async def suggest_trip_endpoint(
    req: TripRequest,
    db: Session = Depends(get_db),
) -> TripResponse:
    """Recommend tourist places for a city based on weather + user intent."""
    logger.info(
        "suggest-trip request: city=%s pref=%r locality=%r max=%s",
        req.city,
        req.preference,
        req.locality,
        req.max_results,
    )
    return await suggest_trip(db, req)


@router.get("/history")
async def history(
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return the most recent ``limit`` persisted ``/suggest-trip`` calls.

    Light-weight summary only (city / preference / latency / suggestion
    count) - the full response payload stays in the DB and is not
    re-shipped to the UI.
    """
    rows = db.execute(
        select(QueryHistory)
        .order_by(QueryHistory.created_at.desc())
        .limit(limit)
    ).scalars().all()

    items: list[dict[str, Any]] = []
    for row in rows:
        payload = row.response or {}
        suggestions = payload.get("suggestions") or []
        meta = payload.get("meta") or {}
        items.append(
            {
                "id": row.id,
                "city": row.city,
                "preference": row.preference,
                "locality": row.locality,
                "suggestion_count": len(suggestions),
                "top_suggestion": suggestions[0]["name"] if suggestions else None,
                "latency_ms": row.latency_ms,
                "degraded": meta.get("degraded") or [],
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
        )
    return {"count": len(items), "items": items}
