"""Thin TTL-aware read/write helpers on top of the cache tables.

Clients (weather/places) delegate to these functions so caching logic stays
centralised and easy to reason about.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models import PlaceCache, WeatherLog


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
def get_fresh_weather(
    db: Session, city: str, ttl_minutes: int
) -> Optional[dict[str, Any]]:
    """Return cached weather payload if still within TTL, else ``None``."""
    row = db.execute(
        select(WeatherLog)
        .where(WeatherLog.city == city.lower())
        .order_by(WeatherLog.fetched_at.desc())
        .limit(1)
    ).scalar_one_or_none()

    if row is None:
        return None
    expires_at = row.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < _utcnow():
        return None
    return row.payload


def get_stale_weather(db: Session, city: str) -> Optional[dict[str, Any]]:
    """Return the most recent payload regardless of TTL (used for degradation)."""
    row = db.execute(
        select(WeatherLog)
        .where(WeatherLog.city == city.lower())
        .order_by(WeatherLog.fetched_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    return row.payload if row else None


def store_weather(
    db: Session, city: str, payload: dict[str, Any], ttl_minutes: int
) -> None:
    entry = WeatherLog(
        city=city.lower(),
        payload=payload,
        expires_at=_utcnow() + timedelta(minutes=ttl_minutes),
    )
    db.add(entry)
    db.commit()


# ---------------------------------------------------------------------------
# Places
# ---------------------------------------------------------------------------
def get_fresh_places(
    db: Session, city: str, ttl_hours: int
) -> Optional[list[dict[str, Any]]]:
    row = db.execute(
        select(PlaceCache).where(PlaceCache.city == city.lower())
    ).scalar_one_or_none()
    if row is None:
        return None
    expires_at = row.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < _utcnow():
        return None
    return row.payload.get("places") if isinstance(row.payload, dict) else None


def get_stale_places(db: Session, city: str) -> Optional[list[dict[str, Any]]]:
    row = db.execute(
        select(PlaceCache).where(PlaceCache.city == city.lower())
    ).scalar_one_or_none()
    if row is None:
        return None
    return row.payload.get("places") if isinstance(row.payload, dict) else None


def store_places(
    db: Session, city: str, places: list[dict[str, Any]], ttl_hours: int
) -> None:
    """Upsert-style: delete existing row for city then insert fresh one.

    Avoids DB-specific ``ON CONFLICT`` so this works on both Postgres and SQLite.
    """
    db.execute(delete(PlaceCache).where(PlaceCache.city == city.lower()))
    db.add(
        PlaceCache(
            city=city.lower(),
            payload={"places": places},
            expires_at=_utcnow() + timedelta(hours=ttl_hours),
        )
    )
    db.commit()
