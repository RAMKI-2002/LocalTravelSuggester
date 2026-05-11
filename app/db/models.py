"""ORM models.

``JSON`` from SQLAlchemy maps to JSONB on PostgreSQL and to TEXT-backed JSON
on SQLite, so these models work for both targets without branching.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class QueryHistory(Base):
    """Every ``/suggest-trip`` call is logged here for auditing / analytics."""

    __tablename__ = "query_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    city: Mapped[str] = mapped_column(String(128), index=True)
    preference: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    locality: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    response: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PlaceCache(Base):
    """Cached Foursquare tourist-attraction results keyed by city."""

    __tablename__ = "place_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    city: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class WeatherLog(Base):
    """Cached weather per city with short TTL (acts as both cache + audit log)."""

    __tablename__ = "weather_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    city: Mapped[str] = mapped_column(String(128), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
