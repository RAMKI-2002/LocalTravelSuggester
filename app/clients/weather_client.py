"""OpenWeather client with DB-backed short-TTL cache and graceful degradation."""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.clients.http_base import UpstreamError, request_json
from app.config import get_settings
from app.db import cache as cache_helpers

logger = logging.getLogger(__name__)

OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"


def _normalise(raw: dict[str, Any]) -> dict[str, Any]:
    """Reduce OpenWeather's verbose payload to what the app actually needs."""
    main = raw.get("main", {}) or {}
    weather_list = raw.get("weather") or [{}]
    w0 = weather_list[0] if weather_list else {}
    return {
        "city": raw.get("name"),
        "temp_c": main.get("temp"),
        "feels_like_c": main.get("feels_like"),
        "humidity": main.get("humidity"),
        "condition": w0.get("main"),  # e.g. 'Clear', 'Rain'
        "description": w0.get("description"),
        "wind_kph": round((raw.get("wind", {}).get("speed", 0) or 0) * 3.6, 1),
    }


class WeatherClient:
    """Cache-through wrapper around the OpenWeather ``/weather`` endpoint."""

    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()

    async def get(self, city: str) -> tuple[dict[str, Any], bool]:
        """Return ``(payload, cache_hit)``. Raises ``UpstreamError`` only when
        the upstream fails *and* no stale cache entry is available.
        """
        fresh = cache_helpers.get_fresh_weather(
            self.db, city, self.settings.weather_cache_ttl_minutes
        )
        if fresh is not None:
            logger.debug("weather cache hit for %s", city)
            return fresh, True

        if not self.settings.openweather_api_key:
            stale = cache_helpers.get_stale_weather(self.db, city)
            if stale is not None:
                logger.warning("OPENWEATHER_API_KEY missing, returning stale cache")
                return stale, True
            raise UpstreamError(
                "OPENWEATHER_API_KEY is not configured and no cached weather exists"
            )

        try:
            raw = await request_json(
                "GET",
                OPENWEATHER_URL,
                params={
                    "q": city,
                    "appid": self.settings.openweather_api_key,
                    "units": "metric",
                },
            )
        except UpstreamError:
            stale = cache_helpers.get_stale_weather(self.db, city)
            if stale is not None:
                logger.warning("OpenWeather failed, returning stale weather cache")
                return stale, True
            raise

        normalised = _normalise(raw)
        try:
            cache_helpers.store_weather(
                self.db, city, normalised, self.settings.weather_cache_ttl_minutes
            )
        except Exception:  # pragma: no cover - cache writes must never break the flow
            logger.exception("failed to store weather cache")
        return normalised, False
