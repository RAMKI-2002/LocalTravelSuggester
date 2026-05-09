"""OpenStreetMap Nominatim geocoding client.

Nominatim's usage policy requires a descriptive ``User-Agent`` and limits
callers to ~1 request/sec. That's fine for our demo traffic, but we wrap
every call in the shared retry/backoff machinery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.clients.http_base import NotFoundError, UpstreamError, request_json
from app.config import get_settings

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


@dataclass(frozen=True)
class GeoPoint:
    lat: float
    lng: float
    display_name: str


class GeocodeClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def geocode(self, query: str) -> Optional[GeoPoint]:
        """Resolve a free-text query (e.g. ``"Gachibowli, Hyderabad"``) to a point.

        Returns ``None`` if no match is found or the call fails - geocoding is
        advisory (used only to compute distances) so we never raise upward.
        """
        headers = {"User-Agent": self.settings.nominatim_user_agent}
        params = {"q": query, "format": "json", "limit": 1}

        try:
            data = await request_json(
                "GET", NOMINATIM_URL, params=params, headers=headers
            )
        except NotFoundError:
            return None
        except UpstreamError as exc:
            logger.warning("Nominatim failed for '%s': %s", query, exc)
            return None

        # Nominatim returns a list; ``request_json`` wraps it when it is an
        # object. For safety we accept either shape.
        if isinstance(data, dict):  # pragma: no cover - unexpected shape
            return None
        if not data:
            return None

        top = data[0]
        try:
            return GeoPoint(
                lat=float(top["lat"]),
                lng=float(top["lon"]),
                display_name=top.get("display_name", query),
            )
        except (KeyError, TypeError, ValueError):
            return None
