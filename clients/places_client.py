"""Foursquare Places API client - new platform (post-2025).

Endpoint:  https://places-api.foursquare.com/places/search
Auth:      Authorization: Bearer <service-api-key>
Version:   X-Places-Api-Version: 2025-06-17

The legacy ``api.foursquare.com/v3/places/search`` endpoint is only available
for developers who signed up before June 17, 2025 and is scheduled for full
deprecation on May 15, 2026. The new Places API is incompatible with the old
one at both request and response level, so we target the new API directly.

Free tier is generous but rate-limited, so we cache the normalised result
per-city for 24 hours. When the upstream fails we fall back to stale cache
rather than breaking the whole request.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.clients.http_base import UpstreamError, request_json
from app.config import get_settings
from app.db import cache as cache_helpers

logger = logging.getLogger(__name__)

FOURSQUARE_URL = "https://places-api.foursquare.com/places/search"
FOURSQUARE_API_VERSION = "2025-06-17"

# Default query used to bias results towards tourism-relevant POIs. We
# deliberately avoid a category filter: the new FSQ taxonomy uses UUID-style
# IDs (e.g. 4d4b7105d754a06377d81259 for Landmarks) that change over time, and
# a simple ``query`` gets us high-quality tourist attractions across cities
# without maintaining a taxonomy map.
DEFAULT_QUERY = "tourist attractions"

FIELDS = ",".join(
    [
        "fsq_place_id",
        "name",
        "latitude",
        "longitude",
        "location",
        "categories",
        "price",
        "rating",
        "description",
        "popularity",
        "tel",
        "website",
    ]
)


def _normalise(raw_place: dict[str, Any]) -> dict[str, Any]:
    categories = [c.get("name") for c in raw_place.get("categories") or [] if c]
    lat = raw_place.get("latitude")
    lng = raw_place.get("longitude")

    # New API returns plain floats but some variants wrap them in {"value": ...}
    if isinstance(lat, dict):
        lat = lat.get("value")
    if isinstance(lng, dict):
        lng = lng.get("value")

    return {
        "fsq_id": raw_place.get("fsq_place_id"),
        "name": raw_place.get("name"),
        "description": raw_place.get("description") or "",
        "categories": [c for c in categories if c],
        "coords": {"lat": lat, "lng": lng},
        "address": (raw_place.get("location") or {}).get("formatted_address"),
        "price_tier": raw_place.get("price"),
        "rating": raw_place.get("rating"),
        "popularity": raw_place.get("popularity"),
        "website": raw_place.get("website"),
    }


class PlacesClient:
    """Cache-through wrapper around the new Foursquare Places /search.

    Query modes:
      * **Anchored** (``lat``+``lng`` supplied): uses ``ll=lat,lng`` and
        ``radius`` so results are tightly bounded to the user's locality
        / city centre. This avoids 60-70 km outliers for cities whose
        admin boundary is huge (e.g. Pune district).
      * **Named** (no anchor): falls back to ``near=<city>``. Used when
        Nominatim is down so we still return *something*.

    The ``query`` parameter is **dynamic** - it's built from the user's
    extracted intent (e.g. "restaurants cafes food" for a "want to eat"
    prompt). Without this the candidate pool would be the same for
    every prompt - food, spiritual and adventure would all return the
    same generic tourist attractions.

    The ``cache_namespace`` parameter is the canonical intent category
    ("food", "spiritual", "tourist", ...). It's part of the cache key
    so different intents on the same city/anchor cache separately.

    Circuit breaker
    ---------------

    Foursquare's free tier exhausts quickly. Once an account hits the
    "no API credits remaining" 429, every subsequent call returns the
    same 429 - calling it costs 2-3 seconds of pure waste per request.
    To stop bleeding latency we keep a class-level ``_billing_disabled``
    flag: the first time we see that specific 429 we set it, log a
    one-time warning, and from then on :meth:`is_disabled` reports True
    so the orchestrator can skip Foursquare entirely and go straight to
    Overpass. The flag also honours the ``FOURSQUARE_ENABLED`` env var
    for hard kill-switch use.
    """

    DEFAULT_RADIUS_M = 25_000

    # Class-level so a single billing-error 429 disables the provider
    # for the *whole process*, not just this request.
    _billing_disabled: bool = False

    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()

    @classmethod
    def is_disabled(cls) -> bool:
        """``True`` if Foursquare should be skipped entirely.

        Combines the static ``FOURSQUARE_ENABLED`` config flag, a
        missing API key, and the runtime circuit breaker. The
        orchestrator checks this BEFORE issuing the call to avoid the
        2-3s wasted round-trip on a permanently-dead provider.
        """
        settings = get_settings()
        if not settings.foursquare_enabled:
            return True
        if not settings.foursquare_api_key:
            return True
        return cls._billing_disabled

    @classmethod
    def trip_billing_breaker(cls, message: str) -> None:
        """Open the circuit breaker. Idempotent."""
        if not cls._billing_disabled:
            cls._billing_disabled = True
            logger.warning(
                "Foursquare circuit-breaker OPEN: %s. Subsequent requests "
                "will skip Foursquare and go straight to Overpass for the "
                "rest of this process.",
                message,
            )

    async def search_tourist_places(
        self,
        city: str,
        *,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        radius_m: int = DEFAULT_RADIUS_M,
        limit: int = 30,
        query: str = DEFAULT_QUERY,
        cache_namespace: str = "tourist",
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return ``(places, cache_hit)``. Places are normalised dicts ready to
        feed to the ranker.
        """
        # Cache namespace is sanitised and length-bounded so it stays
        # well within the 128-char ``place_cache.city`` column.
        ns = "".join(ch for ch in (cache_namespace or "tourist").lower() if ch.isalnum())[:24] or "tourist"

        if lat is not None and lng is not None:
            cache_key = f"fsqv2:{lat:.2f},{lng:.2f}:{radius_m}:{ns}"
            params: dict[str, Any] = {
                "ll": f"{lat},{lng}",
                "radius": radius_m,
                "query": query,
                "limit": limit,
                "sort": "RELEVANCE",
                "fields": FIELDS,
            }
            mode = f"ll={lat:.3f},{lng:.3f} r={radius_m}m q={query!r} ns={ns}"
        else:
            cache_key = f"fsq-name:{city.lower()}:{ns}"
            params = {
                "near": city,
                "query": query,
                "limit": limit,
                "sort": "RELEVANCE",
                "fields": FIELDS,
            }
            mode = f"near={city!r} q={query!r} ns={ns}"

        fresh = cache_helpers.get_fresh_places(
            self.db, cache_key, self.settings.place_cache_ttl_hours
        )
        if fresh is not None:
            logger.debug("places cache hit (%s, %d items)", mode, len(fresh))
            return fresh, True

        # Hard short-circuit BEFORE we waste any time on a known-dead
        # provider. The orchestrator should already have checked this
        # via :meth:`is_disabled`, but we also gate here defensively.
        if PlacesClient.is_disabled():
            stale = cache_helpers.get_stale_places(self.db, cache_key)
            if stale is not None:
                logger.info(
                    "Foursquare disabled (config or breaker); serving stale cache"
                )
                return stale, True
            raise UpstreamError(
                "Foursquare disabled (config or circuit breaker); no stale cache"
            )

        if not self.settings.foursquare_api_key:
            stale = cache_helpers.get_stale_places(self.db, cache_key)
            if stale is not None:
                logger.warning("FOURSQUARE_API_KEY missing, returning stale cache")
                return stale, True
            raise UpstreamError(
                "FOURSQUARE_API_KEY is not configured and no cached places exist"
            )

        headers = {
            "Authorization": f"Bearer {self.settings.foursquare_api_key}",
            "Accept": "application/json",
            "X-Places-Api-Version": FOURSQUARE_API_VERSION,
        }
        logger.info("foursquare fetch (%s)", mode)

        try:
            raw = await request_json(
                "GET", FOURSQUARE_URL, params=params, headers=headers
            )
        except UpstreamError as exc:
            # If this is a billing-related 429, trip the circuit
            # breaker so future requests in this process go straight to
            # Overpass. We pattern-match on the upstream message because
            # 429 by itself can also mean "rate-limited, retry later" -
            # we only want to disable on the permanent "out of credits"
            # variant.
            msg = str(exc).lower()
            if "429" in msg and ("credit" in msg or "billing" in msg):
                PlacesClient.trip_billing_breaker(str(exc)[:160])
            stale = cache_helpers.get_stale_places(self.db, cache_key)
            if stale is not None:
                logger.warning("Foursquare failed, serving stale places cache")
                return stale, True
            raise

        items = raw.get("results") if isinstance(raw, dict) else []
        normalised = [_normalise(p) for p in items or []]
        normalised = [p for p in normalised if p["coords"]["lat"] is not None]

        try:
            cache_helpers.store_places(
                self.db, cache_key, normalised, self.settings.place_cache_ttl_hours
            )
        except Exception:  # pragma: no cover
            logger.exception("failed to store places cache")
        return normalised, False
