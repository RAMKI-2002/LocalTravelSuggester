"""OpenStreetMap Overpass API client - free no-key fallback for tourist POIs.

Used when the primary places provider (Foursquare) fails: 429, out of
credits, auth error, network outage, etc. OSM data is community-maintained
and well-populated in major cities, and the Overpass API requires neither
an API key nor a billing relationship, which makes it a robust last-resort
source.

Response shape is normalised to the same dict our ``ranker`` / ``budget`` /
``trip_service`` layers already consume, so callers don't need to care
which provider served the data.

Quirks we explicitly handle:
  * Overpass returns ``node`` (has lat/lon) and ``way`` (has ``center``)
    elements - we flatten both.
  * OSM tags are free-form; we synthesise a small category list from the
    ``tourism`` / ``historic`` / ``leisure`` tags that drive ranking.
  * The public Overpass endpoint is rate-limited per-IP; we use a single
    slower timeout rather than hammering it on retry.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
from sqlalchemy.orm import Session

from app.clients.http_base import UpstreamError
from app.config import get_settings
from app.db import cache as cache_helpers

logger = logging.getLogger(__name__)

# Public Overpass mirrors. We try them in order on transport errors / 5xx
# / 429 - the main `overpass-api.de` instance is frequently overloaded
# (15-30s response times during peak hours) and silently times out.
# `kumi.systems` and `private.coffee` are the two most reliable
# community mirrors and are explicitly listed by openstreetmap.org as
# alternatives for production use. The list is small on purpose so a
# single bad request never burns more than ~3x the timeout.
OVERPASS_MIRRORS: list[str] = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
# httpx wall-clock per mirror. The QL itself sets ``[timeout:25]`` so
# this needs to be larger than that or the client cuts off mid-query.
OVERPASS_HTTP_TIMEOUT_S: float = 30.0
# Tiny query used by ``ping_overpass`` for health checks - always
# returns one line, so we get a real status code without hitting the
# server's expensive geo-index. Crucially, it *has* a body, so Overpass
# does not return 406 like it does for an empty POST.
OVERPASS_PING_QUERY: str = "[out:json];out;"

# Tag values that correspond to tourist-friendly POIs. Anything not in these
# sets is filtered out (e.g. ``tourism=hotel`` or ``historic=boundary_stone``
# which we don't want to suggest as attractions).
TOURISM_TAGS = {
    "attraction",
    "museum",
    "viewpoint",
    "gallery",
    "theme_park",
    "zoo",
    "aquarium",
    "artwork",
    "monument",
}
LEISURE_TAGS = {"park", "garden", "nature_reserve", "water_park"}
# Amenities are a mixed bag - we include:
#   * ``place_of_worship`` -> temples / mosques / churches (spiritual prompts)
#   * ``marketplace`` -> traditional markets (shopping / food prompts)
#   * ``restaurant`` / ``cafe`` / ``bar`` / ``pub`` / ``fast_food`` /
#     ``food_court`` / ``ice_cream`` / ``biergarten`` -> food prompts
# Without these, prompts like "want to eat" returned zero food places
# and the ranker had to pick from the same tourist-only pool, which is
# why the demo felt static across prompts.
AMENITY_TAGS = {
    "place_of_worship",
    "marketplace",
    "restaurant",
    "cafe",
    "bar",
    "pub",
    "fast_food",
    "food_court",
    "ice_cream",
    "biergarten",
}
# Some natural features double as tourist attractions and rarely have a
# tourism tag (e.g. lakes, hills, beaches).
NATURAL_TAGS = {"water", "beach", "hill", "peak", "wood", "cliff"}
# Shop tags worth suggesting. We deliberately keep this list short -
# generic ``shop=convenience`` is not interesting; ``shop=mall`` and
# similar ARE.
SHOP_TAGS = {
    "mall",
    "department_store",
    "supermarket",
    "books",
    "gift",
    "jewelry",
    "art",
    "craft",
}


def _build_query_around(lat: float, lng: float, radius_m: int, limit: int) -> str:
    """Build an Overpass QL query anchored on a point (around:radius).

    This is the preferred query shape because:
      * It is robust to inconsistent OSM admin boundaries (some cities
        don't carry ``boundary=administrative``).
      * It guarantees results are within ``radius_m`` of the user's
        anchor, so we never surface places 60-70 km away from the city
        the user actually picked.
      * It works for any city as long as we have a lat/lng for it.
    """
    tourism_regex = "|".join(sorted(TOURISM_TAGS))
    leisure_regex = "|".join(sorted(LEISURE_TAGS))
    amenity_regex = "|".join(sorted(AMENITY_TAGS))
    natural_regex = "|".join(sorted(NATURAL_TAGS))
    shop_regex = "|".join(sorted(SHOP_TAGS))
    around = f"around:{int(radius_m)},{lat},{lng}"
    return f"""
[out:json][timeout:25];
(
  node["tourism"~"^({tourism_regex})$"]({around});
  way ["tourism"~"^({tourism_regex})$"]({around});
  node["historic"]({around});
  way ["historic"]({around});
  node["leisure"~"^({leisure_regex})$"]({around});
  way ["leisure"~"^({leisure_regex})$"]({around});
  node["amenity"~"^({amenity_regex})$"]({around});
  way ["amenity"~"^({amenity_regex})$"]({around});
  node["natural"~"^({natural_regex})$"]({around});
  way ["natural"~"^({natural_regex})$"]({around});
  node["shop"~"^({shop_regex})$"]({around});
  way ["shop"~"^({shop_regex})$"]({around});
);
out center tags {limit};
""".strip()


def _build_query_by_name(city: str, limit: int) -> str:
    """Fallback query: by named admin area.

    Used only when we don't have a lat/lng anchor (e.g. Nominatim was
    unreachable). Less precise than the around-query because some cities
    are tagged as huge districts (Pune district covers ~70 km).
    """
    tourism_regex = "|".join(sorted(TOURISM_TAGS))
    leisure_regex = "|".join(sorted(LEISURE_TAGS))
    amenity_regex = "|".join(sorted(AMENITY_TAGS))
    natural_regex = "|".join(sorted(NATURAL_TAGS))
    shop_regex = "|".join(sorted(SHOP_TAGS))
    safe_city = city.replace('"', "")
    return f"""
[out:json][timeout:25];
area["name"="{safe_city}"]["boundary"="administrative"]->.searchArea;
(
  node["tourism"~"^({tourism_regex})$"](area.searchArea);
  way ["tourism"~"^({tourism_regex})$"](area.searchArea);
  node["historic"](area.searchArea);
  way ["historic"](area.searchArea);
  node["leisure"~"^({leisure_regex})$"](area.searchArea);
  way ["leisure"~"^({leisure_regex})$"](area.searchArea);
  node["amenity"~"^({amenity_regex})$"](area.searchArea);
  way ["amenity"~"^({amenity_regex})$"](area.searchArea);
  node["natural"~"^({natural_regex})$"](area.searchArea);
  way ["natural"~"^({natural_regex})$"](area.searchArea);
  node["shop"~"^({shop_regex})$"](area.searchArea);
  way ["shop"~"^({shop_regex})$"](area.searchArea);
);
out center tags {limit};
""".strip()


def _extract_categories(tags: dict[str, Any]) -> list[str]:
    """Turn OSM free-form tags into a small, human-readable category list
    the ranker can match against (e.g. 'Museum', 'Hindu Temple', 'Park').

    The richer this list, the more precisely the rule-based ranker can
    align places with user preferences ("spiritual", "peaceful",
    "shopping", etc.) - so we surface as many relevant tags as we can.
    """
    cats: list[str] = []
    tourism = tags.get("tourism")
    historic = tags.get("historic")
    leisure = tags.get("leisure")
    amenity = tags.get("amenity")
    natural = tags.get("natural")
    religion = tags.get("religion")
    cuisine = tags.get("cuisine")
    shop = tags.get("shop")

    if tourism:
        cats.append(tourism.replace("_", " ").title())
    if historic:
        cats.append(f"Historic {historic.replace('_', ' ').title()}")
    if leisure:
        cats.append(leisure.replace("_", " ").title())
    if amenity == "place_of_worship":
        # religion-prefixed labels make spiritual queries land cleanly
        if religion:
            cats.append(f"{religion.title()} Temple")
        else:
            cats.append("Place of Worship")
    elif amenity == "marketplace":
        cats.append("Market")
    elif amenity in {"restaurant", "cafe", "bar", "pub", "fast_food",
                     "food_court", "ice_cream", "biergarten"}:
        # Food amenities. Cuisine-prefixed labels ("Italian Restaurant")
        # are far more useful to the ranker than just "Restaurant".
        label = amenity.replace("_", " ").title()
        if cuisine and amenity in {"restaurant", "cafe", "fast_food", "food_court"}:
            cats.append(f"{cuisine.title()} {label}")
        else:
            cats.append(label)
    if natural:
        cats.append(natural.replace("_", " ").title())
    if shop:
        # Friendly labels: "Mall", "Supermarket", etc. ``shop`` tag
        # values are already snake-case-ish.
        cats.append(shop.replace("_", " ").title())
        cats.append("Shopping")
    return cats


def _normalise(element: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Convert one Overpass element to our standard place dict. Returns
    ``None`` for rows missing a name or coordinates - those cannot be
    suggested to the user usefully.
    """
    tags = element.get("tags") or {}
    name = tags.get("name") or tags.get("name:en")
    if not name:
        return None

    if element.get("type") == "node":
        lat = element.get("lat")
        lng = element.get("lon")
    else:  # way / relation
        center = element.get("center") or {}
        lat = center.get("lat")
        lng = center.get("lon")

    if lat is None or lng is None:
        return None

    address_parts = [
        tags.get("addr:housenumber"),
        tags.get("addr:street"),
        tags.get("addr:suburb") or tags.get("addr:neighbourhood"),
        tags.get("addr:city"),
    ]
    address = ", ".join(p for p in address_parts if p) or None

    return {
        "fsq_id": f"osm:{element.get('type')}/{element.get('id')}",
        "name": name,
        "description": tags.get("description") or tags.get("wikipedia") or "",
        "categories": _extract_categories(tags),
        "coords": {"lat": lat, "lng": lng},
        "address": address,
        "price_tier": None,  # OSM doesn't model pricing
        "rating": None,      # No ratings in OSM
        "popularity": None,  # No popularity signal in OSM
        "website": tags.get("website") or tags.get("contact:website"),
    }


async def _post_overpass_one(
    url: str, query: str, timeout_s: float
) -> dict[str, Any]:
    """Single Overpass POST. Raises ``UpstreamError`` for any failure
    (transport, 4xx, 5xx, non-JSON) so the caller can decide whether to
    try the next mirror.

    We bypass the shared ``httpx.AsyncClient`` because it ships with
    ``Accept: application/json`` which triggers a 406 from Apache's
    content negotiation at overpass-api.de when the body is empty. We
    use POST with a form-encoded body (``data=<query>``) - that's the
    canonical Overpass call pattern, works for queries of any size, and
    matches what the ``overpy`` Python library does.

    A descriptive User-Agent is required by OSM's acceptable-use policy
    and also avoids generic-python-client filtering that some mirrors
    apply.
    """
    headers = {
        "Accept": "application/json",
        "User-Agent": "local-trip-suggester/1.0 (+tourist-api-demo)",
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
            resp = await client.post(url, data={"data": query}, headers=headers)
    except httpx.HTTPError as exc:
        raise UpstreamError(f"overpass transport error ({url}): {exc}") from exc

    if resp.status_code == 429:
        raise UpstreamError(
            f"overpass 429 ({url}, retry-after={resp.headers.get('retry-after', '?')})"
        )
    if resp.status_code >= 400:
        raise UpstreamError(
            f"overpass {resp.status_code} ({url}): {resp.text[:300]}"
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise UpstreamError(
            f"overpass non-JSON response ({url}): {resp.text[:200]}"
        ) from exc


async def _fetch_overpass(query: str, timeout_s: float) -> dict[str, Any]:
    """Run an Overpass query, failing over across mirrors.

    The public ``overpass-api.de`` instance is regularly overloaded
    during peak hours - a single user query can hang for 30s and then
    return a 504 or just close the connection. To stay responsive we
    try each mirror in :data:`OVERPASS_MIRRORS` in order and only give
    up after all of them have failed.

    Per-mirror failures are logged at WARNING; a global failure
    (everyone died) raises :class:`UpstreamError` with the *last*
    error so the caller can degrade gracefully.
    """
    last_exc: Optional[UpstreamError] = None
    for url in OVERPASS_MIRRORS:
        try:
            return await _post_overpass_one(url, query, timeout_s)
        except UpstreamError as exc:
            logger.warning("overpass mirror failed (%s): %s", url, exc)
            last_exc = exc
            continue
    # All mirrors failed - re-raise the last error so the caller sees
    # something actionable (with the URL inlined for log triage).
    raise last_exc or UpstreamError("overpass: all mirrors failed (no error captured)")


async def ping_overpass(timeout_s: float = 5.0) -> tuple[bool, Optional[str]]:
    """Tiny health probe: ``True`` if at least one mirror responds 200
    to a 1-byte query, ``False`` otherwise.

    Used by ``/health/detailed``. The returned string identifies which
    mirror replied (or which error came back when all of them failed)
    so the dashboard can show *what* is broken, not just *that*
    something is broken.
    """
    last_err: Optional[str] = None
    for url in OVERPASS_MIRRORS:
        try:
            await _post_overpass_one(url, OVERPASS_PING_QUERY, timeout_s)
            return True, url
        except UpstreamError as exc:
            last_err = str(exc)
            continue
    return False, last_err


class OverpassClient:
    """Cache-through wrapper around the public Overpass API.

    Shares the same ``place_cache`` table as the Foursquare client but under
    a distinct cache key namespace so stale Foursquare data doesn't mask a
    working Overpass fetch (or vice versa). The cache key includes the
    anchor lat/lng (rounded to 0.01°, ~1.1 km) so different localities
    inside the same city cache separately.
    """

    DEFAULT_RADIUS_M = 25_000  # 25 km - covers most metros end-to-end

    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()

    async def search_tourist_places(
        self,
        city: str,
        *,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        radius_m: int = DEFAULT_RADIUS_M,
        limit: int = 80,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return ``(places, cache_hit)`` of tourist POIs.

        Two query modes:
          * **Anchored** (``lat``+``lng`` provided): an ``around:radius``
            query - guarantees every place returned is within
            ``radius_m`` metres of the anchor. Strongly preferred.
          * **Named-area fallback** (no anchor): an ``area[name=...]``
            query - used when geocoding failed. Less precise (some
            cities have huge admin areas) but still functional.
        """
        if lat is not None and lng is not None:
            cache_key = f"osmv2:{lat:.2f},{lng:.2f}:{radius_m}"
            query = _build_query_around(lat, lng, radius_m, limit)
            mode = f"around:{radius_m}m@{lat:.3f},{lng:.3f}"
        else:
            cache_key = f"osm-name:{city.lower()}"
            query = _build_query_by_name(city, limit)
            mode = f"name={city!r}"

        fresh = cache_helpers.get_fresh_places(
            self.db, cache_key, self.settings.place_cache_ttl_hours
        )
        if fresh is not None:
            logger.debug("overpass cache hit (%s, %d items)", mode, len(fresh))
            return fresh, True

        logger.info("overpass fetch (%s)", mode)
        try:
            # Use the dedicated long-form timeout (30s default), not the
            # global ``http_timeout_seconds`` (10s) - Overpass's QL
            # ``[timeout:25]`` clause means valid responses can legitimately
            # take ~20-25s, especially for big regions. Cutting off at 10s
            # would turn slow-but-successful queries into transport errors.
            raw = await _fetch_overpass(query, OVERPASS_HTTP_TIMEOUT_S)
        except UpstreamError:
            stale = cache_helpers.get_stale_places(self.db, cache_key)
            if stale is not None:
                logger.warning("Overpass failed, serving stale OSM cache")
                return stale, True
            raise

        elements = raw.get("elements") if isinstance(raw, dict) else []
        normalised: list[dict[str, Any]] = []
        for el in elements or []:
            place = _normalise(el)
            if place is not None:
                normalised.append(place)

        # OSM can return hundreds of small amenities; trim to the request
        # limit so the ranker isn't doing pointless work.
        normalised = normalised[:limit]

        try:
            cache_helpers.store_places(
                self.db, cache_key, normalised, self.settings.place_cache_ttl_hours
            )
        except Exception:  # pragma: no cover
            logger.exception("failed to store overpass cache")
        return normalised, False
