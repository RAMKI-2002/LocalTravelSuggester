"""High-level orchestrator for ``POST /suggest-trip``.

Pipeline (numbered so the logs match the README diagram):

    [1] Resolve            -> parallel: geocode(city), geocode(locality),
                              extract_intent(prompt). Yields the
                              (lat,lng) anchor + a structured intent
                              (category + search_keywords + mood).
    [2] Parallel fan-out   -> weather + places (Foursquare anchored,
                              query built from intent.search_keywords)
    [3] Fallback           -> Overpass (anchored, broad tag set
                              including food/shopping) when Foursquare
                              fails
    [4] Distance + filter  -> drop candidates >30 km from anchor
    [5] Rank + curate      -> rule-based shortlist (top 2x), then LLM
                              picks the final ``max_results`` in ONE
                              call (rule-based order kept as fallback)
    [6] Enrich             -> per-place haversine + budget + reasoning
    [7] Persist + respond  -> query_history + structured TripResponse

Why intent extraction matters:

    Without it, every request hit Foursquare with the static query
    ``tourist attractions``. The candidate pool was identical for
    "want to eat something" and "spiritual evening" - the LLM could
    only re-decorate the descriptions; the actual places were the
    same. By parsing the prompt into a structured intent first and
    feeding ``intent.search_keywords`` into Foursquare's ``query``,
    different prompts now fetch genuinely different pools.

Why anchor on lat/lng instead of "city name":

    OSM's named admin boundaries are inconsistent (some cities are tagged
    as huge districts, e.g. Pune covers ~70 km). Querying ``area[name=X]``
    can return places far outside what the user means by "the city".
    Anchoring on Nominatim's geocoded centre + a 25 km radius gives us
    tight, predictable results AND works for cities that aren't tagged
    with ``boundary=administrative``.

Every external call is wrapped so a single upstream failure degrades
gracefully (the partial response is returned with a tag in
``meta.degraded``) instead of breaking the whole call.
"""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.clients.geocode_client import GeoPoint, GeocodeClient
from app.clients.http_base import UpstreamError
from app.clients.llm_client import LLMProvider, get_llm_provider
from app.clients.overpass_client import OverpassClient
from app.clients.places_client import PlacesClient
from app.clients.weather_client import WeatherClient
from app.config import get_settings
from app.db.models import QueryHistory
from app.schemas.trip import (
    Budget,
    Coords,
    PlaceSuggestion,
    TripIntentMeta,
    TripMeta,
    TripRequest,
    TripResponse,
    Weather,
)
from app.services import ranker
from app.services.budget import estimate_budget
from app.services.distance import haversine_km
from app.services.intent_parser import TripIntent, extract_intent

logger = logging.getLogger(__name__)

# Hard cut: drop any candidate further than this from the anchor. Keeps
# Pune-Viman-Nagar from suggesting Lonavala (60+ km away).
MAX_DISTANCE_KM = 30.0
# Search radius used for upstream queries when we have an anchor. 25 km
# is comfortably inside MAX_DISTANCE_KM so the filter rarely fires - the
# filter is a safety net for unanchored fallbacks.
SEARCH_RADIUS_M = 25_000


async def _geocode_safe(
    geocoder: GeocodeClient, query: str, label: str
) -> Optional[GeoPoint]:
    """Run a geocode that never raises - logs and returns None on failure."""
    try:
        result = await geocoder.geocode(query)
    except UpstreamError as exc:
        logger.warning("geocode failed for %s (%r): %s", label, query, exc)
        return None
    if result is None:
        logger.info("geocode produced no result for %s (%r)", label, query)
    return result


async def suggest_trip(db: Session, req: TripRequest) -> TripResponse:
    settings = get_settings()
    max_results = req.max_results or settings.default_max_results

    start = perf_counter()
    cache_hits: list[str] = []
    degraded: list[str] = []

    weather_client = WeatherClient(db)
    places_client = PlacesClient(db)
    overpass_client = OverpassClient(db)
    geocoder = GeocodeClient()
    llm: LLMProvider = get_llm_provider()

    logger.info(
        "[1/7] resolve: geocode city/locality + extract intent "
        "(city=%r locality=%r prompt=%r max_results=%d)",
        req.city,
        req.locality,
        req.preference,
        max_results,
    )

    # ---- [1] Geocode + intent extraction, all in parallel ----------------
    # We always need the city centre (anchor for upstream queries when no
    # locality is provided). When a locality is given we *also* geocode it
    # and prefer it as the anchor. Intent extraction runs alongside so we
    # don't add latency.
    city_geo_task = asyncio.create_task(
        _geocode_safe(geocoder, req.city, "city")
    )
    if req.locality:
        locality_geo_task = asyncio.create_task(
            _geocode_safe(geocoder, f"{req.locality}, {req.city}", "locality")
        )
    else:
        locality_geo_task = None
    intent_task = asyncio.create_task(extract_intent(req.preference, llm))

    city_point = await city_geo_task
    user_point: Optional[GeoPoint] = None
    if locality_geo_task is not None:
        user_point = await locality_geo_task
    intent: TripIntent = await intent_task

    if req.locality and user_point is None:
        degraded.append("geocode_locality")
    if city_point is None:
        degraded.append("geocode_city")

    # Pick the anchor used for upstream queries: locality > city > none.
    anchor_lat: Optional[float] = None
    anchor_lng: Optional[float] = None
    if user_point is not None:
        anchor_lat, anchor_lng = user_point.lat, user_point.lng
        anchor_label = f"locality {req.locality!r}"
    elif city_point is not None:
        anchor_lat, anchor_lng = city_point.lat, city_point.lng
        anchor_label = f"city {req.city!r}"
    else:
        anchor_label = "none (will fall back to name-based queries)"
    logger.info(
        "[1/7] anchor=%s -> (%s, %s); intent=%s mood=%s keywords=%s (source=%s)",
        anchor_label,
        anchor_lat,
        anchor_lng,
        intent.category,
        intent.mood,
        intent.search_keywords,
        intent.source,
    )

    # ---- [2] Parallel fan-out: weather + places --------------------------
    # Provider selection:
    #   * If Foursquare is enabled AND its circuit-breaker is closed,
    #     we hit it first (richer data, structured ratings).
    #   * Otherwise we go straight to Overpass - no point burning 2-3
    #     seconds on a provider we already know is dead.
    # The provider is named in the log so the demo viewer can tell at a
    # glance which path ran.
    fsq_disabled = PlacesClient.is_disabled()
    primary_label = "overpass" if fsq_disabled else "foursquare"
    logger.info(
        "[2/7] fan-out: weather + places(primary=%s query=%r ns=%s%s)",
        primary_label,
        intent.query_string,
        intent.category,
        "  [foursquare disabled]" if fsq_disabled else "",
    )
    weather_task = asyncio.create_task(weather_client.get(req.city))

    if fsq_disabled:
        # Skip Foursquare entirely. Overpass is the primary now.
        places_task = asyncio.create_task(
            overpass_client.search_tourist_places(
                req.city,
                lat=anchor_lat,
                lng=anchor_lng,
                radius_m=SEARCH_RADIUS_M,
            )
        )
        degraded.append("places_primary_skipped")
    else:
        places_task = asyncio.create_task(
            places_client.search_tourist_places(
                req.city,
                lat=anchor_lat,
                lng=anchor_lng,
                radius_m=SEARCH_RADIUS_M,
                query=intent.query_string,
                cache_namespace=intent.category,
            )
        )

    try:
        weather_payload, weather_cached = await weather_task
        logger.info(
            "weather: %s @ %s (cached=%s)",
            weather_payload.get("condition"),
            weather_payload.get("temp_c"),
            weather_cached,
        )
    except UpstreamError as exc:
        logger.error("weather hard-failed: %s", exc)
        weather_payload, weather_cached = {}, False
        degraded.append("weather")
    if weather_cached:
        cache_hits.append("weather")

    # ---- [3] Places + fallback chain -------------------------------------
    # Two execution paths, depending on whether we started with Foursquare:
    #   * If FSQ was the primary and it raised: try Overpass.
    #   * If Overpass was already the primary (because FSQ is disabled) and
    #     IT raised: there is nothing left to fall back to.
    try:
        places_raw, places_cached = await places_task
        logger.info(
            "places(%s): %d items (cached=%s)",
            primary_label,
            len(places_raw),
            places_cached,
        )
    except UpstreamError as exc:
        logger.warning(
            "[3/7] %s unavailable (%s)%s",
            primary_label,
            exc,
            "; falling back to overpass" if not fsq_disabled else "",
        )
        if fsq_disabled:
            # Overpass *was* the primary. No fallback left.
            places_raw, places_cached = [], False
            degraded.append("places")
        else:
            # FSQ failed, try Overpass.
            degraded.append("places_primary")
            try:
                places_raw, places_cached = await overpass_client.search_tourist_places(
                    req.city,
                    lat=anchor_lat,
                    lng=anchor_lng,
                    radius_m=SEARCH_RADIUS_M,
                )
                logger.info(
                    "places(overpass): %d items (cached=%s)",
                    len(places_raw),
                    places_cached,
                )
                degraded.append("places_fallback_overpass")
            except UpstreamError as fb_exc:
                logger.error("[3/7] overpass fallback also failed: %s", fb_exc)
                places_raw, places_cached = [], False
                degraded.append("places")
    if places_cached:
        cache_hits.append("places")

    # ---- [4] Compute distance + filter outliers --------------------------
    # We compute distance from the *anchor* (locality if provided, else
    # city) and drop everything beyond MAX_DISTANCE_KM. This guarantees
    # the user never sees a "Lonavala 70 km away" suggestion when they
    # asked about Viman Nagar.
    if anchor_lat is not None and anchor_lng is not None:
        before = len(places_raw)
        annotated: list[dict[str, Any]] = []
        for p in places_raw:
            coords = p.get("coords") or {}
            lat = coords.get("lat")
            lng = coords.get("lng")
            if lat is None or lng is None:
                continue
            d = haversine_km(anchor_lat, anchor_lng, lat, lng)
            if d > MAX_DISTANCE_KM:
                continue
            p["_distance_km"] = round(d, 2)
            # Distance from the user's locality specifically (for display).
            # If there is no locality, this stays the same as anchor distance.
            if user_point is not None:
                p["_distance_km_user"] = round(
                    haversine_km(user_point.lat, user_point.lng, lat, lng), 2
                )
            else:
                p["_distance_km_user"] = None
            annotated.append(p)
        logger.info(
            "[4/7] distance filter: %d -> %d within %dkm of %s",
            before,
            len(annotated),
            int(MAX_DISTANCE_KM),
            anchor_label,
        )
        places_raw = annotated
    else:
        # No anchor: still annotate distance as None so ranker uses neutral
        # proximity score and downstream code stays happy.
        for p in places_raw:
            p["_distance_km"] = None
            p["_distance_km_user"] = None
        logger.info(
            "[4/7] distance filter skipped (no anchor); %d candidates",
            len(places_raw),
        )

    # ---- [5] Rank + (optional) LLM curate --------------------------------
    # We build the "effective preference" by combining the canonical
    # intent category with the original prompt. This way the ranker
    # matches BOTH the structured intent ("food" -> food bucket) AND
    # any extra mood/qualifier words ("peaceful temple" -> spiritual
    # bucket via the original prompt) at full strength.
    effective_preference = " ".join(
        s for s in [intent.category, intent.mood or "", req.preference or ""] if s
    ).strip() or None

    # Rule-based ranker scores every candidate using weather, preference,
    # popularity, AND proximity. We take a *shortlist* (top 2x or 8, whichever
    # is larger) and ask the LLM to pick the final max_results from it. If
    # the LLM is unavailable / mocked / hallucinates everything, we fall
    # back to the rule-based top-N.
    shortlist_size = max(max_results * 2, 8)
    logger.info(
        "[5/7] ranking %d candidates -> shortlist of %d",
        len(places_raw),
        shortlist_size,
    )
    shortlist = ranker.rank_places(
        places_raw, weather_payload, effective_preference, shortlist_size
    )
    logger.info(
        "[5/7] rule-based shortlist: %s",
        [(p.get("name"), p.get("_score")) for p in shortlist],
    )

    if not shortlist:
        elapsed_ms = int((perf_counter() - start) * 1000)
        logger.warning(
            "no suggestions after ranking (places=%d, degraded=%s)",
            len(places_raw),
            degraded,
        )
        response = TripResponse(
            city=req.city,
            weather=Weather(**_safe_weather(weather_payload)),
            user_location=_user_coords(user_point),
            suggestions=[],
            meta=TripMeta(
                elapsed_ms=elapsed_ms,
                cache_hits=cache_hits,
                degraded=sorted(set(degraded + (["no_places"] if not places_raw else []))),
                llm_provider=type(llm).__name__,
                intent=_intent_meta(intent),
            ),
        )
        _persist(db, req, response, elapsed_ms)
        return response

    final_places: list[dict[str, Any]]
    used_curate = False
    try:
        # We hand the LLM the *original* prompt (req.preference) so the
        # curate reason mentions what the user actually said - but the
        # rule-based pre-rank used the canonical category, which is what
        # makes the candidate pool correct in the first place.
        curated = await llm.curate_places(
            weather_payload,
            req.preference,
            req.locality,
            shortlist,
            max_results,
        )
    except Exception as exc:
        logger.warning("LLM curate raised, falling back: %s", exc)
        curated = None

    if curated:
        # Curate already attaches `_reasoning`. Top up with rule-based
        # picks if the LLM under-delivered (returned fewer than asked).
        used_curate = True
        if len(curated) < max_results:
            chosen_names = {(p.get("name") or "").lower() for p in curated}
            for p in shortlist:
                if (p.get("name") or "").lower() in chosen_names:
                    continue
                curated.append(p)
                if len(curated) >= max_results:
                    break
        final_places = curated[:max_results]
        logger.info(
            "[5/7] LLM curate accepted %d picks: %s",
            len(final_places),
            [p.get("name") for p in final_places],
        )
    else:
        # Re-rank the shortlist down to max_results with the diversity cap.
        # Only flag llm_curate as degraded when we expected the LLM to do
        # the job but it failed - i.e. we had a non-empty shortlist AND a
        # real (non-mock) provider.
        if shortlist and "Mock" not in type(llm).__name__:
            degraded.append("llm_curate")
        final_places = ranker.rank_places(
            places_raw, weather_payload, effective_preference, max_results
        )
        logger.info(
            "[5/7] curate unavailable - using rule-based top %d: %s",
            len(final_places),
            [p.get("name") for p in final_places],
        )

    # ---- [6] Enrich: per-place reasoning (only if curate didn't supply) +
    #         budget. Distances are already computed in step [4].
    logger.info("[6/7] enriching %d places (budget + reasoning)", len(final_places))
    enriched: list[dict[str, Any]] = []
    needs_reasoning: list[dict[str, Any]] = []
    for place in final_places:
        place["_budget"] = estimate_budget(place, place.get("_distance_km_user"))
        if "_reasoning" not in place:
            needs_reasoning.append(place)
        enriched.append(place)

    if needs_reasoning:
        logger.info(
            "[6/7] %d places need per-place reasoning (curate=%s)",
            len(needs_reasoning),
            used_curate,
        )

        async def _reason(place: dict[str, Any]) -> None:
            try:
                place["_reasoning"] = await llm.generate_place_reasoning(
                    weather_payload, req.preference, place
                )
            except Exception as exc:
                logger.warning(
                    "LLM reasoning failed for %s, using rule fallback: %s",
                    place.get("name"),
                    exc,
                )
                degraded.append("llm")
                place["_reasoning"] = _fallback_reasoning(
                    weather_payload, place, req.preference
                )

        await asyncio.gather(*(_reason(p) for p in needs_reasoning))

    # ---- [7] Assemble response -------------------------------------------
    # Only surface distance to the *user's* locality. If no locality was
    # given, leave distance_km null so the UI doesn't show a misleading
    # "city-centre distance" (the user never asked for that anchor).
    suggestions = [
        PlaceSuggestion(
            name=p.get("name") or "Unknown",
            description=p.get("description") or "",
            categories=p.get("categories") or [],
            reasoning=p.get("_reasoning") or "",
            coords=Coords(
                lat=(p.get("coords") or {}).get("lat"),
                lng=(p.get("coords") or {}).get("lng"),
            ),
            distance_km=p.get("_distance_km_user"),
            estimated_budget=Budget(**p["_budget"]),
            score=p.get("_score", 0.0),
            website=p.get("website"),
            address=p.get("address"),
        )
        for p in enriched
    ]

    elapsed_ms = int((perf_counter() - start) * 1000)
    degraded = sorted(set(degraded))

    response = TripResponse(
        city=req.city,
        weather=Weather(**_safe_weather(weather_payload)),
        user_location=_user_coords(user_point),
        suggestions=suggestions,
        meta=TripMeta(
            elapsed_ms=elapsed_ms,
            cache_hits=cache_hits,
            degraded=degraded,
            llm_provider=type(llm).__name__,
            llm_curate_used=used_curate,
            intent=_intent_meta(intent),
        ),
    )

    logger.info(
        "[7/7] responding with %d suggestions in %d ms (curate=%s, cache=%s, degraded=%s)",
        len(suggestions),
        elapsed_ms,
        used_curate,
        cache_hits,
        degraded,
    )
    _persist(db, req, response, elapsed_ms)
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_weather(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "temp_c": payload.get("temp_c"),
        "feels_like_c": payload.get("feels_like_c"),
        "condition": payload.get("condition"),
        "description": payload.get("description"),
        "humidity": payload.get("humidity"),
        "wind_kph": payload.get("wind_kph"),
    }


def _user_coords(point: Optional[GeoPoint]) -> Optional[Coords]:
    if point is None:
        return None
    return Coords(lat=point.lat, lng=point.lng)


def _intent_meta(intent: TripIntent) -> TripIntentMeta:
    """Project the internal TripIntent into the API-facing schema."""
    return TripIntentMeta(
        category=intent.category,
        search_keywords=intent.search_keywords,
        mood=intent.mood,
        source=intent.source,
    )


def _fallback_reasoning(
    weather: dict[str, Any], place: dict[str, Any], preference: Optional[str]
) -> str:
    """Deterministic one-liner used when the LLM is unavailable."""
    cond = (weather.get("condition") or "current").lower()
    temp = weather.get("temp_c")
    pref = f" (fits '{preference}')" if preference else ""
    return (
        f"{place.get('name')} is recommended given the {cond} weather"
        + (f" at {temp}C" if temp is not None else "")
        + f"{pref}."
    )


def _persist(
    db: Session, req: TripRequest, response: TripResponse, latency_ms: int
) -> None:
    try:
        entry = QueryHistory(
            city=req.city,
            preference=req.preference,
            locality=req.locality,
            response=response.model_dump(),
            latency_ms=latency_ms,
        )
        db.add(entry)
        db.commit()
        logger.debug("persisted query_history id=%s latency=%dms", entry.id, latency_ms)
    except Exception:  # pragma: no cover - never fail the response on history write
        logger.exception("failed to persist query history")
        db.rollback()
