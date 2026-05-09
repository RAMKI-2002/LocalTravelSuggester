"""Rule-based place ranker.

Why rule-based first (interview talking point):

    Filtering / ranking is cheap, deterministic, and explainable. We do
    NOT use the LLM for the *broad cut* because:
        1. it is expensive (one paid call per candidate),
        2. it is non-deterministic - two identical requests could
           return different orderings,
        3. it can silently drop or hallucinate candidates.

    What the LLM *does* do is the FINAL fine-grained selection: see
    ``llm_client.curate_places`` - we hand it the rule-based top-2x
    pool and ask it to pick the user's best ``max_results`` matches in
    a single call.

    Rule-based score (weights tuned for *intent variety* - we want the
    same city to surface different places when the user asks for
    different things, and we want nearby places preferred over distant
    ones):

            0.20 * weather_fit
          + 0.45 * preference_match     <-- dominant signal
          + 0.15 * popularity / rating
          + 0.20 * proximity            <-- 1.0 at 0 km, 0 at >=25 km

    Preference matching looks at categories AND the place's name +
    description, because OSM categories are often generic ("Attraction")
    while names carry the real signal ("Snow World" => fun, "Birla
    Mandir" => spiritual).

    On top of the score we apply a diversity pass: at most two places
    of the same primary category in the final top-N. Without this you
    get five museums in a row for "history" prompts, which is worse
    UX than mixing in one fort + one monument + one heritage building.
"""

from __future__ import annotations

from typing import Any, Optional

# Category buckets for weather-fit. Substring-matched on lower-cased
# category strings so we handle the wide variety of Foursquare and OSM
# labels uniformly.
_OUTDOOR_KEYWORDS = {
    "park", "garden", "lake", "beach", "fort", "hill", "trek", "waterfall",
    "viewpoint", "scenic", "outdoor", "zoo", "safari", "monument", "memorial",
    "natural", "water", "peak",
}
_INDOOR_KEYWORDS = {
    "museum", "gallery", "aquarium", "planetarium", "mall", "cafe",
    "restaurant", "theatre", "cinema", "library", "temple", "mosque",
    "church", "shrine", "spa", "place of worship",
}

# Map free-text user intents to category keywords. Keys are matched by
# substring against the lower-cased preference string, so "I want a
# peaceful evening" hits the "peaceful" bucket cleanly.
_PROMPT_BUCKETS: dict[str, set[str]] = {
    "peaceful": {"park", "garden", "lake", "temple", "monastery", "shrine",
                 "viewpoint", "beach", "place of worship"},
    "calm":     {"park", "garden", "lake", "temple", "viewpoint", "place of worship"},
    "quiet":    {"park", "garden", "library", "place of worship", "temple"},
    "adventure": {"fort", "trek", "hill", "water_park", "waterpark",
                  "adventure", "amusement", "zoo", "safari", "peak", "cliff"},
    "thrill":   {"theme park", "amusement", "fort", "peak", "water_park"},
    "food":     {"market", "cafe", "restaurant", "bakery", "street",
                 "cuisine", "bar", "pub", "fast food", "food court",
                 "ice cream", "biergarten", "biryani"},
    "history":  {"fort", "museum", "monument", "memorial", "heritage",
                 "palace", "ruins", "historic", "archaeological"},
    "historic": {"fort", "museum", "monument", "memorial", "historic",
                 "palace", "heritage"},
    "heritage": {"fort", "monument", "heritage", "historic", "palace"},
    "family":   {"park", "zoo", "aquarium", "museum", "planetarium",
                 "amusement", "mall", "theme park"},
    "kids":     {"zoo", "aquarium", "park", "amusement", "theme park", "planetarium"},
    "shopping": {"mall", "market", "bazaar", "shop", "shopping"},
    "spiritual": {"temple", "mosque", "church", "shrine", "monastery",
                  "place of worship"},
    "religious": {"temple", "mosque", "church", "shrine", "place of worship"},
    "nature":   {"park", "garden", "lake", "hill", "waterfall", "beach",
                 "forest", "viewpoint", "natural", "water", "wood"},
    "view":     {"viewpoint", "hill", "fort", "peak", "tower"},
    "scenic":   {"viewpoint", "hill", "lake", "garden", "beach"},
    "art":      {"gallery", "artwork", "museum", "theatre"},
    "romantic": {"viewpoint", "garden", "lake", "beach", "park"},
    # Nightlife: lounges, rooftop bars, clubs (mostly Foursquare).
    "nightlife": {"bar", "pub", "lounge", "nightclub", "rooftop",
                  "club", "live music"},
}


def _categories_text(place: dict[str, Any]) -> str:
    return " ".join(place.get("categories") or []).lower()


def _intent_text(place: dict[str, Any]) -> str:
    """Lower-case bag-of-words of the place's name, description AND
    categories. Used for prompt matching - many OSM POIs only carry the
    generic ``tourism=attraction`` tag, but the *name* often carries the
    real signal (e.g. "Lumbini Park", "Birla Mandir", "Snow World"). If
    we matched on categories alone, "spiritual" would tie all of them.
    """
    parts = [
        str(place.get("name") or ""),
        str(place.get("description") or ""),
        _categories_text(place),
    ]
    return " ".join(parts).lower()


def _weather_fit(place: dict[str, Any], weather: dict[str, Any]) -> float:
    """0..1 score. Rainy -> indoor favoured; hot/sunny -> outdoor favoured."""
    text = _categories_text(place)
    is_outdoor = any(k in text for k in _OUTDOOR_KEYWORDS)
    is_indoor = any(k in text for k in _INDOOR_KEYWORDS)

    condition = (weather.get("condition") or "").lower()
    temp = weather.get("temp_c")

    if "rain" in condition or "storm" in condition or "snow" in condition:
        if is_indoor:
            return 1.0
        if is_outdoor:
            return 0.15
        return 0.6

    if isinstance(temp, (int, float)) and temp >= 36:
        if is_indoor:
            return 0.9
        if is_outdoor:
            return 0.45
        return 0.65

    if "clear" in condition or "sun" in condition or "cloud" in condition:
        if is_outdoor:
            return 1.0
        if is_indoor:
            return 0.7
        return 0.8

    return 0.7


def _prompt_match(place: dict[str, Any], preference: Optional[str]) -> float:
    """0..1 score for "does this place match what the user asked for".

    The scoring is deliberately punchy: a place that matches a known
    intent bucket scores 1.0, a place with no signal at all scores
    near-zero. That way two different prompts on the same candidate
    pool produce visibly different rankings.

    Matching considers the place's *name + description + categories*,
    not just categories. OSM often returns "Attraction" as the only
    category - in those cases the name ("Lumbini Park", "Snow World",
    "Birla Mandir") is the only intent signal we have.
    """
    if not preference:
        return 0.5  # neutral - no preference to match against

    pref_lower = preference.lower()
    text = _intent_text(place)
    if not text.strip():
        return 0.2

    # Pass 1: known intent buckets ("peaceful", "history", ...)
    matched_buckets = 0
    bucket_hits = 0
    for bucket_key, keywords in _PROMPT_BUCKETS.items():
        if bucket_key in pref_lower:
            matched_buckets += 1
            if any(k in text for k in keywords):
                bucket_hits += 1

    if matched_buckets > 0:
        if bucket_hits == 0:
            # User asked for something specific and this place has none
            # of those signals -> push it down hard so the ranker
            # actually re-orders.
            return 0.1
        return min(1.0, bucket_hits / matched_buckets)

    # Pass 2: no known bucket (e.g. user typed "rooftop bar with friends").
    # Fall back to direct token overlap on name+description+categories.
    tokens = [t for t in pref_lower.split() if len(t) > 3]
    if not tokens:
        return 0.5
    matched = sum(1 for t in tokens if t in text)
    if matched == 0:
        return 0.25
    return 0.5 + min(0.4, 0.2 * matched)


def _proximity(distance_km: Optional[float], horizon_km: float = 25.0) -> float:
    """0..1 closeness factor. 0 km -> 1.0, ``horizon_km`` or further -> 0.

    When distance is unknown (no user locality), we return a neutral
    0.6 so the proximity term doesn't accidentally penalise everything
    just because the user didn't supply a locality.
    """
    if distance_km is None:
        return 0.6
    if distance_km <= 0:
        return 1.0
    if distance_km >= horizon_km:
        return 0.0
    return 1.0 - (distance_km / horizon_km)


def _popularity(place: dict[str, Any]) -> float:
    """Normalised 0..1 popularity from Foursquare fields (best-effort)."""
    pop = place.get("popularity")
    if isinstance(pop, (int, float)):
        return max(0.0, min(1.0, float(pop)))
    rating = place.get("rating")
    if isinstance(rating, (int, float)):
        return max(0.0, min(1.0, float(rating) / 10.0))
    return 0.4


def score_place(
    place: dict[str, Any],
    weather: dict[str, Any],
    preference: Optional[str],
) -> float:
    """Score a single place. Reads ``place["_distance_km"]`` if the
    orchestrator has already attached it (recommended) so far-away
    places drop in the ranking.
    """
    wf = _weather_fit(place, weather)
    pm = _prompt_match(place, preference)
    pop = _popularity(place)
    prox = _proximity(place.get("_distance_km"))
    return round(0.20 * wf + 0.45 * pm + 0.15 * pop + 0.20 * prox, 4)


def _primary_category(place: dict[str, Any]) -> str:
    """Lower-cased first category, used as the diversity bucket key."""
    cats = place.get("categories") or []
    if not cats:
        return "uncategorised"
    return str(cats[0]).strip().lower()


def rank_places(
    places: list[dict[str, Any]],
    weather: dict[str, Any],
    preference: Optional[str],
    max_results: int,
    *,
    max_per_category: int = 2,
) -> list[dict[str, Any]]:
    """Return the top ``max_results`` places, score-annotated and
    category-diversified.

    Diversity rule: at most ``max_per_category`` places sharing the
    same primary category (case-insensitive). Without it we'd often
    hand back five identical-looking museums. If we can't fill the
    quota under the cap, we relax it so the user always gets a full
    list.
    """
    scored: list[dict[str, Any]] = []
    for p in places:
        if not p.get("coords", {}).get("lat"):
            continue
        s = score_place(p, weather, preference)
        enriched = dict(p)
        enriched["_score"] = s
        scored.append(enriched)

    scored.sort(key=lambda row: row["_score"], reverse=True)

    # Pass 1: respect the per-category cap.
    picked: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    leftover: list[dict[str, Any]] = []
    for row in scored:
        cat = _primary_category(row)
        if category_counts.get(cat, 0) < max_per_category:
            picked.append(row)
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if len(picked) >= max_results:
                return picked
        else:
            leftover.append(row)

    # Pass 2: top up from the leftovers if diversity left us short.
    for row in leftover:
        if len(picked) >= max_results:
            break
        picked.append(row)

    return picked
