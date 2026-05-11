"""Coarse-grained budget estimator.

We explicitly label this as *indicative* in the API response so callers
don't treat the numbers as real pricing.
"""

from __future__ import annotations

from typing import Any, Optional

# Rough INR entry-fee bands per Foursquare price tier (1..4).
# Tier 1 covers 'free or very cheap' tourist spots, tier 4 covers premium
# attractions (e.g. theme parks, exclusive gardens).
_PRICE_TIER_TO_INR = {
    1: 50,
    2: 200,
    3: 500,
    4: 1200,
}

# Indian metro taxi/cab average used for the travel-cost proxy.
_PER_KM_INR = 24


def estimate_budget(
    place: dict[str, Any], distance_km: Optional[float]
) -> dict[str, Any]:
    """Return a dict with ``entry``, ``travel``, ``total`` and a currency tag."""
    tier = place.get("price_tier")
    entry = _PRICE_TIER_TO_INR.get(tier, 0) if isinstance(tier, int) else 0

    if distance_km is None:
        travel: Optional[int] = None
    else:
        travel = int(round(distance_km * _PER_KM_INR))

    total: Optional[int]
    if travel is None:
        total = entry if entry else None
    else:
        total = entry + travel

    return {
        "currency": "INR",
        "entry": entry,
        "travel": travel,
        "total": total,
        "note": "Indicative. Entry derived from price tier; travel ~Rs.24/km cab proxy.",
    }
