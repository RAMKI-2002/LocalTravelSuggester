"""Distance helpers.

We only ship the Haversine implementation: it is deterministic, has no
network failure mode, and is more than accurate enough for budgeting and
'is this a reasonable suggestion' decisions over city-scale distances.
"""

from __future__ import annotations

import math

_EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return round(_EARTH_RADIUS_KM * c, 2)
