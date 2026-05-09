"""Intent extraction: free-text prompt -> structured TripIntent.

Why this exists (interview talking point)
-----------------------------------------

Without this layer, every request hit Foursquare with the static query
``tourist attractions``. The candidate pool was therefore the same for
"want to eat something" as for "spiritual evening" - the LLM could only
re-decorate the descriptions; the actual places were identical.

Now the pipeline is:

    user prompt
        |
        v
    parse_intent_rule_based  -- fast, deterministic, no LLM cost
        |
        | (no strong match? AND llm available?)
        v
    LLMProvider.extract_intent -- semantic mapping
        |
        v
    TripIntent { category, search_keywords, mood }

We then use ``search_keywords`` as the Foursquare ``query`` and
``category`` as a strong signal in the ranker + cache namespace.

Design choices:
* Rule-based first because 90% of common prompts ("food", "history",
  "shopping") hit known buckets and we don't want to pay LLM latency
  for them.
* LLM-based extraction is only invoked for ambiguous prompts ("want to
  eat", "I'm feeling tired", "fun weekend") where keyword matching is
  not enough.
* The result is dataclass-immutable + JSON-serialisable so we can put
  it in the audit table and surface it in the API response.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from app.clients.llm_client import LLMProvider

logger = logging.getLogger(__name__)


# Canonical category labels. Anything outside this set is normalised to
# "tourist". The list is intentionally small + closed - it doubles as
# the cache namespace and as a stable enum the UI can switch on.
VALID_CATEGORIES: set[str] = {
    "food",
    "spiritual",
    "nature",
    "adventure",
    "history",
    "shopping",
    "family",
    "art",
    "romantic",
    "nightlife",
    "tourist",  # default catch-all
}

# Canonical mood labels. Optional; null for prompts without obvious mood.
VALID_MOODS: set[str] = {
    "peaceful",
    "fun",
    "energetic",
    "romantic",
    "social",
}

# Keyword rules used by ``parse_intent_rule_based``.
# Order matters: the first rule that hits wins. Put more specific rules
# higher up (e.g. "kids" before generic "fun").
_RULES: list[tuple[list[str], str, list[str], Optional[str]]] = [
    # keywords (substring match, lower-cased), category, search_keywords, mood
    (["eat", "food", "hungry", "restaurant", "cafe", "dine", "lunch",
      "dinner", "breakfast", "snack", "cuisine", "tasty", "biryani",
      "street food", "drink", "bar", "pub", "brewery"],
     "food",
     ["restaurants", "cafes", "food"],
     None),
    (["temple", "spiritual", "religious", "pray", "shrine", "mosque",
      "church", "monastery", "ashram", "meditation", "mandir", "masjid",
      "gurudwara"],
     "spiritual",
     ["temples", "religious places", "spiritual"],
     "peaceful"),
    (["history", "historic", "monument", "fort", "palace", "ancient",
      "heritage", "ruins", "archaeological", "museum"],
     "history",
     ["historic monuments", "forts", "museums"],
     None),
    (["shopping", "mall", "shop ", "buy ", "market", "souvenir",
      "bazaar", "boutique"],
     "shopping",
     ["malls", "markets", "shopping"],
     None),
    (["adventure", "trek", "hike", "thrill", "exciting", "rafting",
      "climb", "zip-line", "ziplin", "rappel"],
     "adventure",
     ["adventure", "trekking", "hiking"],
     "energetic"),
    (["kids", "children", "child", "family", "toddler"],
     "family",
     ["family-friendly attractions", "parks", "zoos", "amusement parks"],
     "fun"),
    (["romantic", "date night", "couple", "anniversary", "honeymoon"],
     "romantic",
     ["romantic spots", "viewpoints", "gardens", "fine dining"],
     "romantic"),
    (["art ", "gallery", "exhibition", "artwork", "painting", "sculpture"],
     "art",
     ["art galleries", "art museums"],
     None),
    (["nightlife", "club", "rooftop", "lounge", "live music", "party"],
     "nightlife",
     ["nightlife", "rooftops", "clubs", "lounges"],
     "social"),
    # Mood-led rules: "peaceful", "calm", "fun" without an explicit
    # category map to nature / family-style picks. They come AFTER the
    # category rules so a prompt like "peaceful temple" still hits
    # "spiritual" first.
    (["peaceful", "calm", "quiet", "relax", "serene", "chill", "unwind",
      "tired"],
     "nature",
     ["parks", "gardens", "lakes", "viewpoints"],
     "peaceful"),
    (["nature", "outdoor", "park ", "garden", "lake", "scenic",
      "viewpoint", "hill", "waterfall", "beach", "fresh air"],
     "nature",
     ["parks", "gardens", "lakes", "viewpoints", "nature"],
     None),
    (["fun", "enjoy", "entertainment", "play", "weekend"],
     "family",
     ["amusement parks", "fun things to do"],
     "fun"),
]


@dataclass
class TripIntent:
    """Structured intent extracted from a user prompt.

    Attributes:
        raw_prompt: The original user prompt (or None).
        category:    Canonical label (one of ``VALID_CATEGORIES``).
        search_keywords: Words used to bias upstream place searches
                         (passed to Foursquare's ``query`` parameter).
        mood:        Optional mood label (one of ``VALID_MOODS``).
        source:      "rule" / "llm" / "default" - useful for logging.
    """

    raw_prompt: Optional[str]
    category: str = "tourist"
    search_keywords: list[str] = field(
        default_factory=lambda: ["tourist attractions", "things to do"]
    )
    mood: Optional[str] = None
    source: str = "default"

    @property
    def query_string(self) -> str:
        """Joined search keywords ready for the Foursquare ``query`` param."""
        return " ".join(self.search_keywords)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Rule-based extraction
# ---------------------------------------------------------------------------
_NON_WORD = re.compile(r"[^a-z0-9]+")


def _normalise(prompt: str) -> str:
    """Collapse whitespace + punctuation, lower-case. We pad with spaces
    so substring-matchers like ``"art "`` (note trailing space) don't
    accidentally hit "heart" or "smart".
    """
    return " " + _NON_WORD.sub(" ", prompt.lower()).strip() + " "


def parse_intent_rule_based(prompt: Optional[str]) -> TripIntent:
    """Deterministic, cheap, no-LLM intent parser.

    Returns a TripIntent. Falls through to the default ``tourist``
    category when no rule matches - the caller can then decide whether
    to escalate to the LLM.
    """
    if not prompt or not prompt.strip():
        return TripIntent(raw_prompt=prompt, source="default")

    haystack = _normalise(prompt)
    for keywords, category, search_kw, mood in _RULES:
        for kw in keywords:
            # We pad single tokens with spaces so partial matches don't
            # bleed (e.g. "park " in "Lumbini Park"). Multi-word phrases
            # are matched as-is.
            needle = kw if " " in kw else f" {kw} "
            if needle in haystack:
                logger.debug(
                    "intent rule hit: keyword=%r -> category=%s", kw, category
                )
                return TripIntent(
                    raw_prompt=prompt,
                    category=category,
                    search_keywords=list(search_kw),
                    mood=mood,
                    source="rule",
                )

    return TripIntent(raw_prompt=prompt, source="default")


def normalise_llm_payload(payload: dict[str, Any], prompt: str) -> TripIntent:
    """Validate + normalise the JSON the LLM returned to a clean
    TripIntent. Anything off-schema is coerced to safe defaults - we
    never trust raw LLM output downstream.
    """
    raw_cat = str(payload.get("category") or "").strip().lower()
    category = raw_cat if raw_cat in VALID_CATEGORIES else "tourist"

    raw_kw = payload.get("search_keywords") or payload.get("keywords") or []
    if isinstance(raw_kw, str):
        raw_kw = [raw_kw]
    keywords = [
        str(k).strip().lower() for k in raw_kw if isinstance(k, (str,)) and str(k).strip()
    ]
    if not keywords:
        # Fall back to a sensible default for the category.
        keywords = _category_default_keywords(category)
    keywords = keywords[:6]  # cap query size

    raw_mood = payload.get("mood")
    if isinstance(raw_mood, str):
        m = raw_mood.strip().lower()
        mood = m if m in VALID_MOODS else None
    else:
        mood = None

    return TripIntent(
        raw_prompt=prompt,
        category=category,
        search_keywords=keywords,
        mood=mood,
        source="llm",
    )


def _category_default_keywords(category: str) -> list[str]:
    """Per-category fallback keywords used when the LLM forgot them."""
    return {
        "food": ["restaurants", "cafes", "food"],
        "spiritual": ["temples", "religious places"],
        "nature": ["parks", "gardens", "lakes", "viewpoints"],
        "adventure": ["adventure", "trekking", "hiking"],
        "history": ["historic monuments", "forts", "museums"],
        "shopping": ["malls", "markets", "shopping"],
        "family": ["family-friendly attractions", "amusement parks"],
        "art": ["art galleries", "art museums"],
        "romantic": ["romantic spots", "viewpoints", "fine dining"],
        "nightlife": ["nightlife", "rooftops", "clubs"],
    }.get(category, ["tourist attractions", "things to do"])


# ---------------------------------------------------------------------------
# Async entry-point used by the orchestrator
# ---------------------------------------------------------------------------
async def extract_intent(
    prompt: Optional[str],
    llm: "LLMProvider",
) -> TripIntent:
    """Best-effort intent extraction.

    Strategy:
        1. Try the rule-based parser. If it returns a non-default
           category, take it (cheap, deterministic - no LLM cost).
        2. Otherwise, ask the LLM. Validate + normalise its JSON.
        3. If the LLM call fails or its output is unusable, fall back
           to the rule-based default.

    This gives us LLM-grade semantic understanding for the ambiguous
    cases ("want to eat something", "feeling tired") without paying
    LLM latency for the obvious ones ("food", "history").
    """
    rule_result = parse_intent_rule_based(prompt)

    if rule_result.category != "tourist" or not prompt:
        # Strong rule match (or empty prompt) - skip the LLM.
        return rule_result

    try:
        payload = await llm.extract_intent(prompt)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("LLM extract_intent raised, using rule fallback: %s", exc)
        return rule_result

    if not payload:
        logger.info("LLM extract_intent returned nothing, using rule fallback")
        return rule_result

    try:
        return normalise_llm_payload(payload, prompt)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("LLM intent payload invalid (%s); using rule fallback", exc)
        return rule_result
