"""LLM client abstraction.

Why an abstraction?
- We can run the whole app offline (``LLM_MOCK=true``) for local dev and CI.
- Swapping Nova Lite for Claude or Titan becomes a one-line change because
  the Bedrock Converse API format is model-agnostic.

How the LLM is used
-------------------

We use the LLM in two distinct ways, intentionally:

1. ``curate_places`` - given a *shortlist* (top 2x) produced by the cheap
   rule-based ranker, the LLM picks the best ``max_results`` for the
   user's preference and writes a one-line reason for each. ONE call
   per request, returns structured JSON. This is what surfaces the
   right places for the user's prompt - "peaceful" vs "adventure" vs
   "food" actually return different picks.

2. ``generate_place_reasoning`` - per-place reasoning, used only when
   curate is unavailable / mocked / the LLM rejects all picks. Falls
   back gracefully so we always show something.

We never let the LLM *invent* places. It can only choose from the
candidate names we passed in - any name it returns that we don't
recognise is dropped, and if it returns nothing usable we fall back
to rule-based ordering. This keeps factuality + cost bounded.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a helpful local trip-suggestion assistant. Given current weather, "
    "the user's preference, and a specific place, write ONE or TWO concise "
    "sentences (max 45 words) explaining why this place is a good fit right "
    "now. Mention the weather context naturally. Do not invent facts about "
    "the place beyond what is given. Output plain text, no markdown."
)

_CURATE_SYSTEM_PROMPT = (
    "You are a local trip-suggestion assistant. You will receive: the current "
    "weather, the user's preference, optional locality, and a numbered list of "
    "candidate places. Your job: pick the BEST matches for the user's "
    "preference (and weather), in priority order. \n"
    "Hard rules:\n"
    "  * Only use names from the candidate list - never invent places.\n"
    "  * Prefer places whose name/categories actually match the preference.\n"
    "  * Prefer nearer places when distance is provided.\n"
    "  * Output STRICT JSON only, no markdown, no prose.\n"
    "JSON schema: {\"picks\":[{\"name\":\"<exact-name>\",\"reason\":\"<one "
    "sentence, <=35 words, mention weather + why this fits the prompt>\"}]}"
)

_INTENT_SYSTEM_PROMPT = (
    "You translate a user's free-text trip preference into structured intent.\n"
    "\n"
    "Output STRICT JSON ONLY (no markdown fences, no prose):\n"
    '{"category": "<one of: food, spiritual, nature, adventure, history, '
    'shopping, family, art, romantic, nightlife, tourist>",\n'
    ' "search_keywords": ["2-5 short keywords used to bias a places-API '
    'search query"],\n'
    ' "mood": "<one of: peaceful, fun, energetic, romantic, social, or null>"}\n'
    "\n"
    "Rules:\n"
    "  * \"want to eat something\" -> category=\"food\", "
    "keywords=[\"restaurants\",\"cafes\",\"food\"].\n"
    "  * \"feeling peaceful\" -> category=\"nature\", mood=\"peaceful\", "
    "keywords=[\"parks\",\"gardens\",\"viewpoints\"].\n"
    "  * \"fun weekend with kids\" -> category=\"family\", mood=\"fun\", "
    "keywords=[\"amusement parks\",\"zoos\",\"family attractions\"].\n"
    "  * unclear/empty -> category=\"tourist\", "
    "keywords=[\"tourist attractions\",\"things to do\"].\n"
    "  * Use ONLY values from the enums above for category and mood.\n"
    "  * keywords must be lowercase, 2-4 words each."
)


def _build_user_prompt(
    weather: dict[str, Any], preference: Optional[str], place: dict[str, Any]
) -> str:
    return (
        f"Weather: {weather.get('condition')} at {weather.get('temp_c')}C "
        f"({weather.get('description') or 'n/a'}).\n"
        f"User preference: {preference or 'No specific preference'}.\n"
        f"Place: {place.get('name')} "
        f"(categories: {', '.join(place.get('categories') or []) or 'n/a'}).\n"
        f"Place description: {place.get('description') or 'n/a'}.\n"
        "Write the recommendation sentence now."
    )


def _build_curate_prompt(
    weather: dict[str, Any],
    preference: Optional[str],
    locality: Optional[str],
    candidates: list[dict[str, Any]],
    max_results: int,
) -> str:
    lines = [
        f"Weather: {weather.get('condition')} at {weather.get('temp_c')}C "
        f"({weather.get('description') or 'n/a'}).",
        f"User preference: {preference or 'No specific preference'}.",
        f"User locality: {locality or 'not provided'}.",
        f"Pick at most {max_results} matches.",
        "",
        "Candidates:",
    ]
    for idx, c in enumerate(candidates, 1):
        cats = ", ".join(c.get("categories") or []) or "n/a"
        dist = c.get("_distance_km")
        dist_str = f"{dist:.1f}km" if isinstance(dist, (int, float)) else "unknown"
        desc = c.get("description") or ""
        if desc:
            desc = " - " + desc[:120]
        lines.append(
            f"  {idx}. {c.get('name')} | cats: {cats} | distance: {dist_str}{desc}"
        )
    lines.append("")
    lines.append(
        "Return JSON in the schema given. Use exact names. Order by best fit first."
    )
    return "\n".join(lines)


def _try_parse_json_object(raw: str) -> Optional[dict[str, Any]]:
    """Best-effort JSON-object extractor.

    LLMs sometimes wrap their output in ``json`` fences, prepend an
    apology, or trail a sentence after the JSON. We try the cheap path
    first, then fenced-block extraction, then a greedy brace match.
    """
    raw = raw.strip()
    if not raw:
        return None

    attempts: list[str] = [raw]

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        attempts.append(fenced.group(1))

    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        attempts.append(brace_match.group(0))

    for attempt in attempts:
        try:
            obj = json.loads(attempt)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _try_parse_curate_json(raw: str) -> Optional[list[dict[str, str]]]:
    """Curate-specific wrapper around :func:`_try_parse_json_object`."""
    obj = _try_parse_json_object(raw)
    if not obj:
        return None
    picks = obj.get("picks")
    if not isinstance(picks, list):
        return None
    cleaned: list[dict[str, str]] = []
    for item in picks:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        reason = item.get("reason") or ""
        if isinstance(name, str) and name.strip():
            cleaned.append({"name": name.strip(), "reason": str(reason).strip()})
    return cleaned


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------
class LLMProvider(ABC):
    @abstractmethod
    async def generate_place_reasoning(
        self,
        weather: dict[str, Any],
        preference: Optional[str],
        place: dict[str, Any],
    ) -> str:
        """Return a short natural-language reason for recommending ``place``."""

    async def curate_places(
        self,
        weather: dict[str, Any],
        preference: Optional[str],
        locality: Optional[str],
        candidates: list[dict[str, Any]],
        max_results: int,
    ) -> Optional[list[dict[str, Any]]]:
        """Return up to ``max_results`` candidates re-ordered by the LLM,
        each with a ``_reasoning`` string attached. Names are validated
        against the input list - hallucinated names are dropped.

        Returns ``None`` if the LLM call fails entirely so the caller
        can fall back to rule-based ordering. Default implementation
        returns ``None`` (used by providers that don't support curate).
        """
        return None

    async def extract_intent(self, prompt: str) -> Optional[dict[str, Any]]:
        """Parse a free-text trip preference into structured intent.

        Returns ``{"category": str, "search_keywords": list[str], "mood":
        str | None}`` or ``None`` on failure. The orchestrator then
        validates / normalises the dict via
        :func:`app.services.intent_parser.normalise_llm_payload`.

        Default implementation returns ``None`` (rule-based fallback
        kicks in upstream).
        """
        return None


# ---------------------------------------------------------------------------
# Bedrock provider (Amazon Nova Lite via Converse API)
# ---------------------------------------------------------------------------
class BedrockLLMProvider(LLMProvider):
    def __init__(self) -> None:
        import boto3  # local import to keep startup fast when mocked

        settings = get_settings()
        self.model_id = settings.bedrock_model_id
        client_kwargs: dict[str, Any] = {"region_name": settings.aws_region}
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = settings.aws_access_key_id
            client_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        self.client = boto3.client("bedrock-runtime", **client_kwargs)

    def _invoke_sync(
        self,
        user_prompt: str,
        *,
        system_prompt: str = _SYSTEM_PROMPT,
        max_tokens: int = 180,
        temperature: float = 0.4,
    ) -> str:
        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system_prompt}],
            messages=[
                {"role": "user", "content": [{"text": user_prompt}]},
            ],
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": 0.9,
            },
        )
        content = response["output"]["message"]["content"]
        if not content:
            raise RuntimeError("Bedrock returned empty content")
        return content[0].get("text", "").strip()

    async def generate_place_reasoning(
        self,
        weather: dict[str, Any],
        preference: Optional[str],
        place: dict[str, Any],
    ) -> str:
        user_prompt = _build_user_prompt(weather, preference, place)
        # boto3 is synchronous - push to a thread so we don't block the loop.
        return await asyncio.to_thread(self._invoke_sync, user_prompt)

    async def curate_places(
        self,
        weather: dict[str, Any],
        preference: Optional[str],
        locality: Optional[str],
        candidates: list[dict[str, Any]],
        max_results: int,
    ) -> Optional[list[dict[str, Any]]]:
        if not candidates:
            return []
        user_prompt = _build_curate_prompt(
            weather, preference, locality, candidates, max_results
        )
        try:
            raw = await asyncio.to_thread(
                self._invoke_sync,
                user_prompt,
                system_prompt=_CURATE_SYSTEM_PROMPT,
                max_tokens=600,
                temperature=0.3,
            )
        except Exception as exc:
            logger.warning("LLM curate call failed: %s", exc)
            return None

        picks = _try_parse_curate_json(raw)
        if not picks:
            logger.warning(
                "LLM curate returned non-JSON output (first 200 chars): %r",
                raw[:200],
            )
            return None

        # Validate names against candidates so the LLM cannot fabricate.
        by_name = {(c.get("name") or "").strip().lower(): c for c in candidates}
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for pick in picks:
            key = pick["name"].strip().lower()
            if key in seen:
                continue
            place = by_name.get(key)
            if place is None:
                logger.debug("dropping hallucinated curate pick: %r", pick["name"])
                continue
            seen.add(key)
            enriched = dict(place)
            if pick.get("reason"):
                enriched["_reasoning"] = pick["reason"]
            out.append(enriched)
            if len(out) >= max_results:
                break

        if not out:
            return None
        logger.info(
            "LLM curate accepted %d/%d picks (max=%d)",
            len(out),
            len(picks),
            max_results,
        )
        return out

    async def extract_intent(self, prompt: str) -> Optional[dict[str, Any]]:
        if not prompt or not prompt.strip():
            return None
        user_prompt = f'User prompt: "{prompt.strip()}"'
        try:
            raw = await asyncio.to_thread(
                self._invoke_sync,
                user_prompt,
                system_prompt=_INTENT_SYSTEM_PROMPT,
                max_tokens=200,
                temperature=0.1,
            )
        except Exception as exc:
            logger.warning("LLM extract_intent call failed: %s", exc)
            return None

        obj = _try_parse_json_object(raw)
        if not obj:
            logger.warning(
                "LLM extract_intent returned non-JSON output (first 200 chars): %r",
                raw[:200],
            )
            return None
        return obj


# ---------------------------------------------------------------------------
# Mock provider - deterministic, offline
# ---------------------------------------------------------------------------
class MockLLMProvider(LLMProvider):
    """Rule-based stand-in so the full flow is testable without AWS creds."""

    async def generate_place_reasoning(
        self,
        weather: dict[str, Any],
        preference: Optional[str],
        place: dict[str, Any],
    ) -> str:
        cond = (weather.get("condition") or "").lower()
        temp = weather.get("temp_c")
        name = place.get("name", "This place")
        cats = ", ".join(place.get("categories") or []) or "local attraction"

        if "rain" in cond or "storm" in cond:
            weather_phrase = f"it's {cond} ({temp}C)"
            suggestion = "an indoor-friendly pick that shields you from the weather"
        elif temp is not None and isinstance(temp, (int, float)) and temp >= 35:
            weather_phrase = f"it's a hot {temp}C"
            suggestion = "best visited in the early morning or late evening"
        elif "clear" in cond or "sun" in cond:
            weather_phrase = f"the {cond} {temp}C weather is ideal"
            suggestion = "great for an outdoor experience"
        else:
            weather_phrase = f"the current {cond} weather at {temp}C"
            suggestion = "a solid pick right now"

        pref_phrase = f" for your '{preference}' preference" if preference else ""
        return (
            f"{name} ({cats}) is {suggestion}{pref_phrase} - {weather_phrase}."
        )

    async def curate_places(
        self,
        weather: dict[str, Any],
        preference: Optional[str],
        locality: Optional[str],
        candidates: list[dict[str, Any]],
        max_results: int,
    ) -> Optional[list[dict[str, Any]]]:
        # The mock can't actually reason - we just take the rule-based
        # order as-is (the orchestrator already pre-ranked) and attach
        # a deterministic reason. This keeps the curate->fall-back path
        # exercised in tests without touching the network.
        if not candidates:
            return []
        out: list[dict[str, Any]] = []
        for c in candidates[:max_results]:
            enriched = dict(c)
            enriched["_reasoning"] = await self.generate_place_reasoning(
                weather, preference, c
            )
            out.append(enriched)
        return out

    async def extract_intent(self, prompt: str) -> Optional[dict[str, Any]]:
        # The mock provider has no model - we delegate to the rule-based
        # parser so the LLM-shaped path is exercised end-to-end in tests
        # without touching the network. We import lazily to avoid a
        # circular import (intent_parser imports LLMProvider for typing).
        from app.services.intent_parser import parse_intent_rule_based

        intent = parse_intent_rule_based(prompt)
        return {
            "category": intent.category,
            "search_keywords": intent.search_keywords,
            "mood": intent.mood,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_provider: Optional[LLMProvider] = None


def get_llm_provider() -> LLMProvider:
    """Return the configured provider (cached for the life of the process)."""
    global _provider
    if _provider is not None:
        return _provider

    settings = get_settings()
    if settings.llm_mock:
        logger.info("LLM_MOCK=true -> using MockLLMProvider")
        _provider = MockLLMProvider()
        return _provider

    try:
        _provider = BedrockLLMProvider()
        logger.info("Using BedrockLLMProvider with model %s", settings.bedrock_model_id)
    except Exception as exc:
        logger.warning(
            "Falling back to MockLLMProvider (Bedrock init failed: %s)", exc
        )
        _provider = MockLLMProvider()
    return _provider


def reset_llm_provider() -> None:
    """Primarily for tests - clear the cached provider."""
    global _provider
    _provider = None
