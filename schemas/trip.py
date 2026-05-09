"""Pydantic request / response models for the trip-suggestion API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TripRequest(BaseModel):
    city: str = Field(..., min_length=2, max_length=128, examples=["Hyderabad"])
    preference: Optional[str] = Field(
        default=None,
        max_length=500,
        examples=["peaceful places with good views"],
    )
    locality: Optional[str] = Field(
        default=None,
        max_length=256,
        examples=["Gachibowli"],
    )
    max_results: Optional[int] = Field(default=None, ge=1, le=10)


class Weather(BaseModel):
    temp_c: Optional[float] = None
    feels_like_c: Optional[float] = None
    condition: Optional[str] = None
    description: Optional[str] = None
    humidity: Optional[int] = None
    wind_kph: Optional[float] = None


class Coords(BaseModel):
    lat: Optional[float] = None
    lng: Optional[float] = None


class Budget(BaseModel):
    currency: str = "INR"
    entry: int = 0
    travel: Optional[int] = None
    total: Optional[int] = None
    note: str = ""


class PlaceSuggestion(BaseModel):
    name: str
    description: str = ""
    categories: list[str] = Field(default_factory=list)
    reasoning: str
    coords: Coords
    distance_km: Optional[float] = None
    estimated_budget: Budget
    score: float = 0.0
    website: Optional[str] = None
    address: Optional[str] = None


class TripIntentMeta(BaseModel):
    """How the user's prompt was parsed.

    Surfaced in the API response so the UI can show the user *what
    we understood* their preference to be (and the interviewer can see
    that the LLM is doing real semantic work, not just re-decorating
    descriptions).
    """

    category: str = "tourist"
    search_keywords: list[str] = Field(default_factory=list)
    mood: Optional[str] = None
    # "rule" | "llm" | "default" - tells the UI whether the LLM was
    # responsible for translating the prompt.
    source: str = "default"


class TripMeta(BaseModel):
    elapsed_ms: int
    cache_hits: list[str] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    llm_provider: str
    # True when the LLM curate stage was used to pick + reason about the
    # final places (single LLM call). False when we fell back to pure
    # rule-based ordering + per-place reasoning. Visible in the UI so
    # the demo viewer can see when AI is actually shaping results.
    llm_curate_used: bool = False
    # Structured intent extracted from req.preference. Driven by the
    # LLM (semantic) when keyword rules can't classify the prompt.
    intent: Optional[TripIntentMeta] = None


class TripResponse(BaseModel):
    city: str
    weather: Weather
    user_location: Optional[Coords] = None
    suggestions: list[PlaceSuggestion]
    meta: TripMeta
