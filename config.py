"""Centralised application settings loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed configuration. Values are sourced from the environment
    (and a local ``.env`` file during development).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openweather_api_key: Optional[str] = Field(default=None)
    foursquare_api_key: Optional[str] = Field(default=None)
    # Hard kill-switch for the Foursquare path. Set to ``false`` in
    # ``.env`` when your account has run out of API credits so we don't
    # waste 2-3 seconds per request hitting a provider that always
    # returns 429. Even when set to ``true``, the client will
    # auto-disable itself for the rest of the process lifetime once it
    # sees a "no credits" response (see :class:`PlacesClient`).
    foursquare_enabled: bool = Field(default=True)
    nominatim_user_agent: str = Field(
        default="local-trip-suggester/1.0 (contact@example.com)"
    )

    aws_region: str = Field(default="us-east-1")
    aws_access_key_id: Optional[str] = Field(default=None)
    aws_secret_access_key: Optional[str] = Field(default=None)
    bedrock_model_id: str = Field(default="amazon.nova-lite-v1:0")
    llm_mock: bool = Field(default=False)

    database_url: str = Field(
        default="sqlite:///./test.db",
        description="SQLAlchemy database URL. Point at RDS PostgreSQL in prod.",
    )

    http_timeout_seconds: float = Field(default=10.0)
    place_cache_ttl_hours: int = Field(default=24)
    weather_cache_ttl_minutes: int = Field(default=30)
    default_max_results: int = Field(default=5)

    log_level: str = Field(default="INFO")

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgres")

    @property
    def bedrock_configured(self) -> bool:
        """Bedrock is usable when creds are available OR when running on an
        EC2/ECS/Lambda role (boto3 discovers those automatically). We only
        explicitly force mock mode when ``llm_mock`` is set.
        """
        return not self.llm_mock


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor so we build ``Settings`` exactly once per process."""
    return Settings()
