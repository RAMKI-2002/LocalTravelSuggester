"""Shared HTTP plumbing for every outbound API client.

- Single ``httpx.AsyncClient`` per-process (connection pooling, HTTP/2 off
  for simplicity, sane timeouts).
- ``request_json`` handles retries with exponential backoff via ``tenacity``.
- Upstream failures raise typed exceptions so callers can degrade gracefully.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error hierarchy - callers catch ``UpstreamError`` for any failure.
# ---------------------------------------------------------------------------
class UpstreamError(Exception):
    """Generic failure from an external HTTP dependency."""


class RateLimitError(UpstreamError):
    """Upstream responded with 429."""


class NotFoundError(UpstreamError):
    """Upstream responded with 404."""


# ---------------------------------------------------------------------------
# Shared client
# ---------------------------------------------------------------------------
_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    """Return the process-wide async HTTP client, constructing on first use."""
    global _client
    if _client is None or _client.is_closed:
        settings = get_settings()
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.http_timeout_seconds),
            headers={"Accept": "application/json"},
            follow_redirects=True,
        )
    return _client


async def close_http_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


# ---------------------------------------------------------------------------
# Request helper
# ---------------------------------------------------------------------------
# We deliberately do NOT retry on 429: quota-based rate limits do not recover
# in the sub-second window our exponential backoff would give, and retrying
# just burns more quota. Callers can fall back to cache on RateLimitError.
_RETRYABLE = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)


async def request_json(
    method: str,
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    json_body: Optional[dict[str, Any]] = None,
    retries: int = 2,
) -> Any:
    """Perform an HTTP request and return parsed JSON. Raises ``UpstreamError``
    on unrecoverable failure. Retries transient failures with backoff.
    """
    client = get_http_client()

    async def _do_request() -> httpx.Response:
        resp = await client.request(
            method=method,
            url=url,
            params=params,
            headers=headers,
            json=json_body,
        )
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after") or "unknown"
            raise RateLimitError(
                f"429 from {url} (retry-after={retry_after}): {resp.text[:300]}"
            )
        if resp.status_code == 404:
            raise NotFoundError(f"404 from {url}")
        if resp.status_code >= 500:
            raise UpstreamError(f"{resp.status_code} from {url}: {resp.text[:200]}")
        if resp.status_code >= 400:
            # 4xx other than 404/429 is usually a client bug - don't retry.
            raise UpstreamError(
                f"{resp.status_code} from {url}: {resp.text[:200]}"
            )
        return resp

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(retries + 1),
            wait=wait_exponential(multiplier=0.4, min=0.4, max=3.0),
            retry=retry_if_exception_type(_RETRYABLE),
            reraise=True,
        ):
            with attempt:
                response = await _do_request()
                break
    except RetryError as exc:  # pragma: no cover - defensive
        raise UpstreamError(str(exc)) from exc

    try:
        return response.json()
    except ValueError as exc:
        raise UpstreamError(f"Non-JSON response from {url}") from exc
