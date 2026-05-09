"""Logging configuration + request-id middleware.

Structured JSON logs are easier to grep in CloudWatch / ELK. We attach a
short request-id per inbound HTTP request so every log line from one call
can be correlated.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.utils.log_buffer import get_buffer_handler

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


class _RequestIdFilter(logging.Filter):
    """Attach the current request-id to every record so the in-memory
    buffer can include it without a custom formatter.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.rid = _request_id.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "rid": getattr(record, "rid", _request_id.get()),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Install a single JSON stdout handler PLUS an in-memory ring-buffer
    handler on the root logger (idempotent).

    The stdout handler is what the operator sees in the terminal /
    CloudWatch. The ring-buffer handler is what the dashboard's "Logs"
    panel reads via ``GET /logs``.
    """
    root = logging.getLogger()
    if getattr(configure_logging, "_installed", False):
        root.setLevel(level.upper())
        return

    formatter = _JsonFormatter()
    request_id_filter = _RequestIdFilter()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(request_id_filter)

    buffer_handler = get_buffer_handler()
    buffer_handler.setFormatter(formatter)
    buffer_handler.addFilter(request_id_filter)

    root.handlers = [stdout_handler, buffer_handler]
    root.setLevel(level.upper())
    logging.getLogger("uvicorn.access").setLevel("WARNING")
    configure_logging._installed = True  # type: ignore[attr-defined]


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Stamp every request with a short UUID + log latency."""

    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = _request_id.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logging.getLogger("http").info(
                "%s %s -> %d (%dms)",
                request.method,
                request.url.path,
                getattr(locals().get("response", None), "status_code", 0),
                elapsed_ms,
            )
            _request_id.reset(token)
        response.headers["x-request-id"] = rid
        return response
