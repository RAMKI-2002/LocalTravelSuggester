"""In-memory ring buffer for recent log records.

We attach this handler to the root logger alongside the JSON stdout
handler so the dashboard can fetch the last N log lines via
``GET /logs`` without us having to ship logs to a separate aggregator.

Why an in-memory buffer rather than tailing a file?
    * Zero ops surface (no file paths, no rotation, no permissions).
    * Works identically on Windows / Linux / Lambda / containers.
    * Bounded memory: a deque with maxlen never grows past the cap.
    * Trivial to fetch: a single ``list(deque)`` snapshot.

Limitations (and the answers if asked in an interview):
    * Lost on process restart -> for production we'd ship to
      CloudWatch / Loki / ELK. The buffer is a local-dev / demo aid.
    * Only logs produced *by this process* are visible -> no
      multi-replica view. Same answer: real deployments use a
      centralised log sink.
"""

from __future__ import annotations

import logging
from collections import deque
from threading import Lock
from typing import Any


# 500 lines is enough to cover several /suggest-trip requests during a
# demo without bloating memory (each record is ~200 bytes).
DEFAULT_BUFFER_SIZE = 500


class _RingBufferHandler(logging.Handler):
    """A :class:`logging.Handler` that keeps the last N records in
    memory. Reads are thread-safe; writes use the standard logging
    machinery's lock plus our own deque lock.
    """

    def __init__(self, capacity: int = DEFAULT_BUFFER_SIZE) -> None:
        super().__init__()
        self._buf: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "ts": self.formatter.formatTime(record, "%Y-%m-%dT%H:%M:%S")
                if self.formatter
                else "",
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
                "rid": getattr(record, "rid", "-"),
            }
            if record.exc_info:
                entry["exc"] = logging.Formatter().formatException(record.exc_info)
        except Exception:  # pragma: no cover - never let logging crash the app
            return
        with self._lock:
            self._buf.append(entry)

    def snapshot(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return a list copy of the buffer, optionally trimmed to the
        last ``limit`` items.
        """
        with self._lock:
            data = list(self._buf)
        if limit is not None and limit > 0:
            data = data[-limit:]
        return data


# Module-level singleton so other modules can import + use it without
# threading state through call sites.
_handler: _RingBufferHandler | None = None


def get_buffer_handler() -> _RingBufferHandler:
    """Return the process-wide ring-buffer handler, creating on first use."""
    global _handler
    if _handler is None:
        _handler = _RingBufferHandler()
    return _handler


def get_recent_logs(limit: int = 200) -> list[dict[str, Any]]:
    """Convenience accessor used by the ``/logs`` route."""
    return get_buffer_handler().snapshot(limit=limit)
