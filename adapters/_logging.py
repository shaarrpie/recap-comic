# adapters/_logging.py
"""Centralized logging configuration for recap-comic.

Provides:
  setup_logging(level, log_dir, json_format) - configure root logger
  JobContext - thread-local job ID injection via logging Filter
  sanitize - strip secrets from strings/dicts before logging
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

_tl = threading.local()

SENSITIVE_KEYS = frozenset({
    "api_key", "apikey", "token", "authorization", "cookie",
    "password", "secret", "x-api-key", "x-auth-token",
})


def _safe_str(value: Any) -> str:
    try:
        text = str(value)
    except Exception:
        return "<unprintable>"
    if len(text) > 500:
        return text[:500] + "...<truncated>"
    return text


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if k.lower() in SENSITIVE_KEYS:
                out[k] = "***REDACTED***"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def sanitize(obj: Any) -> Any:
    """Return a copy of obj with sensitive keys redacted for safe logging."""
    return _redact(obj)


class _JobFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        job_id = getattr(_tl, "job_id", None)
        record.job_id = job_id or "-"
        return True


class JobContext:
    """Thread-local job ID context for logging.

    Usage::

        with JobContext("job123"):
            log.info("something happened")
    """

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    def __enter__(self) -> JobContext:
        _tl.job_id = self.job_id
        return self

    def __exit__(self, *_: Any) -> None:
        _tl.job_id = None


def _build_formatter(json_format: bool) -> logging.Formatter:
    if json_format:
        return logging.Formatter(
            '{"ts":"%(asctime)s","level":"%(levelname)s","job":"%(job_id)s",'
            '"logger":"%(name)s","msg":%(message)s}'
        )
    return logging.Formatter(
        "%(asctime)s %(levelname)-7s [job=%(job_id)s] %(name)s: %(message)s"
    )


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path | None = None,
    json_format: bool = False,
) -> None:
    """Configure root logging once at application startup."""
    root = logging.getLogger()
    if root.handlers:
        for h in root.handlers:
            if not any(isinstance(f, _JobFilter) for f in h.filters):
                h.addFilter(_JobFilter())
        return

    log_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(log_level)

    fmt = _build_formatter(json_format)
    f = _JobFilter()
    root.addFilter(f)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(log_level)
    sh.setFormatter(fmt)
    sh.addFilter(f)
    root.addHandler(sh)

    if log_dir:
        p = Path(log_dir)
        p.mkdir(parents=True, exist_ok=True)
        try:
            from logging.handlers import RotatingFileHandler
            fh = RotatingFileHandler(
                p / "recap-comic.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            fh.setLevel(log_level)
            fh.setFormatter(fmt)
            fh.addFilter(f)
            root.addHandler(fh)
        except Exception as exc:
            root.warning("could not open log file %s: %s", p, exc)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_timing(logger: logging.Logger, op: str, start: float) -> None:
    elapsed = round(time.time() - start, 3)
    logger.debug("%s completed in %.3fs", op, elapsed)
