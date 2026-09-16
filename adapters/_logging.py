# adapters/_logging.py
"""Centralized logging configuration for recap-comic.

Provides:
  setup_logging(level, log_dir, json_format) - configure root logger
  JobContext - thread-local job ID injection via logging Filter
  sanitize - strip secrets from strings/dicts before logging
"""
from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

_tl = threading.local()

# Sensitive-key detection, shared with webapp/jobs.py so the two redaction
# implementations cannot drift. Key names are matched as WORDS (camelCase and
# separators both split) so "apiKey"/"xkiro_api_key" hit while innocent names
# like "author" do not; the markers catch compounds ("authkey") by substring.
_WORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")

_SENSITIVE_WORDS = frozenset({
    "key", "token", "password", "passwd", "secret", "credential",
    "credentials", "bearer", "auth", "apikey", "apisecret",
    "authorization", "accesstoken", "refreshtoken",
})
_SENSITIVE_MARKERS = ("apikey", "token", "password", "passwd", "secret",
                      "credential", "bearer", "auth")
_INNOCENT_NAMES = frozenset({"author", "authors", "authored", "authorname"})

SENSITIVE_KEYS = frozenset({
    "api_key", "apikey", "token", "authorization", "cookie",
    "password", "secret", "x-api-key", "x-auth-token",
})


def is_sensitive_key(key: Any) -> bool:
    """True if a dict key name looks like it holds a credential.

    Exact word matching handles "apiKey", "xkiro_api_key", "GEMINI_API_KEY";
    the marker substring pass catches compounds like "authkey"; the
    innocents list keeps "author" (blanked by the old "auth" marker).
    """
    name = re.sub(r"[^a-z0-9]", "", str(key).lower())
    if name in _INNOCENT_NAMES:
        return False
    words = [w.lower() for w in _WORD_RE.findall(str(key))]
    if any(w in _SENSITIVE_WORDS for w in words):
        return True
    return any(m in name for m in _SENSITIVE_MARKERS)

# Key-shaped substrings that can appear in provider exception messages or
# URLs ("key=sk-...", "Bearer ...", Google AIza keys, etc.). A bare
# alphanumeric run is deliberately NOT matched (too many false positives
# on ordinary words): only well-known key prefixes and key=value forms.
_KEY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"AIza[A-Za-z0-9_\-]{20,}"),
    re.compile(r"(?i)(?:api[_-]?key|token|authorization)"
               r"(?:[\"'\s:=]{1,3})[A-Za-z0-9_\-\.]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{8,}"),
    re.compile(r"(?i)(?:key|token)=([A-Za-z0-9_\-]{8,})"),
]


def _redact_str(text: str) -> str:
    """Scrub key-shaped substrings (sk-..., AIza..., 'Bearer x', 'key=x')
    from an arbitrary string such as a provider exception message."""
    for pat in _KEY_PATTERNS:
        text = pat.sub(_REDACTED, text)
    return text


_REDACTED = "***REDACTED***"


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
            if is_sensitive_key(k):
                out[k] = _REDACTED
            else:
                out[k] = _redact(v)
        return out
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, str):
        # Dict-value strings are only redacted when their KEY is sensitive;
        # scrubbing every dict value would mangle legitimate content. Only
        # clearly key-shaped strings get pattern-scrubbed.
        return value
    return value


def sanitize(obj: Any) -> Any:
    """Return a copy of obj with secrets redacted for safe logging.

    - dicts: sensitive keys redacted (recursively)
    - lists/tuples: redacted per item
    - strings (including str() of exceptions): key-shaped substrings
      (sk-..., AIza..., 'Bearer x', 'key=x') scrubbed
    - other objects: str() them and scrub
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        return _redact_str(obj)
    if isinstance(obj, (dict, list, tuple)):
        red = _redact(obj)
        return red
    return _redact_str(_safe_str(obj))


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


class _JsonFormatter(logging.Formatter):
    """Emit one VALID JSON object per record.

    A plain format string with "msg":%(message)s interpolates the message
    RAW and UNQUOTED, so any message containing a double quote, backslash
    or newline produced invalid JSON and broke every structured-log
    ingesting these lines.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "job": getattr(record, "job_id", "-"),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _build_formatter(json_format: bool) -> logging.Formatter:
    if json_format:
        return _JsonFormatter()
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
