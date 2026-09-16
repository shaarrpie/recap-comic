# tests/test_logging_redaction.py
"""Centralized logging + secret redaction (offline).

Covers:
  * the JSON formatter must emit VALID JSON even when a message contains a
    double quote, backslash or newline (the old format string interpolated
    the message raw and unquoted);
  * sensitive-key detection must catch realistic names (apiKey,
    xkiro_api_key, GEMINI_API_KEY) without blanking innocent ones like
    "author";
  * the webapp's persisted-snapshot redaction shares that one detector.
"""
from __future__ import annotations

import logging

from adapters._logging import (
    _build_formatter,
    is_sensitive_key,
    sanitize,
)


def _format(record_msg: str, *, json_format: bool = True) -> str:
    fmt = _build_formatter(json_format)
    rec = logging.LogRecord(
        name="test.logger", level=logging.INFO, pathname=__file__,
        lineno=1, msg=record_msg, args=(), exc_info=None)
    rec.job_id = "job-1"
    return fmt.format(rec)


def test_json_formatter_valid_on_quotes_and_newlines() -> None:
    import json

    for nasty in [
        'panel said "hello"',
        "back\\slash and \"quote\"",
        "line one\nline two",
        'emoji + "quote" \n and \\ backslash',
    ]:
        out = _format(nasty)
        parsed = json.loads(out)          # must not raise
        assert parsed["msg"] == nasty, f"message mangled: {nasty!r} -> {out}"
        assert parsed["level"] == "INFO"
        assert parsed["job"] == "job-1"
        assert parsed["logger"] == "test.logger"


def test_json_formatter_escapes_exception_tracebacks() -> None:
    import json

    try:
        raise ValueError("boom \"with quotes\"")
    except ValueError:
        import sys
        rec = logging.LogRecord(
            name="t", level=logging.ERROR, pathname=__file__, lineno=1,
            msg="failed", args=(), exc_info=sys.exc_info())
        rec.job_id = "j"
        out = _build_formatter(True).format(rec)
        parsed = json.loads(out)
        assert "ValueError: boom" in parsed["exc_info"]


def test_is_sensitive_key_catches_realistic_names() -> None:
    for k in ["api_key", "apiKey", "xkiro_api_key", "GEMINI_API_KEY",
              "gemini_api_key", "token", "Authorization", "x-auth-token",
              "password", "secret", "bearer", "credentials", "authkey",
              "access_token"]:
        assert is_sensitive_key(k), f"{k!r} should be sensitive"


def test_is_sensitive_key_spared_innocent_names() -> None:
    # The old substring marker "auth" blanked "author" in persisted snapshots.
    for k in ["author", "authors", "chapter_title", "panel_count",
              "keyword", "narrator", "strips_remaining"]:
        assert not is_sensitive_key(k), f"{k!r} should NOT be sensitive"


def test_sanitize_redacts_nested_dicts() -> None:
    obj = {
        "config": {"apiKey": "sk-real-key", "author": "Jane Doe"},
        "nested": [{"password": "hunter2", "keep": "me"}],
    }
    out = sanitize(obj)
    assert out["config"]["apiKey"] == "***REDACTED***"
    assert out["config"]["author"] == "Jane Doe"   # NOT blanked
    assert "hunter2" not in str(out["nested"])
    assert out["nested"][0]["keep"] == "me"
    # By design, dict STRING VALUES are only scrubbed when their KEY is
    # sensitive — blanket pattern-scrubbing every value would mangle
    # legitimate narration/panel text. Key-shaped substrings in a bare
    # string (e.g. an exception message) ARE scrubbed: see below.
    assert out["config"]["author"] == "Jane Doe"


def test_sanitize_scrubs_key_shaped_strings() -> None:
    assert "sk-abcd1234efgh" not in sanitize("failed with sk-abcd1234efgh")
    assert "AIzaSy" + "x" * 25 not in sanitize("got AIzaSy" + "x" * 25)
    assert "Bearer mF_abc12345" not in sanitize("Authorization: Bearer mF_abc12345")
    # ordinary text is untouched
    assert sanitize("panel 3 is 1200px tall") == "panel 3 is 1200px tall"


def test_webapp_snapshot_redaction_uses_shared_detector() -> None:
    """webapp/jobs._redact_config must catch the same keys (no drift)."""
    from webapp.jobs import JobStore

    cfg = {"apiKey": "sk-real", "author": "Jane", "xkiro_api_key": "k-1",
           "chapter": 47}
    out = JobStore._redact_config(cfg)
    assert out["apiKey"] == ""
    assert out["xkiro_api_key"] == ""
    assert out["author"] == "Jane"      # the over-redaction bug
    assert out["chapter"] == 47
