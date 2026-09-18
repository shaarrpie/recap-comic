# tests/test_ai_key_resolution.py
"""API-key resolution chain in adapters.ai_models (offline).

Contract: manual webapp key (webapp_output/settings.json "api_key")
wins over .env / env-var keys; explicit caller key wins over everything.
The chain after the manual key: AGNES_API_KEY, AGNES_API_KEYS (pool,
first). Agnes is the sole AI provider. Key values must never be logged.
"""
from __future__ import annotations

import json
import logging

import pytest

from adapters import ai_models as ai


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """No key env vars, no manual settings file influence."""
    for var in ("AGNES_API_KEY", "AGNES_API_KEYS"):
        monkeypatch.delenv(var, raising=False)
    yield monkeypatch


@pytest.fixture()
def fake_settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point the module-level OUTPUT_DIR at a temp dir; returns a writer
    so each test controls the manual-key content."""
    out = tmp_path / "webapp_output"
    out.mkdir()
    monkeypatch.setattr(ai, "OUTPUT_DIR", out)
    f = out / "settings.json"

    def _write(data) -> None:
        if data is None:
            f.unlink(missing_ok=True)
        else:
            f.write_text(json.dumps(data), "utf-8")
    return _write


def test_explicit_param_beats_everything(clean_env, fake_settings) -> None:
    fake_settings({"api_key": "manual"})
    clean_env.setenv("AGNES_API_KEY", "env-key")
    assert ai.api_key_from_env("explicit-key") == "explicit-key"


def test_manual_key_wins_over_env(clean_env, fake_settings) -> None:
    fake_settings({"api_key": "from-settings-json"})
    clean_env.setenv("AGNES_API_KEY", "from-env")
    assert ai.api_key_from_env() == "from-settings-json"


def test_env_used_when_no_manual_key(clean_env, fake_settings) -> None:
    fake_settings(None)
    clean_env.setenv("AGNES_API_KEY", "agnes-env")
    assert ai.api_key_from_env() == "agnes-env"


def test_agnes_pool_first_entry_taken(clean_env, fake_settings) -> None:
    fake_settings(None)
    clean_env.setenv("AGNES_API_KEYS", "k1, k2 ,k3")
    assert ai.api_key_from_env() == "k1"


def test_single_key_beats_pool(clean_env, fake_settings) -> None:
    fake_settings(None)
    clean_env.setenv("AGNES_API_KEY", "single")
    clean_env.setenv("AGNES_API_KEYS", "pool1,pool2")
    assert ai.api_key_from_env() == "single"


def test_no_key_returns_none(clean_env, fake_settings) -> None:
    fake_settings(None)
    assert ai.api_key_from_env() is None


def test_malformed_settings_json_is_tolerated(clean_env, fake_settings) -> None:
    (ai.OUTPUT_DIR / "settings.json").write_text("{not json", "utf-8")
    clean_env.setenv("AGNES_API_KEY", "env-fallback")
    assert ai.api_key_from_env() == "env-fallback"


def test_settings_non_dict_and_blank_key_ignored(clean_env, fake_settings) -> None:
    fake_settings(["not", "a", "dict"])
    assert ai.api_key_from_env() is None
    fake_settings({"api_key": "   "})
    assert ai.api_key_from_env() is None
    fake_settings({"other": "x"})
    assert ai.api_key_from_env() is None


def test_require_api_key_error_mentions_chain(clean_env, fake_settings) -> None:
    fake_settings(None)
    with pytest.raises(RuntimeError) as ei:
        ai.require_api_key()
    msg = str(ei.value)
    assert "AGNES_API_KEY" in msg


def test_agnes_backend_explicit_key_wins(
        clean_env, fake_settings) -> None:
    import strip_analyzer as sa
    fake_settings({"api_key": "manual-agnes"})
    clean_env.setenv("AGNES_API_KEY", "env-agnes")
    b = sa.AgnesVisionBackend(api_key="explicit-agnes",
                              request_fn=lambda *a: ("{}", None))
    assert b._api_key == "explicit-agnes"


def test_agnes_backend_falls_back_to_manual_key(
        clean_env, fake_settings) -> None:
    import strip_analyzer as sa
    fake_settings({"api_key": "manual-agnes"})
    b = sa.AgnesVisionBackend(request_fn=lambda *a: ("{}", None))
    assert b._api_key == "manual-agnes"


def test_agnes_backend_manual_beats_env(
        clean_env, fake_settings) -> None:
    import strip_analyzer as sa
    fake_settings({"api_key": "manual-agnes"})
    clean_env.setenv("AGNES_API_KEY", "env-agnes")
    # NOTE: manual settings key wins over env by contract.
    b = sa.AgnesVisionBackend(request_fn=lambda *a: ("{}", None))
    assert b._api_key == "manual-agnes"


def test_key_never_logged(caplog: pytest.LogCaptureFixture,
                          clean_env, fake_settings) -> None:
    fake_settings({"api_key": "secret-manual-key"})
    with caplog.at_level(logging.DEBUG, logger="adapters.ai_models"):
        ai.api_key_from_env()
    assert "secret-manual-key" not in caplog.text


def test_api_key_pool_explicit_wins_alone(clean_env, fake_settings) -> None:
    fake_settings({"api_key": "manual"})
    clean_env.setenv("AGNES_API_KEY", "env-key")
    clean_env.setenv("AGNES_API_KEYS", "p1,p2")
    assert ai.api_key_pool("explicit-key") == ["explicit-key"]


def test_api_key_pool_order_and_dedupe(clean_env, fake_settings) -> None:
    fake_settings({"api_key": "manual"})
    clean_env.setenv("AGNES_API_KEY", "single")
    clean_env.setenv("AGNES_API_KEYS", "single, p1, manual, p2, p1")
    assert ai.api_key_pool() == ["manual", "single", "p1", "p2"]


def test_api_key_pool_empty(clean_env, fake_settings) -> None:
    fake_settings(None)
    assert ai.api_key_pool() == []


def test_pool_start_index_round_robins() -> None:
    ai._pool_cursor = 0
    assert [ai.pool_start_index(3) for _ in range(4)] == [0, 1, 2, 0]
    assert ai.pool_start_index(1) == 0


def test_is_rate_limit_error() -> None:
    assert ai.is_rate_limit_error(None) is False
    assert ai.is_rate_limit_error(ValueError("429 Too Many Requests")) is True
    assert ai.is_rate_limit_error(
        RuntimeError("rate_limit_exceeded, retry later")) is True
    assert ai.is_rate_limit_error(
        RuntimeError("RESOURCE_EXHAUSTED quota hit")) is True
    assert ai.is_rate_limit_error(ValueError("invalid JSON")) is False
    wrapped = ai.AIFallbackError(
        "op", ValueError("429 boom"), ValueError("other"))
    assert ai.is_rate_limit_error(wrapped) is True
    wrapped_ok = ai.AIFallbackError(
        "op", ValueError("bad json"), ValueError("empty"))
    assert ai.is_rate_limit_error(wrapped_ok) is False
