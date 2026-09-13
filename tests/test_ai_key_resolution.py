# tests/test_ai_key_resolution.py
"""API-key resolution chain in adapters.ai_models (offline).

Contract: manual webapp key (webapp_output/settings.json "api_key")
wins over .env / env-var keys; explicit caller key wins over everything.
The chain after the manual key: XKIRO_API_KEY, XKIRO_API_KEYS (pool,
first), GEMINI_API_KEYS (pool, first), GEMINI_API_KEY. OpenAI /
Anthropic backends in strip_analyzer fall back to the manual key when
their env vars are unset. Key values must never be logged.
"""
from __future__ import annotations

import json
import logging

import pytest

from adapters import ai_models as ai


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """No key env vars, no manual settings file influence."""
    for var in ("XKIRO_API_KEY", "XKIRO_API_KEYS",
                "GEMINI_API_KEYS", "GEMINI_API_KEY",
                "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
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
    clean_env.setenv("XKIRO_API_KEY", "env-key")
    assert ai.api_key_from_env("explicit-key") == "explicit-key"


def test_manual_key_wins_over_env(clean_env, fake_settings) -> None:
    fake_settings({"api_key": "from-settings-json"})
    clean_env.setenv("XKIRO_API_KEY", "from-env")
    clean_env.setenv("OPENAI_API_KEY", "from-env-openai")
    assert ai.api_key_from_env() == "from-settings-json"


def test_env_used_when_no_manual_key(clean_env, fake_settings) -> None:
    fake_settings(None)
    clean_env.setenv("XKIRO_API_KEY", "xkiro-env")
    assert ai.api_key_from_env() == "xkiro-env"


def test_xkiro_pool_first_entry_taken(clean_env, fake_settings) -> None:
    fake_settings(None)
    clean_env.setenv("XKIRO_API_KEYS", "k1, k2 ,k3")
    assert ai.api_key_from_env() == "k1"


def test_gemini_pool_and_single_key_in_chain(clean_env, fake_settings) -> None:
    fake_settings(None)
    clean_env.setenv("GEMINI_API_KEYS", "g1,g2")
    assert ai.api_key_from_env() == "g1"
    clean_env.delenv("GEMINI_API_KEYS")
    clean_env.setenv("GEMINI_API_KEY", "g-single")
    assert ai.api_key_from_env() == "g-single"


def test_chain_order_xkiro_before_gemini(clean_env, fake_settings) -> None:
    fake_settings(None)
    clean_env.setenv("XKIRO_API_KEYS", "xk")
    clean_env.setenv("GEMINI_API_KEY", "gm")
    assert ai.api_key_from_env() == "xk"


def test_no_key_returns_none(clean_env, fake_settings) -> None:
    fake_settings(None)
    assert ai.api_key_from_env() is None


def test_malformed_settings_json_is_tolerated(clean_env, fake_settings) -> None:
    (ai.OUTPUT_DIR / "settings.json").write_text("{not json", "utf-8")
    clean_env.setenv("XKIRO_API_KEY", "env-fallback")
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
    assert "XKIRO_API_KEY" in msg and "GEMINI_API_KEY" in msg


def test_openai_backend_falls_back_to_manual_key(
        clean_env, fake_settings) -> None:
    import strip_analyzer as sa
    fake_settings({"api_key": "manual-openai"})
    b = sa.OpenAIVisionBackend(model="gpt-4o-mini")
    assert b._api_key == "manual-openai"


def test_openai_backend_env_beats_manual_key(
        clean_env, fake_settings) -> None:
    import strip_analyzer as sa
    fake_settings({"api_key": "manual-openai"})
    clean_env.setenv("OPENAI_API_KEY", "env-openai")
    b = sa.OpenAIVisionBackend(model="gpt-4o-mini")
    assert b._api_key == "env-openai"


def test_anthropic_backend_falls_back_to_manual_key(
        clean_env, fake_settings) -> None:
    import strip_analyzer as sa
    fake_settings({"api_key": "manual-anthropic"})
    b = sa.AnthropicVisionBackend(model="claude-3-5-sonnet-latest")
    assert b._api_key == "manual-anthropic"


def test_key_never_logged(caplog: pytest.LogCaptureFixture,
                          clean_env, fake_settings) -> None:
    fake_settings({"api_key": "secret-manual-key"})
    with caplog.at_level(logging.DEBUG, logger="adapters.ai_models"):
        ai.api_key_from_env()
    assert "secret-manual-key" not in caplog.text
