# tests/test_ai_fallback.py
"""Qwen3.5-397B-A17B primary + Mistral Medium 3.5 fallback (offline, mocked).

Covers: primary success (no fallback call), primary failure/timeout/
invalid-JSON -> fallback, fallback success metadata, both-fail error,
image+prompt parity, cache reuse, deterministic cutter independence.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

import guided_pipeline as gp
import strip_analyzer as sa
from adapters import ai_models as ai


def _img(w: int = 64, h: int = 64) -> Image.Image:
    return Image.new("RGB", (w, h), (120, 130, 140))


def _panel_payload(narration: str = "A hero stands.") -> str:
    return json.dumps({
        "panels": [{"panel_index": 1, "y_start": 0, "y_end": 1000,
                    "narration": narration, "dialogue": "",
                    "panel_type": "single", "confidence": 0.9,
                    "bubble_boxes": []}],
        "characters": [],
    })


def test_wrapper_primary_success_no_fallback_call() -> None:
    calls: list[str] = []
    out = ai.call_ai_with_fallback(
        "op", lambda: (calls.append("p"), "ok")[1],
        lambda: (calls.append("f"), "bad")[1])
    assert out.result == "ok"
    assert out.model_used == ai.PRIMARY_MODEL
    assert out.fallback_used is False
    assert calls == ["p"]
    assert out.as_dict() == {"result": "ok", "model_used": ai.PRIMARY_MODEL,
                             "fallback_used": False}


def test_wrapper_primary_fail_fallback_success() -> None:
    def _boom() -> str:
        raise ConnectionError("connection reset by peer")
    out = ai.call_ai_with_fallback("op", _boom, lambda: "fallback-ok")
    assert out.result == "fallback-ok"
    assert out.model_used == ai.FALLBACK_MODEL
    assert out.fallback_used is True
    assert isinstance(out.primary_error, ConnectionError)


def test_wrapper_timeout_triggers_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(ai.time, "sleep", lambda s: sleeps.append(s))

    def _timeout() -> str:
        raise TimeoutError("request timed out")
    out = ai.call_ai_with_fallback("op", _timeout, lambda: "recovered")
    assert out.fallback_used is True
    assert out.result == "recovered"
    # one short retry for the transient timeout before switching
    assert sleeps == [1]


def test_wrapper_invalid_json_triggers_fallback() -> None:
    def _bad_json() -> str:
        json.loads("not json{{{")
        raise AssertionError("unreachable")
    out = ai.call_ai_with_fallback("op", _bad_json, lambda: "good")
    assert out.result == "good" and out.fallback_used is True


def test_wrapper_both_fail_raises_with_both_causes() -> None:
    with pytest.raises(ai.AIFallbackError) as ei:
        ai.call_ai_with_fallback(
            "narrate", lambda: (_ for _ in ()).throw(ValueError("qwen 500")),
            lambda: (_ for _ in ()).throw(ValueError("mistral 503")))
    assert "narrate" in str(ei.value)
    assert isinstance(ei.value.primary_error, ValueError)
    assert isinstance(ei.value.fallback_error, ValueError)
    assert "qwen 500" in str(ei.value) and "mistral 503" in str(ei.value)


def test_resolve_model_id_aliases() -> None:
    assert ai.resolve_model_id("Qwen3.5-397B-A17B") == ai.PRIMARY_MODEL
    assert ai.resolve_model_id("Mistral Medium 3.5") == ai.FALLBACK_MODEL
    assert ai.resolve_model_id("qwen") == ai.PRIMARY_MODEL
    assert ai.resolve_model_id("mistral") == ai.FALLBACK_MODEL
    assert ai.resolve_model_id(ai.PRIMARY_MODEL) == ai.PRIMARY_MODEL


def test_xkiro_backend_primary_success_mistral_not_called() -> None:
    seen: list[str] = []

    def _fake(model_id: str, prompt: str, b64: str, image: Image.Image) -> str:
        seen.append(model_id)
        assert b64  # actual image bytes must be passed
        assert "panels" in prompt  # same task instructions
        return _panel_payload()
    b = sa.XkiroVisionBackend(primary_model=ai.PRIMARY_MODEL,
                              fallback_model=ai.FALLBACK_MODEL,
                              api_key="test", request_fn=_fake)
    entries, _chars = b.analyze_chunk(_img())
    assert len(entries) == 1 and entries[0].narration == "A hero stands."
    assert seen == [ai.PRIMARY_MODEL]
    assert b.last_model_used == ai.PRIMARY_MODEL
    assert b.fallback_used is False


def test_xkiro_backend_invalid_json_falls_back_with_same_input() -> None:
    prompts: dict[str, str] = {}
    images: dict[str, str] = {}

    def _fake(model_id: str, prompt: str, b64: str, image: Image.Image) -> str:
        prompts[model_id] = prompt
        images[model_id] = b64
        if model_id == ai.PRIMARY_MODEL:
            return "garbage {{{ not json"
        return _panel_payload("Fallback narration.")
    b = sa.XkiroVisionBackend(primary_model=ai.PRIMARY_MODEL,
                              fallback_model=ai.FALLBACK_MODEL,
                              api_key="test", request_fn=_fake)
    entries, _ = b.analyze_chunk(_img())
    assert entries[0].narration == "Fallback narration."
    assert b.fallback_used is True
    assert b.last_model_used == ai.FALLBACK_MODEL
    # input contract preserved: same prompt + same image to both models
    assert prompts[ai.PRIMARY_MODEL] == prompts[ai.FALLBACK_MODEL]
    assert images[ai.PRIMARY_MODEL] == images[ai.FALLBACK_MODEL]
    assert isinstance(b.last_primary_error, Exception)


def test_xkiro_backend_empty_response_falls_back() -> None:
    def _fake(model_id: str, prompt: str, b64: str, image: Image.Image) -> str:
        return "" if model_id == ai.PRIMARY_MODEL else _panel_payload()
    b = sa.XkiroVisionBackend(api_key="test", request_fn=_fake)
    entries, _ = b.analyze_chunk(_img())
    assert b.fallback_used is True and len(entries) == 1


def test_xkiro_backend_empty_panels_falls_back() -> None:
    empty = json.dumps({"panels": [], "characters": []})

    def _fake(model_id: str, prompt: str, b64: str, image: Image.Image) -> str:
        return empty if model_id == ai.PRIMARY_MODEL else _panel_payload("M")
    b = sa.XkiroVisionBackend(api_key="test", request_fn=_fake)
    entries, _ = b.analyze_chunk(_img())
    assert b.fallback_used is True
    assert entries[0].narration == "M"


def test_xkiro_backend_both_fail_surfaces_error() -> None:
    def _fake(model_id: str, prompt: str, b64: str, image: Image.Image) -> str:
        raise RuntimeError(f"{model_id} unavailable")
    b = sa.XkiroVisionBackend(api_key="test", request_fn=_fake)
    with pytest.raises(sa.VisionAnalysisError) as ei:
        b.analyze_chunk(_img())
    msg = str(ei.value).lower()
    assert "qwen" in msg and "mistral" in msg


def test_build_backend_accepts_xkiro_names() -> None:
    for name in ("xkiro", "qwen", "mistral"):
        b = gp.build_backend(name, api_key="test-key")
        assert isinstance(b, sa.XkiroVisionBackend)
        assert b.primary_model and b.fallback_model


def test_cache_prevents_duplicate_calls(tmp_path: Path) -> None:
    strip = tmp_path / "strip.png"
    Image.new("RGB", (200, 1200), (200, 200, 200)).save(strip)
    calls: list[str] = []

    def _fake(model_id: str, prompt: str, b64: str, image: Image.Image) -> str:
        calls.append(model_id)
        return json.dumps({
            "panels": [{"panel_index": 1, "y_start": 0, "y_end": 1000,
                        "narration": "n", "dialogue": "",
                        "panel_type": "single", "confidence": 0.9,
                        "bubble_boxes": []}],
            "characters": []})
    backend = sa.XkiroVisionBackend(api_key="test", request_fn=_fake)
    cache = tmp_path / "cache"
    plan1, used1 = sa.analyze_strip(strip, backend, cache_dir=cache)
    assert used1 is False
    n_calls = len(calls)
    assert n_calls >= 1
    plan2, used2 = sa.analyze_strip(strip, backend, cache_dir=cache)
    assert used2 is True  # rerun reuses cache: no new API calls
    assert len(calls) == n_calls
    assert [e.panel_index for e in plan2.entries] == [1]
    assert plan1.input_hash == plan2.input_hash


def test_deterministic_cutter_untouched_by_ai() -> None:
    # Blank/crop logic must not import AI models.
    import inspect

    import blank_detector
    import guided_cutter
    for mod in (blank_detector, guided_cutter):
        src = inspect.getsource(mod)
        assert "Qwen" not in src and "Mistral" not in src
        assert "ai_models" not in src


def test_text_fallback_wrapper() -> None:
    def _boom(model_id: str, system: str, user: str) -> str:
        if model_id == ai.PRIMARY_MODEL:
            raise ValueError("invalid JSON from primary")
        assert "ONLY JSON" in user or system
        return '{"entries": [{"panel_id": "p1", "speaker": null, "text": "Hi"}]}'
    out = ai.generate_text_with_fallback("sys", "user ONLY JSON", request_fn=_boom)
    assert out.fallback_used is True
    assert out.model_used == ai.FALLBACK_MODEL
    assert "Hi" in out.result
