# tests/test_ai_fallback.py
"""Agnes 2.5 Flash primary + 2.0 Flash fallback (offline, mocked).

Covers: primary success (no fallback call), primary failure/timeout/
invalid-JSON -> fallback, fallback success metadata, both-fail error,
image+prompt parity, cache reuse, deterministic cutter independence.
"""
from __future__ import annotations

import json
import types
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
            "narrate", lambda: (_ for _ in ()).throw(ValueError("primary 500")),
            lambda: (_ for _ in ()).throw(ValueError("fallback 503")))
    assert "narrate" in str(ei.value)
    assert isinstance(ei.value.primary_error, ValueError)
    assert isinstance(ei.value.fallback_error, ValueError)
    assert "primary 500" in str(ei.value) and "fallback 503" in str(ei.value)


def test_resolve_model_id_aliases() -> None:
    assert ai.resolve_model_id("Agnes 2.5 Flash") == ai.PRIMARY_MODEL
    assert ai.resolve_model_id("Agnes 2.0 Flash") == ai.FALLBACK_MODEL
    assert ai.resolve_model_id("agnes-2.5-flash") == ai.PRIMARY_MODEL
    assert ai.resolve_model_id("agnes-2.0-flash") == ai.FALLBACK_MODEL
    assert ai.resolve_model_id(ai.PRIMARY_MODEL) == ai.PRIMARY_MODEL


def test_agnes_backend_primary_success_fallback_not_called() -> None:
    seen: list[str] = []

    def _fake(model_id: str, prompt: str, b64: str, image: Image.Image) -> str:
        seen.append(model_id)
        assert b64  # actual image bytes must be passed
        assert "panels" in prompt  # same task instructions
        return _panel_payload()
    b = sa.AgnesVisionBackend(primary_model=ai.PRIMARY_MODEL,
                              fallback_model=ai.FALLBACK_MODEL,
                              api_key="test", request_fn=_fake)
    entries, _chars = b.analyze_chunk(_img())
    assert len(entries) == 1 and entries[0].narration == "A hero stands."
    assert seen == [ai.PRIMARY_MODEL]
    assert b.last_model_used == ai.PRIMARY_MODEL
    assert b.fallback_used is False


def test_agnes_backend_invalid_json_falls_back_with_same_input() -> None:
    prompts: dict[str, str] = {}
    images: dict[str, str] = {}

    def _fake(model_id: str, prompt: str, b64: str, image: Image.Image) -> str:
        prompts[model_id] = prompt
        images[model_id] = b64
        if model_id == ai.PRIMARY_MODEL:
            return "garbage {{{ not json"
        return _panel_payload("Fallback narration.")
    b = sa.AgnesVisionBackend(primary_model=ai.PRIMARY_MODEL,
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


def test_agnes_backend_empty_response_falls_back() -> None:
    def _fake(model_id: str, prompt: str, b64: str, image: Image.Image) -> str:
        return "" if model_id == ai.PRIMARY_MODEL else _panel_payload()
    b = sa.AgnesVisionBackend(api_key="test", request_fn=_fake)
    entries, _ = b.analyze_chunk(_img())
    assert b.fallback_used is True and len(entries) == 1


def test_agnes_backend_empty_panels_falls_back() -> None:
    empty = json.dumps({"panels": [], "characters": []})

    def _fake(model_id: str, prompt: str, b64: str, image: Image.Image) -> str:
        return empty if model_id == ai.PRIMARY_MODEL else _panel_payload("M")
    b = sa.AgnesVisionBackend(api_key="test", request_fn=_fake)
    entries, _ = b.analyze_chunk(_img())
    assert b.fallback_used is True
    assert entries[0].narration == "M"


def test_agnes_backend_both_fail_surfaces_error() -> None:
    def _fake(model_id: str, prompt: str, b64: str, image: Image.Image) -> str:
        raise RuntimeError(f"{model_id} unavailable")
    b = sa.AgnesVisionBackend(api_key="test", request_fn=_fake)
    with pytest.raises(sa.VisionAnalysisError) as ei:
        b.analyze_chunk(_img())
    msg = str(ei.value).lower()
    assert "agnes-2.5-flash" in msg and "agnes-2.0-flash" in msg


def test_agnes_backend_both_empty_panels_is_blank_chunk() -> None:
    """Both models returning an empty panel list is a VALID blank chunk.

    A genuinely empty 2000px stretch between scenes used to fail both
    models, surface as VisionAnalysisError, and silently degrade the whole
    strip to the no-narration gutter fallback.
    """
    empty = json.dumps({"panels": [], "characters": []})
    called: list[str] = []

    def _fake(model_id: str, prompt: str, b64: str, image: Image.Image) -> str:
        called.append(model_id)
        return empty
    b = sa.AgnesVisionBackend(api_key="test", request_fn=_fake)
    entries, characters = b.analyze_chunk(_img())
    assert entries == []
    assert characters == []
    # The fallback model still got its chance before the chunk was accepted
    # as blank (fallback_used is only set on the success path, so verify
    # via the request log).
    assert called == [ai.PRIMARY_MODEL, ai.FALLBACK_MODEL]


def test_agnes_backend_honors_explicit_api_key() -> None:
    """An explicit api_key (webapp settings UI, --api-key) must win over env."""
    b = sa.AgnesVisionBackend(api_key="explicit-key-from-ui",
                              request_fn=lambda *a: ("{}", None))
    assert b._api_key == "explicit-key-from-ui"
    for name in ("agnes", "xkiro", "gemini", "openai", "local", "cloudflare"):
        b = gp.build_backend(name, api_key="test-key")
        assert isinstance(b, sa.AgnesVisionBackend)
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
    backend = sa.AgnesVisionBackend(api_key="test", request_fn=_fake)
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
        assert "agnes-2.5-flash" not in src and "agnes-2.0-flash" not in src
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

# ------------------------------------------------- truncation + resume -----
# Silent-failure hardening: truncated LLM output, per-chunk resume, and
# corrupt-cache tolerance (all offline; no API key involved).


def test_truncation_reason_normalization() -> None:
    """Every provider's 'hit the output ceiling' reason maps to True, and
    normal completions do not."""
    enum_like = types.SimpleNamespace(name="MAX_TOKENS")   # SDK enum shape
    assert sa.is_truncated_response(enum_like) is True
    assert sa.is_truncated_response("FinishReason.MAX_TOKENS") is True
    assert sa.is_truncated_response("length") is True          # OpenAI-compat
    assert sa.is_truncated_response("max_tokens") is True
    assert sa.is_truncated_response("stop") is False
    assert sa.is_truncated_response("end_turn") is False
    assert sa.is_truncated_response(None) is False


def test_truncation_raises_before_parsing() -> None:
    """finish_reason 'length' must raise TruncatedResponseError instead of
    surfacing as a JSON parse error, retrying verbatim, and silently
    degrading the strip to the no-AI gutter fallback."""
    resp = types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            finish_reason="length",
            message=types.SimpleNamespace(content='{"panels": [{"panel_index'))])
    with pytest.raises(sa.TruncatedResponseError, match="TRUNCATED"):
        sa.raise_if_truncated(sa._openai_finish_reason(resp), backend="agnes",
                              model="m", max_tokens=4096,
                              image_size=(800, 2000))
    # a normal completion never raises
    ok = types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            finish_reason="stop",
            message=types.SimpleNamespace(content=_panel_payload()))])
    sa.raise_if_truncated(sa._openai_finish_reason(ok), backend="agnes",
                          model="m", max_tokens=4096, image_size=(800, 2000))


def test_truncated_chunk_is_retried_with_shorter_answer_instruction(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """An identical prompt at temperature 0 reproduces the same truncation,
    so the retry must carry the truncation feedback instead."""
    monkeypatch.setattr(sa.time, "sleep", lambda _s: None)
    prompts: list[str] = []

    class _TruncThenOk:
        name = "trunc"

        def analyze_chunk(self, image: Image.Image, previous_context: str = "",
                          retry_feedback: str = ""
                          ) -> tuple[list[sa.PanelPlanEntry], list[str]]:
            prompts.append(retry_feedback)
            if prompts[-1]:
                return [sa.PanelPlanEntry(panel_index=1, y_start=0, y_end=100,
                                          narration="ok")], []
            raise sa.TruncatedResponseError("agnes response was TRUNCATED")

    entries, _chars = sa._call_with_retry(_TruncThenOk(), _img(), attempts=3)
    assert [e.narration for e in entries] == ["ok"]
    assert prompts[0] == ""
    assert prompts[1] == sa.TRUNCATION_RETRY_FEEDBACK
    assert "TRUNCATED" in prompts[1]


def test_agnes_truncated_primary_falls_back_to_secondary() -> None:
    calls: list[str] = []

    def _fake(model_id: str, prompt: str, b64: str,
              image: Image.Image) -> str:
        calls.append(model_id)
        if len(calls) == 1:
            raise sa.TruncatedResponseError(
                "agnes response was TRUNCATED (finish reason 'length')")
        return _panel_payload()

    backend = sa.AgnesVisionBackend(api_key="test", request_fn=_fake)
    entries, _chars = backend.analyze_chunk(_img())
    assert [e.narration for e in entries] == ["A hero stands."]
    assert backend.fallback_used is True
    assert len(calls) == 2   # primary truncated, secondary completed


def _fake_openai_client(monkeypatch: pytest.MonkeyPatch,
                        calls: list[str], fail_keys: set[str]):
    """Patch openai.OpenAI with an in-memory fake.

    Calls using a key in `fail_keys` raise a 429; all others return a
    valid one-panel JSON payload. Records every key the client is built
    with (proves rotation order without ever logging key values).
    """
    import openai as _openai

    class _FakeClient:
        def __init__(self, api_key: str | None = None, **kwargs) -> None:
            self._key = api_key

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, model: str | None = None, **kwargs):
            calls.append(self._key or "")
            if (self._key or "") in fail_keys:
                raise RuntimeError("429 Too Many Requests")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(
                finish_reason="stop",
                message=types.SimpleNamespace(content=_panel_payload()))])

    monkeypatch.setattr(_openai, "OpenAI", _FakeClient)


def _isolated_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                   pool: str) -> None:
    """Two-key pool with no manual key and no single-key interference."""
    monkeypatch.delenv("AGNES_API_KEY", raising=False)
    monkeypatch.setenv("AGNES_API_KEYS", pool)
    monkeypatch.setattr(ai, "OUTPUT_DIR", tmp_path)
    ai._pool_cursor = 0  # deterministic round-robin start for assertions
    # No waiting by default: wait behavior has dedicated tests below.
    monkeypatch.setenv("AGNES_RATE_WAIT_ROUNDS", "0")


def test_agnes_backend_rotates_keys_on_rate_limit(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Key #1 429s on both models -> the SAME chunk succeeds on key #2."""
    _isolated_pool(monkeypatch, tmp_path, "k1,k2")
    calls: list[str] = []
    _fake_openai_client(monkeypatch, calls, fail_keys={"k1"})
    b = sa.AgnesVisionBackend()
    entries, _ = b.analyze_chunk(_img())
    assert [e.narration for e in entries] == ["A hero stands."]
    # primary(k1) 429 -> fallback(k1) 429 -> rotate -> primary(k2) ok
    assert calls == ["k1", "k1", "k2"]
    assert b.last_model_used == ai.PRIMARY_MODEL


def test_agnes_backend_gives_up_after_all_keys_rate_limited(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every key 429s -> VisionAnalysisError after trying each once."""
    _isolated_pool(monkeypatch, tmp_path, "k1,k2")
    calls: list[str] = []
    _fake_openai_client(monkeypatch, calls, fail_keys={"k1", "k2"})
    b = sa.AgnesVisionBackend()
    with pytest.raises(sa.VisionAnalysisError, match="all 2 key"):
        b.analyze_chunk(_img())
    assert calls == ["k1", "k1", "k2", "k2"]


def test_rate_wait_config_defaults_and_env(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGNES_RATE_WAIT_S", raising=False)
    monkeypatch.delenv("AGNES_RATE_WAIT_ROUNDS", raising=False)
    assert ai._rate_wait_config() == (120.0, 2)
    monkeypatch.setenv("AGNES_RATE_WAIT_S", "30")
    monkeypatch.setenv("AGNES_RATE_WAIT_ROUNDS", "1")
    assert ai._rate_wait_config() == (30.0, 1)
    monkeypatch.setenv("AGNES_RATE_WAIT_S", "junk")
    monkeypatch.setenv("AGNES_RATE_WAIT_ROUNDS", "-5")
    assert ai._rate_wait_config() == (120.0, 0)


def test_key_rotation_waits_then_retries_pool(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Whole-pool 429 -> sleep once (linear backoff) -> retry succeeds."""
    sleeps: list[float] = []
    monkeypatch.setattr(ai.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setenv("AGNES_RATE_WAIT_S", "30")
    monkeypatch.setenv("AGNES_RATE_WAIT_ROUNDS", "1")
    ai._pool_cursor = 0
    calls: list[str] = []

    def _fn(key: str) -> str:
        calls.append(key)
        if len(calls) <= 2:  # both keys throttled on round 1
            raise ai.AIFallbackError("op", ValueError("429 slow down"),
                                     ValueError("429 slow down"))
        return "recovered"

    out = ai.call_with_key_rotation("op", _fn, keys=["k1", "k2"])
    assert out == "recovered"
    assert calls == ["k1", "k2", "k1"]  # round-robin restart after the wait
    assert sleeps == [30.0]


def test_key_rotation_exhaustion_raises_last_error(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Rounds used up -> the last AIFallbackError surfaces (no hang)."""
    sleeps: list[float] = []
    monkeypatch.setattr(ai.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setenv("AGNES_RATE_WAIT_S", "30")
    monkeypatch.setenv("AGNES_RATE_WAIT_ROUNDS", "1")
    ai._pool_cursor = 0

    def _fn(key: str) -> str:
        raise ai.AIFallbackError("op", ValueError("429 down"),
                                 ValueError("429 down"))

    with pytest.raises(ai.AIFallbackError):
        ai.call_with_key_rotation("op", _fn, keys=["k1", "k2"])
    assert sleeps == [30.0]  # one wait between the two rounds, then raise


def test_key_rotation_non_rate_error_fails_fast(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse errors never wait: only rate limits deserve the sleep."""
    sleeps: list[float] = []
    monkeypatch.setattr(ai.time, "sleep", lambda s: sleeps.append(s))
    ai._pool_cursor = 0
    calls: list[str] = []

    def _fn(key: str) -> str:
        calls.append(key)
        raise ai.AIFallbackError("op", ValueError("bad json"),
                                 ValueError("bad json"))

    with pytest.raises(ai.AIFallbackError):
        ai.call_with_key_rotation("op", _fn, keys=["k1", "k2"])
    assert calls == ["k1"] and sleeps == []


def test_agnes_chunk_waits_out_throttle_then_succeeds(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Vision chunk: pool throttled -> one wait -> same chunk lands."""
    _isolated_pool(monkeypatch, tmp_path, "k1,k2")
    monkeypatch.setenv("AGNES_RATE_WAIT_S", "30")
    monkeypatch.setenv("AGNES_RATE_WAIT_ROUNDS", "1")
    sleeps: list[float] = []
    monkeypatch.setattr(sa.time, "sleep", lambda s: sleeps.append(s))
    calls: list[str] = []

    def _fake(model_id: str, prompt: str, b64: str, image) -> str:
        calls.append(model_id)
        if len(calls) <= 4:  # k1+k2 throttled across both models
            raise RuntimeError("429 Too Many Requests")
        return _panel_payload()

    b = sa.AgnesVisionBackend(request_fn=_fake)
    entries, _ = b.analyze_chunk(_img())
    assert [e.narration for e in entries] == ["A hero stands."]
    assert len(calls) == 5  # 4 throttled + 1 success
    assert sleeps == [30.0]


def test_failed_chunk_resumes_from_chunk_cache(tmp_path: Path) -> None:
    """One failing chunk must not waste the tokens already spent: the chunks
    that succeeded are cached per-chunk and a re-run only re-sends the rest."""
    strip = tmp_path / "strip.png"
    Image.new("RGB", (200, 2200), (200, 200, 200)).save(strip)
    cache = tmp_path / "cache"
    ok_entries = [sa.PanelPlanEntry(panel_index=1, y_start=0, y_end=1000,
                                    narration="n")]

    class HealingTailChunk:
        """Chunk 1 always answers; the short tail chunk fails until healed."""
        name = "healing-tail"
        model = "m"

        def __init__(self) -> None:
            self.calls = 0
            self.healed = False

        def analyze_chunk(self, image: Image.Image, previous_context: str = "",
                          retry_feedback: str = ""
                          ) -> tuple[list[sa.PanelPlanEntry], list[str]]:
            self.calls += 1
            if not self.healed and image.height < 1000:
                raise RuntimeError("chunk 2 exploded")
            return ok_entries, []

    first = HealingTailChunk()
    with pytest.raises(sa.VisionAnalysisError, match="are cached"):
        sa.analyze_strip(strip, first, cache_dir=cache, attempts=1)
    assert first.calls == 2                  # chunk 1 ok, chunk 2 failed
    # Rerun: chunk 1 must come from the chunk cache — only chunk 2 is re-sent.
    second = HealingTailChunk()
    with pytest.raises(sa.VisionAnalysisError):
        sa.analyze_strip(strip, second, cache_dir=cache, attempts=1)
    assert second.calls == 1
    assert list((cache / "chunks").glob("chunk_*.json"))

    # Once the model behaves, the resumed run completes with ONE model call
    # and the full plan is cached for later runs.
    third = HealingTailChunk()
    third.healed = True
    plan, used = sa.analyze_strip(strip, third, cache_dir=cache, attempts=1)
    assert used is False
    assert [e.panel_index for e in plan.entries] == [1, 2]
    assert third.calls == 1                  # chunk 1 was a chunk-cache hit
    _plan2, used2 = sa.analyze_strip(strip, HealingTailChunk(), cache_dir=cache)
    assert used2 is True                     # full plan cache now exists


def test_corrupt_phase1_cache_is_reanalyzed_not_fatal(tmp_path: Path) -> None:
    """A truncated/garbage cache file used to hard-crash every later run of
    the strip; it must be discarded and rebuilt instead.

    With the per-chunk cache the rebuild needs NO model calls at all (chunk
    answers are still on disk), which is exactly the cost profile we want.
    """
    strip = tmp_path / "strip.png"
    Image.new("RGB", (200, 600), (200, 200, 200)).save(strip)
    cache = tmp_path / "cache"
    plan1, _used1 = sa.analyze_strip(
        strip, sa.AgnesVisionBackend(api_key="test",
                                     request_fn=lambda *a: _panel_payload()),
        cache_dir=cache)
    assert plan1.entries
    plan_path = next(cache.glob("plan_*.json"))
    plan_path.write_text('{"source": "strip.png", "wid', encoding="utf-8")

    def _counting(model_id: str, prompt: str, b64: str,
                  image: Image.Image) -> str:
        raise AssertionError("the per-chunk cache must avoid a model call")

    plan2, used2 = sa.analyze_strip(
        strip, sa.AgnesVisionBackend(api_key="test", request_fn=_counting),
        cache_dir=cache)
    assert used2 is False                       # corrupt plan cache discarded
    assert [e.panel_index for e in plan2.entries] == [1]
    assert plan2.input_hash == plan1.input_hash
    # the plan cache file was rewritten with a VALID artifact
    sa.PanelPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))

    # A hash-mismatched cache (different strip, same file name) is discarded
    # too — never returned as a hit.
    wrong = sa.PanelPlan(
        source="strip.png", width=200, height=600, model="test",
        config_hash="t", input_hash="f" * 64,
        entries=[sa.PanelPlanEntry(panel_index=1, y_start=0, y_end=100,
                                   narration="stale")])
    plan_path.write_text(wrong.model_dump_json(), encoding="utf-8")
    _plan3, used3 = sa.analyze_strip(
        strip, sa.AgnesVisionBackend(api_key="test", request_fn=_counting),
        cache_dir=cache)
    assert used3 is False
