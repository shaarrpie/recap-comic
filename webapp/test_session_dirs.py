# webapp/test_session_dirs.py
"""Session-dir hygiene + voice-preview cache correctness.

Covers three fixes:
  * GET /api/voice/<id> and /api/editor/<id> must NOT materialize a phantom
    session directory (it used to, and the empty dir then appeared in
    /api/projects).
  * The voice-preview cache key must cover the SAME text that is synthesized.
    It hashed text[:80] while synth received text[:160], so two prompts
    identical for the first 80 chars shared one cache file and the FIRST
    one's audio was served for both.
  * A failed preview synthesis must not leave a file behind that later calls
    accept (callers gate on is_file() only).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from webapp import editor_api, voice_api
from webapp import main as webmain

_VALID = "0123456789ab"   # 12 hex chars, matches the session-id regex


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Point both API modules at an isolated root so a created directory is
    # observable and never touches the real webapp_output.
    monkeypatch.setattr(voice_api, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(editor_api, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(webmain, "BASE_DIR", tmp_path)
    return TestClient(webmain.app)


def test_voice_get_creates_no_phantom_session(client, tmp_path):
    r = client.get(f"/api/voice/{_VALID}")
    assert r.status_code == 200
    assert not (tmp_path / _VALID).exists(), (
        "GET /api/voice/<id> must not materialize a session directory")


def test_editor_get_creates_no_phantom_session(client, tmp_path):
    r = client.get(f"/api/editor/{_VALID}")
    assert r.status_code in (400, 404)
    assert not (tmp_path / _VALID).exists(), (
        "GET /api/editor/<id> must not materialize a session directory")


def test_preview_cache_key_covers_full_synthesized_text(client, monkeypatch):
    """Distinct prompts sharing an 80-char prefix must not collide."""
    written: dict[str, str] = {}

    async def fake_synth_one(text, voice, out, rate="", pitch="", timeout_s=60):
        # Mimic the atomic write the real helper now performs.
        out.write_text(f"audio-for:{text}", encoding="utf-8")
        written[text] = out.name

    monkeypatch.setattr("webapp.tts_helpers.synth_one", fake_synth_one)

    prefix = "a" * 80          # identical for both requests
    r1 = client.get("/api/voice/preview",
                    params={"voice": "en-US-AriaNeural", "text": prefix + "TAIL-ONE"})
    r2 = client.get("/api/voice/preview",
                    params={"voice": "en-US-AriaNeural", "text": prefix + "TAIL-TWO"})
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.content != r2.content, "distinct prompts served identical audio"
    assert written[prefix + "TAIL-ONE"] != written[prefix + "TAIL-TWO"], (
        "distinct prompts shared one cache file")


def test_preview_cache_reuses_for_identical_text(client, monkeypatch):
    """The same prompt must hit the cache (one synthesis, not two)."""
    calls = []

    async def fake_synth_one(text, voice, out, rate="", pitch="", timeout_s=60):
        calls.append(text)
        out.write_text(f"audio-for:{text}", encoding="utf-8")

    monkeypatch.setattr("webapp.tts_helpers.synth_one", fake_synth_one)

    params = {"voice": "en-US-AriaNeural", "text": "same prompt twice"}
    first = client.get("/api/voice/preview", params=params)
    second = client.get("/api/voice/preview", params=params)
    assert first.status_code == 200 and second.status_code == 200
    assert first.content == second.content
    assert len(calls) == 1, "identical prompt must be served from cache"


def test_failed_preview_leaves_no_cached_file(client, monkeypatch):
    """A failed synthesis must not poison the cache for later calls."""
    async def boom(text, voice, out, rate="", pitch="", timeout_s=60):
        raise RuntimeError("network down")

    monkeypatch.setattr("webapp.tts_helpers.synth_one", boom)

    r = client.get("/api/voice/preview",
                   params={"voice": "en-US-AriaNeural", "text": "fails"})
    assert r.status_code == 502

    cache = next((webmain.BASE_DIR / ".cache" / "voice_preview").glob("*.mp3"), None)
    assert cache is None, "a failed synthesis must leave no cached mp3 behind"
