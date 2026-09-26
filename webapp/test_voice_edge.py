"""Webapp voice studio: provider='edge' is selectable and stays coherent.

Offline: these exercise only voice_api's config/validation/coherence logic,
never a real network synth call. The point is that the (provider, voice) pair
is never persisted broken, because edge-tts fails SILENTLY on a Kokoro id
(see the project pitfall notes).
"""
from __future__ import annotations

import asyncio
import secrets

from webapp import voice_api


def _sid() -> str:
    return secrets.token_hex(6)  # 12 hex chars (matches _session_dir regex)


def test_edge_is_a_supported_provider():
    assert "edge" in voice_api.PROVIDERS


def test_put_voice_edge_coerces_kokoro_voice():
    cfg = voice_api.put_voice(_sid(), {"provider": "edge", "voice": "af_heart"})
    assert cfg["provider"] == "edge"
    # af_heart would make edge-tts fail silently -> coerced to a real edge id
    assert cfg["voice"] == voice_api.DEFAULT_EDGE_VOICE


def test_get_voice_preserves_edge_and_lists_catalogue():
    s = _sid()
    voice_api.put_voice(s, {"provider": "edge", "voice": "en-US-GuyNeural"})
    got = voice_api.get_voice(s)
    assert got["provider"] == "edge"
    assert got["voice"] == "en-US-GuyNeural"
    listed = asyncio.run(voice_api.list_voices("edge"))
    assert listed["count"] == len(voice_api.EDGE_VOICES)
    assert "en-US-AriaNeural" in {v["id"] for v in listed["voices"]}


def test_put_voice_kokoro_rejects_neural_voice():
    cfg = voice_api.put_voice(_sid(), {"provider": "kokoro",
                                       "voice": "en-US-AriaNeural"})
    assert cfg["provider"] == "kokoro"
    assert cfg["voice"] == "af_heart"
