"""Edge TTS is selectable in the render config (webapp rewrite Phase 1).

``VideoConfig`` is a std dataclass, so its ``Literal`` is a type hint only;
these tests lock the documented edge/kokoro/none surface the webapp relies on
(the adapters.tts dispatcher already routes provider=="edge" to tts_edge).
"""
from __future__ import annotations

from recap_video import VideoConfig


def test_video_config_tts_accepts_edge():
    cfg = VideoConfig(tts="edge", voice="en-US-AriaNeural")
    assert cfg.tts == "edge"


def test_tts_literal_documents_all_providers():
    ann = str(VideoConfig.__annotations__["tts"])
    assert "edge" in ann and "kokoro" in ann and "none" in ann
