# tests/test_speech_fit.py
"""Offline tests for the opt-in narration-fit (tempo) backstop.

By default the narrator is never tempo-shifted (speech_fit_audio is off) —
a line keeps its natural pace and the panel catches up. When the backstop is
explicitly enabled, every MEASURED TTS clip over speech_max_seconds is
tempo-fitted (ffmpeg atempo, pitch preserved) back inside the budget;
overrun beyond the bounded speedup is trimmed with a loud warning
(pathological audio only). These tests turn the flag on to exercise it.
Rendering needs ffmpeg; tests skip cleanly when it is unavailable.
"""
from __future__ import annotations

import wave

import pytest

import recap_video as rv
from adapters.schemas import AudioArtifact, AudioEntry, Meta, NarrationArtifact, NarrationEntry


def _has_ffmpeg() -> bool:
    try:
        rv._resolve_ffmpeg()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_ffmpeg(),
                                reason="ffmpeg not available")


def _write_wav(path, seconds: float, rate: int = 8000) -> None:
    n = int(rate * seconds)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(b"\x00\x00" * n)


def _probe(path) -> float:
    return rv.probe_duration(path)


def _entry(seconds: float, words: list[dict] | None = None) -> AudioEntry:
    return AudioEntry(entry_id="001", path="001.wav",
                      duration_seconds=seconds, words=words or [])


# ------------------------------------------------------------------- unit --
def test_short_clip_passes_through_untouched(tmp_path):
    cfg = rv.VideoConfig(speech_fit_audio=True)
    p = tmp_path / "001.wav"
    _write_wav(p, 2.0)
    data = p.read_bytes()
    out = rv.fit_narration_clip(_entry(2.0), tmp_path, cfg, _probe)
    assert out.duration_seconds == 2.0
    assert p.read_bytes() == data            # bytes untouched


def test_overlong_clip_is_tempo_fitted_under_budget(tmp_path):
    cfg = rv.VideoConfig(speech_fit_audio=True)
    p = tmp_path / "001.wav"
    _write_wav(p, 8.0)
    out = rv.fit_narration_clip(_entry(8.0), tmp_path, cfg, _probe)
    assert out.duration_seconds <= cfg.speech_max_seconds + 0.05
    # the clip file on disk IS the fitted one (probe and entry agree)
    assert _probe(p) == pytest.approx(out.duration_seconds, abs=0.2)


def test_word_timestamps_scale_with_tempo(tmp_path):
    cfg = rv.VideoConfig(speech_fit_audio=True)
    p = tmp_path / "001.wav"
    _write_wav(p, 8.0)
    factor = 8.0 / cfg.speech_max_seconds      # 1.74x, inside the bound
    entry = _entry(8.0, [{"start": 1.0, "end": 2.0, "text": "hi"}])
    out = rv.fit_narration_clip(entry, tmp_path, cfg, _probe)
    assert out.words[0]["start"] == pytest.approx(1.0 / factor, abs=0.2)
    assert out.words[0]["end"] == pytest.approx(2.0 / factor, abs=0.2)


def test_trim_fallback_when_max_speedup_cannot_fit(tmp_path):
    cfg = rv.VideoConfig(speech_fit_audio=True, speech_fit_max_speedup=2.0)
    p = tmp_path / "001.wav"
    _write_wav(p, 20.0)                       # 20/4.6 = 4.35x > 2.0x bound
    out = rv.fit_narration_clip(_entry(20.0), tmp_path, cfg, _probe)
    assert out.duration_seconds <= cfg.speech_max_seconds + 0.05
    # no temp litter left behind
    assert not (tmp_path / "001.fit.wav").exists()
    assert not (tmp_path / "001.trim.wav").exists()


def test_fit_disabled_by_flags(tmp_path):
    p = tmp_path / "001.wav"
    _write_wav(p, 8.0)
    data = p.read_bytes()
    for cfg in (rv.VideoConfig(speech_window=False, speech_fit_audio=True),
                rv.VideoConfig(speech_fit_audio=False)):
        out = rv.fit_narration_clip(_entry(8.0), tmp_path, cfg, _probe)
        assert out.duration_seconds == 8.0
    assert p.read_bytes() == data


def test_missing_clip_file_is_skipped(tmp_path):
    cfg = rv.VideoConfig(speech_fit_audio=True)
    out = rv.fit_narration_clip(_entry(8.0), tmp_path, cfg, _probe)
    assert out.duration_seconds == 8.0


def test_fit_fields_participate_in_hashes():
    assert (rv.VideoConfig().hash()
            != rv.VideoConfig(speech_fit_max_speedup=3.0).hash())
    assert (rv.VideoConfig(speech_fit_audio=True).hash_essentials()
            != rv.VideoConfig(speech_fit_audio=False).hash_essentials())


# ------------------------------------------------------- end-to-end (TTS) --
def test_synthesize_audio_fits_measured_clips(tmp_path, monkeypatch):
    """A slow 'voice' handing back an 8s clip comes back under the budget."""
    def fake_synth(entry, out_dir, **kwargs):
        _write_wav(out_dir / "001.wav", 8.0)
        return AudioEntry(entry_id=entry.id, path="001.wav",
                          duration_seconds=8.0, words=[]), None

    monkeypatch.setattr("adapters.tts.synthesize_entry", fake_synth)
    cfg = rv.VideoConfig(tts="kokoro", speech_fit_audio=True)
    nar = NarrationArtifact(
        meta=Meta(schema_version=1, generator="t", config_hash="c",
                  input_hashes={}),
        mode="narrator",
        entries=[NarrationEntry(id="001", panel_id="001", order=1,
                                speaker=None, text="He runs.", quotes=[])])
    aud = rv.synthesize_audio(nar, tmp_path, cfg, force=True)
    assert aud.entries, "clip must be synthesized"
    assert aud.entries[0].duration_seconds <= cfg.speech_max_seconds + 0.05


def test_synthesize_audio_refits_over_budget_clip_on_cache_hit(tmp_path):
    """A cached clip that measures over budget — e.g. one whose fit was
    skipped in an earlier run because ffmpeg was momentarily unavailable —
    must be re-fitted on a cache hit, so stale audio can never push a panel
    past the sub-5s window."""
    cfg = rv.VideoConfig(tts="kokoro", speech_fit_audio=True)
    nar = NarrationArtifact(
        meta=Meta(schema_version=1, generator="t", config_hash="c",
                  input_hashes={}),
        mode="narrator",
        entries=[NarrationEntry(id="001", panel_id="001", order=1,
                                speaker=None, text="He runs.", quotes=[])])
    # Plant an audio.json whose hashes match `nar`/`cfg` but whose clip is a
    # stale 8s file (over the 4.6s budget), as a prior un-fit run would leave.
    input_hashes = {
        "narration.json": rv._sha256_text(nar.model_dump_json()),
        "tts_input": rv._sha256_text(
            f"001:He runs.:{cfg.voice}:{cfg.tts}"),
    }
    stale = AudioArtifact(
        meta=Meta(schema_version=1, generator="t",
                  config_hash=cfg.hash_essentials(), input_hashes=input_hashes),
        voice=cfg.voice,
        entries=[AudioEntry(entry_id="001", path="001.wav",
                            duration_seconds=8.0, words=[])])
    (tmp_path / "audio.json").write_text(stale.model_dump_json())
    _write_wav(tmp_path / "001.wav", 8.0)

    aud = rv.synthesize_audio(nar, tmp_path, cfg, force=False)
    assert aud.entries[0].duration_seconds <= cfg.speech_max_seconds + 0.05
    # the on-disk clip was actually re-fit, not just the metadata
    assert _probe(tmp_path / "001.wav") <= cfg.speech_max_seconds + 0.2
