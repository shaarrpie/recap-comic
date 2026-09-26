# tests/test_sfx.py
"""Offline tests for the automatic SFX system.

Covers the bank loader (category discovery + default/ fallback), the plan
tagger (transitions at cuts, word-timed action hits, reveal risers, nothing
on calm panels), determinism, bounds, cache-hash behavior, and a real ffmpeg
mixdown against a tiny generated mp4 (skipped when ffmpeg is unavailable).
"""
from __future__ import annotations

import subprocess
import wave

import pytest

import recap_video as rv
from adapters.schemas import (
    AudioArtifact,
    AudioEntry,
    BBox,
    Meta,
    NarrationArtifact,
    NarrationEntry,
    PanSpec,
    TimelineArtifact,
    TimelineEntry,
)


def _has_ffmpeg() -> bool:
    try:
        rv._resolve_ffmpeg()
        return True
    except Exception:
        return False


def _meta() -> Meta:
    return Meta(schema_version=1, generator="t", config_hash="c",
                input_hashes={})


def _entry(pid: str, order: int, start: float, dur: float,
           audio: bool = True) -> TimelineEntry:
    return TimelineEntry(
        panel_id=pid, order=order,
        source_image=f"panel_{order:03d}.png",
        bbox=BBox(x=0, y=0, w=800, h=800),
        start_seconds=start, duration_seconds=dur,
        audio_path=f"{pid}.mp3" if audio else None,
        pan=PanSpec(kind="static", scaled_w=1080,
                    scaled_h=1080, travel_px=0))


def _timeline() -> TimelineArtifact:
    # 001: calm intro, 002: action cut, 003: reveal cut (all spoken)
    return TimelineArtifact(
        meta=_meta(), width=1080, height=1920, fps=30,
        gap_seconds=0.35, min_display_seconds=2.0,
        entries=[_entry("001", 1, 0.0, 4.0),
                 _entry("002", 2, 4.0, 3.0),
                 _entry("003", 3, 7.0, 3.0)])


def _narration(texts: dict[str, str]) -> NarrationArtifact:
    return NarrationArtifact(
        meta=_meta(), mode="narrator",
        entries=[NarrationEntry(id=pid, panel_id=pid, order=i,
                                speaker=None, text=text, quotes=[])
                 for i, (pid, text) in enumerate(texts.items(), start=1)])


def _audio(words_by_panel: dict[str, list[dict]] | None = None) -> AudioArtifact:
    entries = []
    for pid, words in (words_by_panel or {}).items():
        entries.append(AudioEntry(entry_id=pid, path=f"{pid}.mp3",
                                  duration_seconds=2.0, words=words))
    return AudioArtifact(meta=_meta(), voice="v", entries=entries)


def _bank(root, *, transition=2, action=2, reveal=2, default=0):
    root.parent.mkdir(parents=True, exist_ok=True)
    for kind, n in (("transition", transition), ("action", action),
                    ("reveal", reveal), ("default", default)):
        d = root / kind
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"{kind}_{i:02d}.wav").write_bytes(b"RIFFdummy")
    (root / "transition" / "notes.txt").write_text("not audio")
    return root


def _cfg(bank, **kw) -> rv.VideoConfig:
    return rv.VideoConfig(sfx_dir=bank, **kw)


def _bank_tmp():
    # plan tests share one throwaway bank dir (content is never read as audio)
    import tempfile
    from pathlib import Path
    return _bank(Path(tempfile.mkdtemp(prefix="sfx_bank_")) / "bank")


# ------------------------------------------------------------------- bank --
def test_bank_loader_categories(tmp_path):
    bank = _bank(tmp_path / "bank", reveal=1)
    b = rv.load_sfx_bank(bank)
    assert [p.name for p in b["transition"]] == ["transition_00.wav",
                                                 "transition_01.wav"]
    assert [p.name for p in b["action"]] == ["action_00.wav", "action_01.wav"]
    assert [p.name for p in b["reveal"]] == ["reveal_00.wav"]
    # non-audio files are ignored
    assert all(p.suffix.lower() in rv._SFX_AUDIO_EXTS
               for files in b.values() for p in files)


def test_bank_loader_default_fallback(tmp_path):
    root = tmp_path / "bank"
    d = root / "default"
    d.mkdir(parents=True)
    (d / "hit_00.wav").write_bytes(b"RIFFdummy")
    b = rv.load_sfx_bank(root)
    assert [p.name for p in b["action"]] == ["hit_00.wav"]
    assert [p.name for p in b["transition"]] == ["hit_00.wav"]


def test_bank_loader_errors(tmp_path):
    with pytest.raises(rv.VideoError):
        rv.load_sfx_bank(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "transition").mkdir()
    with pytest.raises(rv.VideoError):
        rv.load_sfx_bank(empty)


# ------------------------------------------------------------------- plan --
def test_transitions_at_every_cut_except_first():
    plan = rv.build_sfx_plan(_timeline(), _narration({}), _audio(),
                             _cfg(_bank_tmp()))
    trans = [e for e in plan.events if e.kind == "transition"]
    assert [e.at_seconds for e in trans] == [pytest.approx(4.0),
                                             pytest.approx(7.0)]
    assert all(e.trigger == "panel_cut" for e in trans)


def test_transitions_skip_unnarrated_beats():
    """Un-narrated montage beats (no audio clip) must not fire a whoosh at
    every cut — with fillers in the timeline that is machine-gun sfx. A
    whoosh only opens a SPOKEN beat."""
    tl = TimelineArtifact(
        meta=_meta(), width=1080, height=1920, fps=30,
        gap_seconds=0.35, min_display_seconds=2.0,
        entries=[_entry("001", 1, 0.0, 4.0, audio=True),
                 _entry("002", 2, 4.0, 1.0, audio=False),   # filler beat
                 _entry("003", 3, 5.0, 1.0, audio=False),   # filler beat
                 _entry("004", 4, 6.0, 3.0, audio=True)])
    plan = rv.build_sfx_plan(tl, _narration({}), _audio(),
                             _cfg(_bank_tmp()))
    trans = [e for e in plan.events if e.kind == "transition"]
    # only the cut into the spoken 004 gets a whoosh
    assert [e.panel_id for e in trans] == ["004"]
    assert [e.at_seconds for e in trans] == [pytest.approx(6.0)]


def test_plan_source_resolves_in_category_bank(tmp_path):
    """Regression: build_sfx_plan must keep the category subfolder in the
    recorded source so mix_sfx can resolve ``bank_dir / source`` against the
    real ``<bank>/<kind>/`` layout. A bare basename pointed at a non-existent
    file and silently dropped every event at mixdown."""
    bank = _bank(tmp_path / "bank")
    plan = rv.build_sfx_plan(_timeline(), _narration({}), _audio(), _cfg(bank))
    trans = [e for e in plan.events if e.kind == "transition"]
    assert trans, "expected transition events"
    for ev in trans:
        assert str(ev.source).startswith("transition/")
        assert (bank / ev.source).is_file(), ev.source


def test_action_keyword_uses_word_timestamp(monkeypatch):
    # only panels WITH narration text are action; textless panels are calm
    monkeypatch.setattr("cinematic_effects.classify_panel",
                        lambda p: "action" if p.get("narration") else "calm")
    words = [{"start": 0.10, "end": 0.30, "text": "He"},
             {"start": 0.30, "end": 0.55, "text": "swung"},
             {"start": 0.55, "end": 0.70, "text": "his"},
             {"start": 0.70, "end": 0.95, "text": "blade."}]
    plan = rv.build_sfx_plan(
        _timeline(), _narration({"001": "He swung his blade fast."}),
        _audio({"001": words}), _cfg(_bank_tmp()))
    action = [e for e in plan.events if e.kind == "action"]
    assert len(action) == 1
    assert action[0].panel_id == "001"
    assert action[0].at_seconds == pytest.approx(0.30, abs=0.02)
    assert action[0].trigger == "keyword:swung"


def test_action_falls_back_to_panel_start(monkeypatch):
    monkeypatch.setattr("cinematic_effects.classify_panel",
                        lambda p: "action" if p.get("narration") else "calm")
    plan = rv.build_sfx_plan(
        _timeline(), _narration({"002": "He swung his blade fast."}),
        _audio(), _cfg(_bank_tmp()))     # no word timings at all
    action = [e for e in plan.events if e.kind == "action"]
    assert len(action) == 1
    assert action[0].at_seconds == pytest.approx(4.15, abs=0.01)
    assert action[0].trigger == "panel_class:action"


def test_reveal_class_gets_a_beat(monkeypatch):
    monkeypatch.setattr("cinematic_effects.classify_panel",
                        lambda p: "reveal" if p.get("narration") else "calm")
    plan = rv.build_sfx_plan(
        _timeline(), _narration({"003": "The gate opened slowly."}),
        _audio(), _cfg(_bank_tmp()))
    reveal = [e for e in plan.events if e.kind == "reveal"]
    assert len(reveal) == 1
    assert reveal[0].at_seconds == pytest.approx(7.1, abs=0.01)


def test_calm_panels_get_no_beat_sounds(monkeypatch):
    monkeypatch.setattr("cinematic_effects.classify_panel",
                        lambda p: "calm")
    plan = rv.build_sfx_plan(_timeline(), _narration({}), _audio(),
                             _cfg(_bank_tmp()))
    assert [e.kind for e in plan.events] == ["transition", "transition"]


def test_plan_is_deterministic_and_bounded():
    tl = _timeline()
    nar = _narration({"002": "He swung his blade fast.",
                      "003": "The gate opened slowly."})
    aud = _audio()
    cfg = _cfg(_bank_tmp())
    a = rv.build_sfx_plan(tl, nar, aud, cfg)
    b = rv.build_sfx_plan(tl, nar, aud, cfg)
    assert a.model_dump_json() == b.model_dump_json()
    total = rv.total_seconds(tl)
    assert all(0.0 <= e.at_seconds <= max(total - 0.2, 0.0)
               for e in a.events)
    # the plan survives a JSON round-trip through its pydantic model
    again = rv.SfxArtifact.model_validate_json(a.model_dump_json())
    assert again.events == a.events


def test_sfx_fields_excluded_from_essentials(tmp_path):
    assert (rv.VideoConfig().hash_essentials()
            == rv.VideoConfig(sfx_dir=tmp_path).hash_essentials())
    assert (rv.VideoConfig().hash()
            != rv.VideoConfig(sfx_dir=tmp_path).hash())


# ------------------------------------------------------------------- mix --
def _write_wav(path, seconds: float, rate: int = 8000) -> None:
    n = int(rate * seconds)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(b"\x00\x00" * n)


def _make_video(path, ffmpeg_exe: str, seconds: float = 1.5) -> None:
    subprocess.run(
        [ffmpeg_exe, "-y", "-f", "lavfi",
         "-i", "color=black:size=320x320:rate=30",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True)


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
def test_mix_sfx_overlays_and_keeps_video(tmp_path):
    ffmpeg = rv._resolve_ffmpeg()
    video = tmp_path / "recap.mp4"
    _make_video(video, ffmpeg)
    bank = tmp_path / "bank"
    bank.mkdir()
    _write_wav(bank / "hit.wav", 0.3)
    _write_wav(bank / "whoosh.wav", 0.3)
    plan = rv.SfxArtifact(meta=_meta(), bank_dir=str(bank), events=[
        rv.SfxEvent(id="sfx_001", kind="action", panel_id="001",
                    at_seconds=0.2, source="hit.wav", volume=0.9,
                    trigger="panel_class:action"),
        rv.SfxEvent(id="sfx_002", kind="transition", panel_id="002",
                    at_seconds=0.8, source="whoosh.wav", volume=0.5,
                    trigger="panel_cut")])
    cfg = rv.VideoConfig(sfx_dir=bank)
    assert rv.mix_sfx(video, plan, cfg) is True
    # the video stream survives (stream-copied): same file, same duration
    assert video.is_file()
    assert rv.probe_duration(video) == pytest.approx(1.5, abs=0.3)
    # regression: the narration/audio track must span the WHOLE video. With
    # amix duration=first the mix used to end with the first (delayed) SFX
    # clip, truncating all audio after ~0.5s while the picture kept going.
    assert _decoded_audio_seconds(video, ffmpeg) == pytest.approx(1.5,
                                                                  abs=0.3)
    # no temp litter
    assert not (tmp_path / "recap.sfx.mp4").exists()


def _decoded_audio_seconds(path, ffmpeg_exe: str) -> float:
    """Length of the decoded audio stream (last ffmpeg status timestamp)."""
    import re
    proc = subprocess.run([ffmpeg_exe, "-i", str(path), "-map", "0:a",
                           "-f", "null", "-"],
                          capture_output=True, text=True, check=False)
    stamps = re.findall(r"time=(\d+):(\d{2}):(\d{2}(?:[.,]\d+)?)",
                        proc.stderr)
    assert stamps, f"no audio stream decoded from {path}"
    h, m, s = stamps[-1]
    return int(h) * 3600 + int(m) * 60 + float(s.replace(",", "."))


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
def test_mix_sfx_skips_missing_bank_files(tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    plan = rv.SfxArtifact(meta=_meta(), bank_dir=str(bank), events=[
        rv.SfxEvent(id="sfx_001", kind="action", panel_id="001",
                    at_seconds=0.2, source="gone.wav", volume=0.9,
                    trigger="panel_class:action")])
    cfg = rv.VideoConfig(sfx_dir=bank)
    assert rv.mix_sfx(tmp_path / "nonexistent.mp4", plan, cfg) is False
