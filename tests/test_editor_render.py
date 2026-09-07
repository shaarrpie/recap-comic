# tests/test_editor_render.py
"""Test rendering an edited project without regenerating narration/audio."""
import io
import json

import pytest
from PIL import Image

from adapters.editor import Editor, EditorProject
from adapters.schemas import AudioArtifact, AudioEntry, BBox, Meta, NarrationArtifact, NarrationEntry, TimelineArtifact
from recap_video import VideoConfig, render_edited_project, total_seconds


def _make_panel(path, color=(100, 150, 200), size=(800, 1200)):
    img = Image.new("RGB", size, color)
    img.save(path, "PNG")


def test_render_edited_project_reorders_and_changes_duration(tmp_path):
    panels_dir = tmp_path / "panels"
    panels_dir.mkdir()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # create panel images
    for i in range(1, 3):
        _make_panel(panels_dir / f"panel_{i:03d}.png", color=(100 + i * 50, 150, 200))

    # create audio files (silent WAV) using wave module for validity
    for i in range(1, 3):
        wav_path = audio_dir / f"panel_{i:03d}.wav"
        import wave
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(48000)
            w.writeframes(b"\x00\x00" * 48000)

    # build narration + audio artifacts
    narration = NarrationArtifact(
        meta=Meta(schema_version=1, generator="test", config_hash="x", input_hashes={}),
        mode="narrator",
        entries=[
            NarrationEntry(id="panel_001", panel_id="panel_001", order=1, text="First panel narration"),
            NarrationEntry(id="panel_002", panel_id="panel_002", order=2, text="Second panel narration"),
        ],
    )
    (tmp_path / "narration.json").write_text(narration.model_dump_json(indent=2) + "\n", encoding="utf-8")

    audio = AudioArtifact(
        meta=Meta(schema_version=1, generator="test", config_hash="x", input_hashes={}),
        voice="test",
        entries=[
            AudioEntry(entry_id="panel_001", path="panel_001.wav", duration_seconds=1.5, words=[]),
            AudioEntry(entry_id="panel_002", path="panel_002.wav", duration_seconds=2.0, words=[]),
        ],
    )
    (tmp_path / "audio.json").write_text(audio.model_dump_json(indent=2) + "\n", encoding="utf-8")

    # build original timeline
    tl = TimelineArtifact(
        meta=Meta(schema_version=1, generator="test", config_hash="x", input_hashes={}),
        width=1080, height=1920, fps=30,
        gap_seconds=0.35, min_display_seconds=2.0,
        entries=[
            {"panel_id": "panel_001", "order": 1, "source_image": str(panels_dir / "panel_001.png"),
             "bbox": BBox(x=0, y=0, w=800, h=1200).model_dump(),
             "start_seconds": 0.0, "duration_seconds": 3.0,
             "audio_path": str(audio_dir / "panel_001.wav"),
             "pan": {"kind": "static", "scaled_w": 1080, "scaled_h": 1920, "travel_px": 0}},
            {"panel_id": "panel_002", "order": 2, "source_image": str(panels_dir / "panel_002.png"),
             "bbox": BBox(x=0, y=1200, w=800, h=1200).model_dump(),
             "start_seconds": 3.0, "duration_seconds": 3.5,
             "audio_path": str(audio_dir / "panel_002.wav"),
             "pan": {"kind": "static", "scaled_w": 1080, "scaled_h": 1920, "travel_px": 0}},
        ],
    )
    (tmp_path / "timeline.json").write_text(tl.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (tmp_path / "panels.json").write_text(json.dumps({"source": "x.png", "width": 800, "height": 2400, "plan_hash": "x", "config": {}, "panels": []}), "utf-8")

    # create editor project with reorder + duration change + zoom effect
    proj = EditorProject(
        session="test",
        original_timeline=[e.model_dump() for e in tl.entries],
        edited_timeline=[
            dict(tl.entries[1].model_dump(), order=1, start_seconds=0.0, duration_seconds=2.5),
            dict(tl.entries[0].model_dump(), order=2, start_seconds=2.5, duration_seconds=4.0),
        ],
        captions=[
            {"id": "cap_001", "panel_id": "panel_002", "text": "Hello world",
             "start_seconds": 0.0, "end_seconds": 2.5,
             "automated_text": "Hello world", "automated_start_seconds": 0.0, "automated_end_seconds": 2.5},
            {"id": "cap_002", "panel_id": "panel_001", "text": "Second panel",
             "start_seconds": 2.5, "end_seconds": 6.5,
             "automated_text": "Second panel", "automated_start_seconds": 2.5, "automated_end_seconds": 6.5},
        ],
        transitions=[
            {"from_panel_id": "panel_002", "to_panel_id": "panel_001", "type": "cut", "duration": 0.0},
        ],
        effects=[
            {"panel_id": "panel_002", "kind": "pan_down", "duration": 2.5},
            {"panel_id": "panel_001", "kind": "static", "duration": 4.0},
        ],
    )
    editor_path = tmp_path / "editor.json"
    Editor(proj).save(editor_path)

    out_mp4 = tmp_path / "recap_edited.mp4"
    cfg = VideoConfig(tts="none")
    result = render_edited_project(editor_path, out_mp4, cfg)

    assert out_mp4.is_file()
    assert out_mp4.stat().st_size > 10_000
    assert result["panels"] == 2
    assert result["srt_cues"] == 2
    assert (tmp_path / "recap_edited.srt").is_file()
    srt = (tmp_path / "recap_edited.srt").read_text("utf-8")
    assert "Hello world" in srt
    assert "Second panel" in srt


def test_render_edited_project_landscape_panel_with_pan(tmp_path):
    """Repro: pad filter fails when scaled image is wider/taller than canvas."""
    panels_dir = tmp_path / "panels"
    panels_dir.mkdir()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    _make_panel(panels_dir / "panel_001.png", color=(100, 150, 200), size=(800, 1200))

    wav_path = audio_dir / "panel_001.wav"
    import wave
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"\x00\x00" * 48000)

    narration = NarrationArtifact(
        meta=Meta(schema_version=1, generator="test", config_hash="x", input_hashes={}),
        mode="narrator",
        entries=[NarrationEntry(id="panel_001", panel_id="panel_001", order=1, text="Narration")],
    )
    (tmp_path / "narration.json").write_text(narration.model_dump_json(indent=2) + "\n", encoding="utf-8")

    audio = AudioArtifact(
        meta=Meta(schema_version=1, generator="test", config_hash="x", input_hashes={}),
        voice="test",
        entries=[AudioEntry(entry_id="panel_001", path="panel_001.wav", duration_seconds=1.5, words=[])],
    )
    (tmp_path / "audio.json").write_text(audio.model_dump_json(indent=2) + "\n", encoding="utf-8")

    tl = TimelineArtifact(
        meta=Meta(schema_version=1, generator="test", config_hash="x", input_hashes={}),
        width=1080, height=1920, fps=30,
        gap_seconds=0.35, min_display_seconds=2.0,
        entries=[
            {"panel_id": "panel_001", "order": 1, "source_image": str(panels_dir / "panel_001.png"),
             "bbox": BBox(x=0, y=0, w=800, h=1200).model_dump(),
             "start_seconds": 0.0, "duration_seconds": 3.0,
             "audio_path": str(audio_dir / "panel_001.wav"),
             "pan": {"kind": "pan_down", "scaled_w": 2000, "scaled_h": 1000, "travel_px": 0}},
        ],
    )
    (tmp_path / "timeline.json").write_text(tl.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (tmp_path / "panels.json").write_text(json.dumps({"source": "x.png", "width": 800, "height": 1200, "plan_hash": "x", "config": {}, "panels": []}), "utf-8")

    proj = EditorProject(
        session="test",
        original_timeline=[e.model_dump() for e in tl.entries],
        edited_timeline=[dict(tl.entries[0].model_dump())],
        captions=[],
        transitions=[],
        effects=[{"panel_id": "panel_001", "kind": "pan_down", "duration": 3.0}],
    )
    editor_path = tmp_path / "editor.json"
    Editor(proj).save(editor_path)

    out_mp4 = tmp_path / "recap_edited.mp4"
    cfg = VideoConfig(tts="none")
    result = render_edited_project(editor_path, out_mp4, cfg)

    assert out_mp4.is_file()
    assert out_mp4.stat().st_size > 10_000
    assert result["panels"] == 1
