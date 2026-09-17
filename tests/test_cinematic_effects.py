# tests/test_cinematic_effects.py
"""Offline tests for cinematic_effects' ffmpeg command construction.

The per-panel clip builder must map BOTH the filtered video and the audio
stream. A lone "-map <audio>" makes ffmpeg drop every unmapped stream, so
non-action clips (no speedlines overlay) used to render audio-only --
39/50 clips of a real render had no video stream at all.
"""
from __future__ import annotations

from pathlib import Path

import cinematic_effects as ce


def _panel(pid: str, out_w: int, out_h: int, text: str = "") -> dict:
    return {
        "id": pid,
        "image_file": f"panel_{pid}.png",
        "output_width": out_w,
        "output_height": out_h,
        "narration": text,
        "dialogue": "",
    }


def _capture_cmd(monkeypatch, panel: dict, tmp_path: Path,
                 *, speedlines: bool, with_audio: bool) -> list[str]:
    captured: list[list[str]] = []

    def fake_run(cmd, label=""):
        captured.append(cmd)

    monkeypatch.setattr(ce, "_run", fake_run)

    img = tmp_path / panel["image_file"]
    img.write_bytes(b"not a real png")  # never decoded: _run is stubbed
    audio = None
    if with_audio:
        audio = tmp_path / f"{panel['id']}.mp3"
        audio.write_bytes(b"not a real mp3")
    sl = tmp_path / "speed_lines.png"
    if speedlines:
        sl.write_bytes(b"not a real png")

    ce._build_panel_clip(
        panel=panel,
        image_path=img,
        audio_path=audio,
        duration_s=5.0,
        clip_out=tmp_path / f"clip_{panel['id']}.mp4",
        speedlines_png=sl if speedlines else None,
        cfg=ce.CinematicConfig(),
        ffmpeg="ffmpeg",
    )
    assert len(captured) == 1
    return captured[0]


def test_clip_maps_video_without_speedlines(monkeypatch, tmp_path):
    """Plain panel (no speedlines): the -vf output MUST be mapped, or a lone
    audio -map drops the video stream entirely."""
    cmd = _capture_cmd(monkeypatch, _panel("001", 800, 5055, "calm narration"),
                       tmp_path, speedlines=False, with_audio=True)
    assert "-vf" in cmd
    # video mapped as well as audio (order: 0:v before 1:a)
    assert ("0:v" in cmd) and ("1:a" in cmd)
    assert cmd.index("0:v") < cmd.index("1:a")


def test_clip_maps_video_with_speedlines(monkeypatch, tmp_path):
    """Action panel (speedlines overlay): [vout] mapped via filter_complex."""
    cmd = _capture_cmd(monkeypatch, _panel("009", 800, 2200,
                                           "He attacks with a huge punch!"),
                       tmp_path, speedlines=True, with_audio=True)
    assert "-filter_complex" in cmd
    assert "[vout]" in cmd
    assert "2:a" in cmd


def test_clip_no_audio_maps_only_video(monkeypatch, tmp_path):
    """Silent panel: -an, no audio map; video still mapped."""
    cmd = _capture_cmd(monkeypatch, _panel("003", 800, 3000, "silent"),
                       tmp_path, speedlines=False, with_audio=False)
    assert "-an" in cmd
    assert "-map" not in cmd


def test_scale_fills_frame_not_pads(monkeypatch, tmp_path):
    """Cover-scale: the panel fills the canvas, no black-bar padding."""
    f = ce._scale_pad_filter(1080, 1920)
    assert "force_original_aspect_ratio=increase" in f
    assert "crop=1080:1920" in f
    assert "pad=" not in f
