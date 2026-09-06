# adapters/render_ffmpeg.py
"""Pure-FFmpeg renderer (renderer implementation 1 of 2).

Output: 1080x1920, H.264 (libx264, CRF 20, veryfast), AAC 192k, constant
30 fps, yuv420p. Audio normalized with loudnorm I=-16:TP=-1.5:LRA=11 in
single-pass mode (two-pass is more accurate but needs a second full run; the
recap audio is TTS-only and near-constant loudness, so single-pass drift is
negligible for this use case).
Timeline contract: entries are CONTIGUOUS; each entry's duration_seconds
INCLUDES its trailing silence gap, and its audio is padded to that exact
duration with apad=whole_dur. Durations come from probed files upstream,
never from estimated speech rate — this is what prevents A/V drift and
keeps the two renderers bit-comparable.

Filter facts verified against FFmpeg source, 2026-09-05:
- vf_crop.c: x/y accept per-frame expressions; variable 't' (seconds) IS
  available; size options are w/out_w and h/out_h; x/y default to centred.
- af_apad.c: 'whole_dur' pads a stream with silence up to a target duration.
- af_loudnorm.c: options I (-70..-5, default -24), LRA (1..50, default 7),
  TP (-9..0, default -2), print_format=JSON, measured_* options exist for
  two-pass mode.
Never centre-crops to 16:9: panels are scaled so that BOTH scaled dims cover
the 1080x1920 frame (overflow axis gets the pan; exact-fit gets static).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .schemas import TimelineArtifact

WIDTH, HEIGHT = 1080, 1920


class RenderError(RuntimeError):
    def __init__(self, cmd: list[str], stderr_tail: str):
        self.cmd, self.stderr_tail = cmd, stderr_tail
        super().__init__(f"ffmpeg failed (last stderr lines):\n{stderr_tail}")


def build_command(timeline: TimelineArtifact, out_path: Path,
                  ffmpeg_exe: str = "ffmpeg") -> list[str]:
    """Build the FULL command. All chains go into ONE -filter_complex:
    repeating -filter_complex would create separate graphs whose labels are
    not visible to the concat graph (observed as a hang/garbage output)."""
    cmd: list[str] = [ffmpeg_exe, "-y", "-nostdin"]
    chains: list[str] = []
    vlabels: list[str] = []
    alabels: list[str] = []
    for i, e in enumerate(timeline.entries):
        dur = f"{e.duration_seconds:.3f}"
        cmd += ["-loop", "1", "-t", dur, "-i", e.source_image]
        if e.audio_path:
            cmd += ["-i", e.audio_path]
        else:  # silent panel: explicit silence, same length
            cmd += ["-f", "lavfi", "-t", dur,
                    "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        sw, sh = e.pan.scaled_w, e.pan.scaled_h
        crop = {  # verified: 't' is a valid crop x/y expression variable
            "pan_down": f"crop={WIDTH}:{HEIGHT}:x=0:y='(ih-{HEIGHT})*t/{e.duration_seconds:.3f}'",
            "pan_right": f"crop={WIDTH}:{HEIGHT}:x='(iw-{WIDTH})*t/{e.duration_seconds:.3f}':y=0",
            "static": f"crop={WIDTH}:{HEIGHT}:x=0:y=0",
        }[e.pan.kind]
        chains.append(
            f"[{2*i}:v]scale={sw}:{sh},{crop},setsar=1,fps={timeline.fps}[v{i}];"
            f"[{2*i+1}:a]aresample=48000,aformat=channel_layouts=stereo,"
            f"apad=whole_dur={dur}[a{i}]")
        vlabels.append(f"[v{i}]")
        alabels.append(f"[a{i}]")
    n = len(timeline.entries)
    chains.append(f"{''.join(vlabels)}concat=n={n}:v=1:a=0[vcat];"
                  f"{''.join(alabels)}concat=n={n}:v=0:a=1[acat]")
    if any(e.audio_path for e in timeline.entries):
        # loudnorm cannot normalize pure silence (EBU integrated loudness of
        # silence is -inf); only apply when at least one entry has audio.
        chains.append("[acat]loudnorm=I=-16:TP=-1.5:LRA=11,"
                      "aresample=48000[aout]")
        alabel_out = "[aout]"
    else:
        chains.append("[acat]aresample=48000[aout]")
        alabel_out = "[aout]"
    cmd += ["-filter_complex", ";".join(chains),
            "-map", "[vcat]", "-map", alabel_out,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", str(timeline.fps),
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            str(out_path)]
    return cmd


def render(timeline: TimelineArtifact, out_path: Path,
           ffmpeg_exe: str = "ffmpeg", timeout: int = 3600) -> None:
    cmd = build_command(timeline, out_path, ffmpeg_exe)
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, shell=False,  # NEVER shell=True
                          check=False)  # returncode handled explicitly below
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-20:])
        raise RenderError(cmd, tail)
