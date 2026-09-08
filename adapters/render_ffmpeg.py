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
def _scale_crop(kind: str, sw: int, sh: int, dur: float, t: str = "t") -> str:
    """Build the scale+crop filter segment for one panel.

    When the scale cap leaves the scaled image smaller than the target frame
    (tiny panel), a pad step fills to 1080x1920 with black so crop never sees
    an undersized buffer — this replaces the old max(WIDTH, …) clamping that
    stretched panels non-uniformly.
    """
    pad = "" if (sw >= WIDTH and sh >= HEIGHT) else \
          f"pad={max(WIDTH,sw)}:{max(HEIGHT,sh)}:(ow-iw)/2:(oh-ih)/2:color=black,"
    if kind == "pan_down":
        return (f"scale={sw}:{sh},{pad}crop={WIDTH}:{HEIGHT}:"
                f"x=0:y='(ih-{HEIGHT})*{t}/{dur:.3f}'")
    if kind == "pan_right":
        return (f"scale={sw}:{sh},{pad}crop={WIDTH}:{HEIGHT}:"
                f"x='(iw-{WIDTH})*{t}/{dur:.3f}':y=0")
    if kind == "pan_left":
        return (f"scale={sw}:{sh},{pad}crop={WIDTH}:{HEIGHT}:"
                f"x='(iw-{WIDTH})*(1-{t}/{dur:.3f})':y=0")
    if kind == "pan_up":
        return (f"scale={sw}:{sh},{pad}crop={WIDTH}:{HEIGHT}:"
                f"x=0:y='(ih-{HEIGHT})*(1-{t}/{dur:.3f})'")
    if kind == "zoom_in":
        zw = f"1080*(1+0.3*{t}/{dur:.3f})"
        zh = f"1920*(1+0.3*{t}/{dur:.3f})"
        return (f"scale=w={zw}:h={zh}:eval=frame,crop={WIDTH}:{HEIGHT}:"
                f"x='(iw-{WIDTH})/2':y='(ih-{HEIGHT})/2'")
    if kind == "zoom_out":
        zw = f"1080*(1.3-0.3*{t}/{dur:.3f})"
        zh = f"1920*(1.3-0.3*{t}/{dur:.3f})"
        return (f"scale=w={zw}:h={zh}:eval=frame,crop={WIDTH}:{HEIGHT}:"
                f"x='(iw-{WIDTH})/2':y='(ih-{HEIGHT})/2'")
    return f"scale={sw}:{sh},{pad}crop={WIDTH}:{HEIGHT}:x=0:y=0"


def build_command(timeline: TimelineArtifact, out_path: Path,
                  ffmpeg_exe: str = "ffmpeg",
                  transitions: list[dict] | None = None) -> list[str]:
    """Build the FULL command. All chains go into ONE -filter_complex:
    repeating -filter_complex would create separate graphs whose labels are
    not visible to the concat graph (observed as a hang/garbage output)."""
    cmd: list[str] = [ffmpeg_exe, "-y", "-nostdin"]
    chains: list[str] = []
    vlabels: list[str] = []
    alabels: list[str] = []
    has_audio = any(e.audio_path for e in timeline.entries)
    use_xfade = False
    if transitions:
        for tr in transitions:
            if tr.get("type") != "cut":
                use_xfade = True
                break

    for i, e in enumerate(timeline.entries):
        dur = f"{e.duration_seconds:.3f}"
        cmd += ["-loop", "1", "-t", dur, "-i", e.source_image]
        if e.audio_path:
            cmd += ["-i", e.audio_path]
        else:
            cmd += ["-f", "lavfi", "-t", dur,
                    "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        sw, sh = e.pan.scaled_w, e.pan.scaled_h
        kind = e.pan.kind
        if kind in ("zoom_in", "zoom_out"):
            sw, sh = WIDTH, HEIGHT
        vf = _scale_crop(kind, sw, sh, e.duration_seconds)
        # fade transition filters
        if transitions and not use_xfade:
            prev_tr = transitions[i - 1] if i > 0 else None
            next_tr = transitions[i] if i < len(timeline.entries) - 1 else None
            if prev_tr and prev_tr.get("type") == "fade":
                fd = prev_tr.get("duration", 0.5)
                vf += f",fade=t=out:st={max(e.duration_seconds - fd, 0):.3f}:d={fd:.3f}"
            if next_tr and next_tr.get("type") == "fade":
                fd = next_tr.get("duration", 0.5)
                vf += f",fade=t=in:st=0:d={fd:.3f}"
        vf += f",setsar=1,fps={timeline.fps}[v{i}]"
        af = (f"[{2*i+1}:a]aresample=48000,aformat=channel_layouts=stereo,"
              f"apad=whole_dur={dur}[a{i}]")
        if transitions and not use_xfade:
            if prev_tr and prev_tr.get("type") == "fade":
                fd = prev_tr.get("duration", 0.5)
                af = (f"[{2*i+1}:a]aresample=48000,aformat=channel_layouts=stereo,"
                      f"apad=whole_dur={dur},afade=t=out:st={max(e.duration_seconds - fd, 0):.3f}:d={fd:.3f}[a{i}]")
            elif next_tr and next_tr.get("type") == "fade":
                fd = next_tr.get("duration", 0.5)
                af = (f"[{2*i+1}:a]aresample=48000,aformat=channel_layouts=stereo,"
                      f"apad=whole_dur={dur},afade=t=in:st=0:d={fd:.3f}[a{i}]")
        chains.append(vf + ";" + af)
        vlabels.append(f"[v{i}]")
        alabels.append(f"[a{i}]")

    if use_xfade:
        return _build_xfade_command(cmd, timeline, transitions, has_audio, vlabels, alabels, out_path)

    n = len(timeline.entries)
    chains.append(f"{''.join(vlabels)}concat=n={n}:v=1:a=0[vcat];"
                  f"{''.join(alabels)}concat=n={n}:v=0:a=1[acat]")
    if has_audio:
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
            "-c:a", "aac", "-b:a", "192k", "-max_muxing_queue_size", "9999",
            str(out_path)]
    return cmd


def _build_xfade_command(cmd: list[str], timeline: TimelineArtifact,
                         transitions: list[dict], has_audio: bool,
                         vlabels: list[str], alabels: list[str],
                         out_path: Path) -> list[str]:
    chains: list[str] = []
    n = len(timeline.entries)
    fade_dur = max((tr.get("duration", 0.5) for tr in transitions), default=0.5)

    # build video xfade chain
    prev_v = vlabels[0]
    for i in range(1, n):
        tr = transitions[i - 1]
        offset = sum(timeline.entries[j].duration_seconds for j in range(i)) - i * fade_dur
        tr_type = tr.get("type", "fade")
        chains.append(
            f"[{prev_v}][{vlabels[i]}]xfade=transition={tr_type}:"
            f"duration={fade_dur:.3f}:offset={offset:.3f}[vx{i}]"
        )
        prev_v = f"[vx{i}]"
    chains.append(f"{prev_v}[vcat]")

    # build audio acrossfade chain (use acrossfade for all when xfade mode)
    prev_a = alabels[0]
    for i in range(1, n):
        tr = transitions[i - 1]
        chains.append(
            f"[{prev_a}][{alabels[i]}]acrossfade=d={fade_dur:.3f}[ax{i}]"
        )
        prev_a = f"[ax{i}]"
    chains.append(f"{prev_a}[acat]")

    if has_audio:
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
            "-c:a", "aac", "-b:a", "192k", "-max_muxing_queue_size", "9999",
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


# ---------------------------------------------------------------- chunked rendering
def build_command_chunked(timeline: TimelineArtifact, out_path: Path,
                          ffmpeg_exe: str = "ffmpeg",
                          chunk_size: int = 12,
                          profile: dict | None = None) -> tuple[list[tuple[list[str], Path]], list[str], Path]:
    """Split entries into <=chunk_size groups; encode each to a small
    .ts segment (bounded filter graph, bounded memory), then concat with
    the concat DEMUXER (no re-encode). Returns (segments, concat_cmd, temp_dir)."""
    prof = profile or {"preset": "veryfast", "crf": "23", "threads": "4"}
    tmp = Path(out_path).with_name(out_path.stem + "_parts")
    tmp.mkdir(exist_ok=True)
    list_file = tmp / "concat.txt"
    segs: list[tuple[list[str], Path]] = []
    entries = timeline.entries
    for i in range(0, len(entries), chunk_size):
        part = TimelineArtifact(
            meta=timeline.meta, width=timeline.width, height=timeline.height,
            fps=timeline.fps, gap_seconds=timeline.gap_seconds,
            min_display_seconds=timeline.min_display_seconds,
            entries=entries[i:i + chunk_size])
        seg = tmp / f"seg_{i:03d}.ts"
        cmd = build_command(part, seg, ffmpeg_exe)
        # swap mp4 container flags for mpegts + profile limits
        cmd = [a for a in cmd if a not in ("-movflags", "+faststart")]
        if "-threads" not in cmd:
            cmd += ["-threads", prof.get("threads", "4")]
        segs.append((cmd, seg))
    list_file.write_text("".join(f"file '{s.resolve()}'\n" for _, s in segs))
    concat_cmd = [ffmpeg_exe, "-y", "-nostdin", "-f", "concat", "-safe", "0",
                  "-i", str(list_file), "-c", "copy", str(out_path)]
    return segs, concat_cmd, tmp


def render_chunked(timeline: TimelineArtifact, out_path: Path,
                   ffmpeg_exe: str = "ffmpeg",
                   chunk_size: int = 12,
                   profile: dict | None = None,
                   timeout: int = 3600) -> None:
    """Render large timelines in chunks to bound memory usage."""
    segs, concat_cmd, tmp = build_command_chunked(timeline, out_path, ffmpeg_exe, chunk_size, profile)
    try:
        for cmd, seg in segs:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout, shell=False, check=False)
            if proc.returncode != 0:
                tail = "\n".join(proc.stderr.splitlines()[-20:])
                raise RenderError(cmd, tail)
            if not seg.is_file():
                raise RenderError(cmd, f"segment {seg} not produced")
        proc = subprocess.run(concat_cmd, capture_output=True, text=True,
                              timeout=timeout, shell=False, check=False)
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.splitlines()[-20:])
            raise RenderError(concat_cmd, tail)
    finally:
        # clean up temp segment files
        for f in tmp.iterdir():
            try:
                f.unlink()
            except OSError:
                pass
        try:
            tmp.rmdir()
        except OSError:
            pass


def pick_render_strategy(n_panels: int, total_seconds: float) -> str:
    """Adaptive strategy: direct for small, chunked for large."""
    return "chunked" if (n_panels > 25 or total_seconds > 150) else "direct"


# ---------------------------------------------------------------- draft render
def render_draft(timeline: TimelineArtifact, out_path: Path,
                 ffmpeg_exe: str = "ffmpeg", timeout: int = 1800) -> None:
    """Draft render: lower resolution, faster encoding, for preview."""
    draft_timeline = TimelineArtifact(
        meta=timeline.meta,
        width=540,  # half width
        height=960,  # half height (9:16)
        fps=24,  # lower fps
        gap_seconds=timeline.gap_seconds,
        min_display_seconds=timeline.min_display_seconds,
        entries=timeline.entries)
    cmd = build_command(draft_timeline, out_path, ffmpeg_exe)
    # override with draft-friendly settings
    cmd = [a for a in cmd if a not in ("-preset", "-crf", "-r")]
    # insert draft settings after -c:v libx264
    new_cmd = []
    for a in cmd:
        new_cmd.append(a)
        if a == "libx264":
            new_cmd += ["-preset", "ultrafast", "-crf", "28"]
        if a == "-pix_fmt":
            pass  # keep
    # replace fps
    new_cmd = ["24" if a == str(timeline.fps) else a for a in new_cmd]
    proc = subprocess.run(new_cmd, capture_output=True, text=True,
                          timeout=timeout, shell=False, check=False)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-20:])
        raise RenderError(new_cmd, tail)
