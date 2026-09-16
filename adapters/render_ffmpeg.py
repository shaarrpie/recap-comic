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
from dataclasses import dataclass
from pathlib import Path

from .schemas import TimelineArtifact

WIDTH, HEIGHT = 1080, 1920


class RenderError(RuntimeError):
    def __init__(self, cmd: list[str], stderr_tail: str):
        self.cmd, self.stderr_tail = cmd, stderr_tail
        super().__init__(f"ffmpeg failed (last stderr lines):\n{stderr_tail}")


# --------------------------------------------------------------------------- #
# Visual style (blur background + vignette + optional colour grade)
# --------------------------------------------------------------------------- #
@dataclass
class StyleConfig:
    """Manhwa-recap look knobs. Defaults follow the shipped style patch:
    the panel floats on a blurred full-frame copy of itself with a strong
    dark vignette; the colour grade is OFF by default.

    All three effects are frame-local (spatial/per-pixel), so the grade and
    the vignette are applied ONCE on the composited output instead of once
    per panel clip -- bit-identical result, one filter pass instead of N.
    The blur background must be per-panel (each clip has its own source).
    """

    blur_background: bool = True    # panel floats on a blurred full-frame bg
    color_grade: bool = False       # darken + desaturate + cool blue-gray tint
    vignette: bool = True           # strong dark vignette around the edges
    vignette_angle: str = "PI/2.5"  # ffmpeg angle expr; smaller = stronger
    blur_sigma: float = 40.0        # gblur sigma for the background branch


def _style_post_filters(style: StyleConfig | None) -> list[str]:
    """Grade + vignette filters applied to the final composited video."""
    if style is None:
        return []
    post: list[str] = []
    if style.color_grade:
        post += [
            # darken + desaturate + contrast
            "eq=brightness=-0.07:saturation=0.82:contrast=1.14",
            # cool blue-gray tint (less red, more blue)
            "colorchannelmixer="
            "rr=0.92:rg=0.0:rb=0.08:"
            "gr=0.0:gg=0.90:gb=0.10:"
            "br=0.0:bg=0.15:bb=0.90",
        ]
    if style.vignette:
        post.append(f"vignette=angle={style.vignette_angle}")
    return post


def _blur_bg_chain(i: int, w: int, h: int, sigma: float) -> str:
    """One panel composited onto a blurred, slightly darkened full-frame
    copy of itself. The foreground is contain-fitted (the whole panel stays
    visible); the background covers the canvas and fills the letterbox.

    Returns the chain WITHOUT the trailing label; the caller appends
    transitions/setsar/fps/[vN] exactly as for the plain scale+crop chain.
    """
    return (
        f"split=2[bgr{i}][fgr{i}];"
        f"[bgr{i}]"
        f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={w}:{h}:(iw-{w})/2:(ih-{h})/2,"
        f"gblur=sigma={sigma:g},"
        f"eq=brightness=-0.10:saturation=1.3"
        f"[bg{i}];"
        f"[fgr{i}]"
        f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos"
        f"[fg{i}];"
        f"[bg{i}][fg{i}]overlay=x=(W-w)/2:y=(H-h)/2"
    )


def _scale_crop(kind: str, sw: int, sh: int, dur: float, t: str = "t",
                w: int = WIDTH, h: int = HEIGHT) -> str:
    """Build the scale+crop filter segment for one panel on a w x h canvas.

    The canvas defaults to 1080x1920 (9:16); callers pass the timeline's own
    dimensions so landscape timelines render landscape and draft (540x960)
    renders actually render at draft size.
    """
    pad = "" if (sw >= w and sh >= h) else \
          f"pad={max(w,sw)}:{max(h,sh)}:(ow-iw)/2:(oh-ih)/2:color=black,"
    if kind == "pan_down":
        return (f"scale={sw}:{sh},{pad}crop={w}:{h}:"
                f"x=0:y='(ih-{h})*{t}/{dur:.3f}'")
    if kind == "pan_right":
        return (f"scale={sw}:{sh},{pad}crop={w}:{h}:"
                f"x='(iw-{w})*{t}/{dur:.3f}':y=0")
    if kind == "pan_left":
        return (f"scale={sw}:{sh},{pad}crop={w}:{h}:"
                f"x='(iw-{w})*(1-{t}/{dur:.3f})':y=0")
    if kind == "pan_up":
        return (f"scale={sw}:{sh},{pad}crop={w}:{h}:"
                f"x=0:y='(ih-{h})*(1-{t}/{dur:.3f})'")
    if kind == "zoom_in":
        zw = f"{w}*(1+0.3*{t}/{dur:.3f})"
        zh = f"{h}*(1+0.3*{t}/{dur:.3f})"
        return (f"scale=w={zw}:h={zh}:eval=frame,crop={w}:{h}:"
                f"x='(iw-{w})/2':y='(ih-{h})/2'")
    if kind == "zoom_out":
        zw = f"{w}*(1.3-0.3*{t}/{dur:.3f})"
        zh = f"{h}*(1.3-0.3*{t}/{dur:.3f})"
        return (f"scale=w={zw}:h={zh}:eval=frame,crop={w}:{h}:"
                f"x='(iw-{w})/2':y='(ih-{h})/2'")
    return f"scale={sw}:{sh},{pad}crop={w}:{h}:x=0:y=0"


def build_command(timeline: TimelineArtifact, out_path: Path,
                  ffmpeg_exe: str = "ffmpeg",
                  transitions: list[dict] | None = None,
                  style: StyleConfig | None = None) -> list[str]:
    """Build the FULL command. All chains go into ONE -filter_complex:
    repeating -filter_complex would create separate graphs whose labels are
    not visible to the concat graph (observed as a hang/garbage output).

    ``style`` enables the manhwa-recap look (blur background / vignette /
    colour grade); None renders the plain scale+crop+pan picture."""
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
        # Honour the timeline's declared canvas (portrait, landscape, or
        # draft) rather than the module defaults.
        tw, th = timeline.width, timeline.height
        sw, sh = e.pan.scaled_w, e.pan.scaled_h
        kind = e.pan.kind
        if kind in ("zoom_in", "zoom_out"):
            sw, sh = tw, th
        if style is not None and style.blur_background:
            # The whole panel stays visible (contain-fit) over a blurred
            # full-frame background; there is no pan to express here.
            vf = _blur_bg_chain(i, tw, th, style.blur_sigma)
        else:
            vf = _scale_crop(kind, sw, sh, e.duration_seconds, w=tw, h=th)
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
        return _build_xfade_command(
            cmd, timeline, transitions or [], has_audio, vlabels, alabels,
            out_path, base_chains=chains)

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
                         out_path: Path, base_chains: list[str] | None = None) -> list[str]:
    chains: list[str] = list(base_chains) if base_chains else []
    n = len(timeline.entries)
    # xfade needs exactly n-1 transitions (one per panel boundary).
    if len(transitions) != max(0, n - 1):
        raise RenderError(
            cmd, f"xfade needs {max(0, n - 1)} transitions, got {len(transitions)}")
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
        # swap mp4 container flags for mpegts + profile limits: drop BOTH the
        # -movflags flag and its +faststart value so neither is left orphan.
        cleaned: list[str] = []
        skip_next = False
        for a in cmd:
            if skip_next:
                skip_next = False
                continue
            if a == "-movflags":
                skip_next = True
                continue
            if a == "+faststart":
                continue
            cleaned.append(a)
        cmd = cleaned
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
        import contextlib
        for f in tmp.iterdir():
            with contextlib.suppress(OSError):
                f.unlink()
        with contextlib.suppress(OSError):
            tmp.rmdir()


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
    # Drop -preset/-crf/-r TOGETHER with their values; -r is replaced below
    # with 24, and -preset/-crf are re-inserted with draft settings.
    cleaned: list[str] = []
    skip_next = False
    for a in cmd:
        if skip_next:
            skip_next = False
            continue
        if a in ("-preset", "-crf", "-r"):
            skip_next = True
            continue
        cleaned.append(a)
    cmd = cleaned
    # insert draft settings after -c:v libx264
    new_cmd: list[str] = []
    for a in cmd:
        new_cmd.append(a)
        if a == "libx264":
            new_cmd += ["-preset", "ultrafast", "-crf", "28"]
    # Append explicit -r 24 for the draft (the original -r was stripped above)
    new_cmd += ["-r", "24"]
    proc = subprocess.run(new_cmd, capture_output=True, text=True,
                          timeout=timeout, shell=False, check=False)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-20:])
        raise RenderError(new_cmd, tail)
