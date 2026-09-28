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

import re
import subprocess
import threading
from collections.abc import Callable
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
    # Ken-Burns push-in strength as a fraction of the panel's fitted size:
    # the foreground grows from its fitted size to (1 + zoom_strength)x over
    # the clip, so a 0.25 yields a 1.25x push-in by the last frame. Applies
    # to both the blur-background foreground and the plain zoom_in/zoom_out
    # kinds. 0 disables the animation (static fitted frame). Kept small so
    # zoom animations stay slow and cinematic.
    zoom_strength: float = 0.25


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


def _blur_bg_chain(i: int, w: int, h: int, sigma: float,
                   zoom: float = 0.0, dur: float = 1.0,
                   pan_x: float = 0.0, pan_y: float = 0.0,
                   kind: str | None = None,
                   zoom_mag: float = 0.35, pan_frac: float = 0.32,
                   columns: int = 0, split_dir: int = 1,
                   gap_frac: float = 0.05,
                   pan_overflow: float = 0.15,
                   split_ss: float = 3.0,
                   pan_travel_frac: float = 1.0,
                   pan_width_margin: float = 0.9,
                   fps: float = 30.0) -> str:
    """One panel composited onto a neutral blurred full-frame copy of itself.

    Background branch (always): cover-scaled to the canvas and centre-cropped
    BEFORE the blur, so the blurred copy fills the whole w x h frame edge to
    edge (never letterboxed) and is a true, un-colour-adjusted blur.

    Foreground branch has two modes:

    1. Cinematic, kind-driven Ken Burns (used when a motion preset supplies
       ``kind``). The edit-rotation cycle becomes actually visible because
       each kind now produces a DISTINCT, slow (full-clip) move:
         - zoom_in  : grow from fit-height to (1+zoom_mag)x
         - zoom_out : shrink from (1+zoom_mag)x to fit-height
         - pan_down : hold taller than the frame, drift top -> bottom
         - pan_up   : hold taller than the frame, drift bottom -> top
       For tall manhwa panels the foreground stays narrower than the canvas,
       so the blurred side pillars remain visible throughout.

    2. Legacy contain->cover push-in (``kind`` is None, e.g. no motion preset
       or a plain style render): the foreground starts contain-fitted (whole
       panel visible) and is pushed in until it covers the frame, reaching
       (cover x (1 + zoom)) on the last frame. zoom=0 (or dur<=0) reproduces
       the static contain-fit. pan_x/pan_y are TOTAL overlay travels in
       canvas px, interpolated smoothly as t/dur.

    The blurred background always fills the frame, so no move can reveal
    empty/black areas. Returns the chain WITHOUT the trailing label.
    """
    # Loop-hold: the caller feeds ONE source frame (no input -loop), so every
    # static filter (the full-frame gblur background, and a pan foreground of
    # constant size) is computed ONCE and then duplicated to n frames with a
    # clean PTS ramp. Recomputing an identical blur/scale on every frame is pure
    # waste; duplicating its result is bit-identical. n = the frame count the
    # old `-loop 1 -t dur` + `fps=` path produced, so output length is unchanged.
    n = max(1, int(round(dur * fps)))
    lp = f"loop=loop={n - 1}:size=1:start=0,setpts=N/{fps:g}/TB"
    bg = (f"[bgr{i}]"
          f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={w}:{h}:(iw-{w})/2:(ih-{h})/2,"
          f"gblur=sigma={sigma:g},{lp}"
          f"[bg{i}]")
    head = f"split=2[bgr{i}][fgr{i}];"
    centre = f"[bg{i}][fg{i}]overlay=x=(W-w)/2:y=(H-h)/2"

    # ---- super-tall panel: split into `columns` horizontal bands laid out
    # side by side directly on the blurred background (no hstack), so a gap
    # between the bands shows the blurred copy. Each band slowly pans
    # vertically; neighbouring bands move in OPPOSITE directions and the pair
    # flips per split panel via ``split_dir``. No zoom. Bands fill the frame
    # height; the whole panel reads ~columns x closer at once.
    #
    # Smoothness: the pan is a `crop` whose y ffmpeg rounds to whole pixels.
    # A slow pan (~27px/s) advances 0-or-1 px unevenly per frame -> visible
    # jitter. So we pan in a vertically SUPER-SAMPLED space (``split_ss`` x
    # taller), where one integer step is only 1/split_ss of a final pixel, then
    # Lanczos-downscale back to the frame height -- the downsample blends the
    # sub-pixel positions into smooth, jitter-free motion.
    if columns >= 2:
        ss = max(1.0, float(split_ss))
        vh = int(round(h * ss))                    # supersampled viewport height
        bh_big = int(round(h * (1.0 + pan_overflow) * ss))  # band height (ss x)
        ov_big = bh_big - vh                       # supersampled pan travel px
        gap = int(round(w * gap_frac))             # gap between bands (px)
        dt = dur if dur and dur > 0 else 1.0
        copies = "".join(f"[sl{k}_{i}]" for k in range(columns))
        # duplicate the single frame to n first, THEN split into bands so each
        # band's crop y(t) animates across the clip.
        chain = f"[fgr{i}]{lp},split={columns}{copies};"
        for k in range(columns):
            d = split_dir * (1 if k % 2 == 0 else -1)
            if d > 0:                              # pan down: top -> bottom
                yexpr = f"{ov_big:.3f}*t/{dt:.3f}"
            else:                                  # pan up: bottom -> top
                yexpr = f"{ov_big:.3f}-{ov_big:.3f}*t/{dt:.3f}"
            chain += (
                f"[sl{k}_{i}]crop=iw:trunc(ih/{columns}):0"
                f":trunc(ih*{k}/{columns})[sc{k}_{i}];"
                f"[sc{k}_{i}]scale=-2:{bh_big}:flags=lanczos[sd{k}_{i}];"
                f"[sd{k}_{i}]crop=iw:{vh}:0:'{yexpr}'[cd{k}_{i}];"
                f"[cd{k}_{i}]scale=-2:{h}:flags=lanczos[vp{k}_{i}];")
        prev = f"bg{i}"
        for k in range(columns):
            xk = (f"(W-({columns}*w+{(columns - 1) * gap}))/2"
                  f"+{k}*(w+{gap})")
            if k == columns - 1:
                chain += f"[{prev}][vp{k}_{i}]overlay=x='{xk}':y=0"
            else:
                nxt = f"o{k}_{i}"
                chain += (f"[{prev}][vp{k}_{i}]overlay=x='{xk}':y=0"
                          f"[{nxt}];")
                prev = nxt
        return f"{head}{bg};{chain}"

    # ---- cinematic, kind-driven Ken Burns (only when a motion preset set kind)
    if kind in ("zoom_in", "zoom_out", "pan_down", "pan_up",
                "pan_left", "pan_right") and dur > 0:
        zm = max(zoom_mag, float(zoom or 0.0))
        if kind in ("zoom_in", "zoom_out"):
            # zoom genuinely varies per frame -> duplicate first, then animate
            # the scale with eval=frame over the ramped timestamps.
            if kind == "zoom_in":
                zexpr = f"(({h}/ih)*(1+{zm:g}*t/{dur:.3f}))"
            else:
                zexpr = f"(({h}/ih)*({1.0 + zm:g}-{zm:g}*t/{dur:.3f}))"
            fg = (f"[fgr{i}]{lp},scale=w='iw*{zexpr}':h='ih*{zexpr}'"
                  f":eval=frame:flags=lanczos[fg{i}]")
            return f"{head}{bg};{fg};{centre}"
        # pan_down / pan_up: hold the foreground taller than the frame so the
        # camera drifts through the artwork; width stays contain (pillars).
        # pan_left / pan_right: same closer framing, but the whole panel slides
        # ACROSS the blurred backdrop -- it needs no horizontal overflow, so it
        # adds lateral variety without cropping any art away.
        #
        # Width clamp: the closer framing scales the foreground to
        # (h/ih)*(1+pan_frac), which is right for TALL art (its width stays
        # under the canvas so blur pillars show). But a short/near-canvas-aspect
        # panel scaled that far blows its width past the canvas too, so it
        # covers the whole frame and the blurred background vanishes. Cap the
        # scale so the displayed width never exceeds ``pan_width_margin`` of the
        # canvas: min(height-closer-framing, margin * w/iw). Tall panels never
        # bind (their w/iw is large); squat panels keep side blur and still
        # overflow vertically enough to pan. The overlay composites onto the
        # full-frame blurred bg, so any uncovered edge shows blur, never black.
        s = (f"min(({h}/ih)*(1+{pan_frac:g}),"
             f"{pan_width_margin:g}*{w}/iw)")
        # pan foreground is a CONSTANT size -> scale once, then duplicate; only
        # the overlay position animates.
        fg = (f"[fgr{i}]scale=w='iw*{s}':h='ih*{s}'"
              f":eval=init:flags=lanczos,{lp}[fg{i}]")
        # The resolver already applied its travel fraction, so pan_x/pan_y are
        # FINAL canvas px and must not be scaled a second time (double-scaling
        # turned a 154px slide into an imperceptible 54px drift). The
        # h*pan_frac*pan_travel_frac form is only the fallback for timelines
        # that carry no per-panel travel.
        if kind in ("pan_left", "pan_right"):
            travel = abs(pan_x) if pan_x else h * pan_frac * pan_travel_frac
            if kind == "pan_right":
                ox = f"(W-w)/2-{travel:.3f}+{2 * travel:.3f}*t/{dur:.3f}"
            else:
                ox = f"(W-w)/2+{travel:.3f}-{2 * travel:.3f}*t/{dur:.3f}"
            overlay = f"[bg{i}][fg{i}]overlay=x='{ox}':y=(H-h)/2"
            return f"{head}{bg};{fg};{overlay}"
        travel = abs(pan_y) if pan_y else h * pan_frac * pan_travel_frac
        if kind == "pan_down":
            oy = f"(H-h)/2+{travel:.3f}-{travel:.3f}*t/{dur:.3f}"
        else:
            oy = f"(H-h)/2-{travel:.3f}+{travel:.3f}*t/{dur:.3f}"
        overlay = f"[bg{i}][fg{i}]overlay=x=(W-w)/2:y='{oy}'"
        return f"{head}{bg};{fg};{overlay}"

    # ---- legacy contain->cover push-in (motion preset inactive)
    if zoom > 0 and dur > 0:
        # Per-frame scale factor: contain-fit at t=0 -> cover x (1+zoom) at
        # the end. iw/ih are the panel PNG dims; w/h the canvas literals.
        # Genuinely time-varying -> duplicate first, then animate the scale.
        contain = f"min({w}/iw,{h}/ih)"
        cover = f"max({w}/iw,{h}/ih)"
        zexpr = (f"({contain}+({cover}*(1+{zoom:g})-{contain})*t/{dur:.3f})")
        fg = (f"[fgr{i}]{lp},"
              f"scale=w='iw*{zexpr}':h='ih*{zexpr}':eval=frame:flags=lanczos"
              f"[fg{i}]")
    else:
        # static contain-fit: constant size -> scale once, then duplicate.
        fg = (f"[fgr{i}]"
              f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
              f"{lp}[fg{i}]")
    if (pan_x or pan_y) and dur > 0:
        overlay = (f"[bg{i}][fg{i}]overlay="
                   f"x='(W-w)/2+({pan_x:g})*t/{dur:.3f}':"
                   f"y='(H-h)/2+({pan_y:g})*t/{dur:.3f}'")
    else:
        overlay = centre
    return f"{head}{bg};{fg};{overlay}"


def _scale_crop(kind: str, sw: int, sh: int, dur: float, t: str = "t",
                w: int = WIDTH, h: int = HEIGHT,
                zoom: float = 0.3) -> str:
    """Build the scale+crop filter segment for one panel on a w x h canvas.

    The canvas defaults to 1080x1920 (9:16); callers pass the timeline's own
    dimensions so landscape timelines render landscape and draft (540x960)
    renders actually render at draft size. ``zoom`` is the push-in strength
    for the zoom_in/zoom_out kinds (0.3 = up to 1.3x).
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
        zw = f"{w}*(1+{zoom:g}*{t}/{dur:.3f})"
        zh = f"{h}*(1+{zoom:g}*{t}/{dur:.3f})"
        return (f"scale=w={zw}:h={zh}:eval=frame,crop={w}:{h}:"
                f"x='(iw-{w})/2':y='(ih-{h})/2'")
    if kind == "zoom_out":
        zw = f"{w}*(1+{zoom:g}-{zoom:g}*{t}/{dur:.3f})"
        zh = f"{h}*(1+{zoom:g}-{zoom:g}*{t}/{dur:.3f})"
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

    blur = style is not None and style.blur_background
    for i, e in enumerate(timeline.entries):
        dur = f"{e.duration_seconds:.3f}"
        if blur:
            # Feed ONE frame; _blur_bg_chain computes the static gblur/scale
            # ONCE and loop-holds to the frame count (bit-identical, ~2-3x less
            # filter work). Non-blur paths keep the input-level -loop.
            cmd += ["-framerate", str(timeline.fps), "-i", e.source_image]
        else:
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
            # Default look: contain-fit foreground pushed in over the clip.
            # With a motion preset the foreground additionally pans via the
            # overlay (per-panel zoom_strength + pan_x/y from timeline.motion;
            # fallback to the global style so old timelines render unchanged).
            # The blurred background always fills the frame, so overlay pans
            # can never reveal empty areas.
            motion = e.motion or {}
            panel_zoom = float(motion.get("zoom_strength",
                                          style.zoom_strength))
            pan_x = float(motion.get("pan_x_px", 0.0) or 0.0)
            pan_y = float(motion.get("pan_y_px", 0.0) or 0.0)
            # A motion preset supplies the pan kind; drive a distinct, slow
            # Ken-Burns move per shot (the edit-rotation cycle). Without a
            # preset, kind stays None and the legacy push-in is reproduced.
            ref_kind = kind if motion.get("preset") else None
            # Closer framing: the resolver reports the scale each shot is held
            # at (tall art AND normal art both get one now). The pan branch
            # holds the foreground at (h/ih)*(1+pan_frac) and drifts by
            # canvas*pan_frac*pan_travel_frac, so pan_frac = panel_scale - 1
            # traverses the new overflow, slowed by the travel fraction.
            # Panels with no reported scale keep the 0.32 default.
            pan_frac = 0.32
            try:
                _scale = motion.get("panel_scale")
                if _scale is not None:
                    pan_frac = max(0.0, float(_scale) - 1.0)
            except (TypeError, ValueError):
                pan_frac = 0.2
            try:
                travel_frac = float(motion.get("pan_travel_frac", 1.0) or 1.0)
            except (TypeError, ValueError):
                travel_frac = 1.0
            vf = _blur_bg_chain(i, tw, th, style.blur_sigma,
                                zoom=panel_zoom,
                                dur=e.duration_seconds,
                                pan_x=pan_x, pan_y=pan_y,
                                kind=ref_kind, pan_frac=pan_frac,
                                columns=int(motion.get("split_columns", 0)
                                            or 0),
                                split_dir=int(motion.get("split_dir", 1)
                                              or 1),
                                gap_frac=float(motion.get(
                                    "split_gap_frac", 0.05) or 0.05),
                                pan_overflow=float(motion.get(
                                    "split_pan_frac", 0.15) or 0.15),
                                split_ss=float(motion.get(
                                    "split_ss", 3.0) or 3.0),
                                pan_travel_frac=travel_frac,
                                fps=float(timeline.fps))
        else:
            motion = e.motion or {}
            default_zoom = style.zoom_strength if style else 0.3
            try:
                panel_zoom = float(motion.get("zoom_strength", default_zoom))
            except (TypeError, ValueError):
                panel_zoom = default_zoom
            vf = _scale_crop(kind, sw, sh, e.duration_seconds, w=tw, h=th,
                             zoom=panel_zoom)
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
            out_path, base_chains=chains, style=style)

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
    # Grade + vignette are frame-local: one pass over the composited video
    # is identical to one pass per clip (and to the xfade path below).
    post = _style_post_filters(style)
    if post:
        chains.append(f"[vcat]{','.join(post)}[vout]")
        vlabel_out = "[vout]"
    else:
        vlabel_out = "[vcat]"
    cmd += ["-filter_complex", ";".join(chains),
            "-map", vlabel_out, "-map", alabel_out,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", str(timeline.fps),
            "-c:a", "aac", "-b:a", "192k", "-max_muxing_queue_size", "9999",
            # Faststart in the encode itself: the moov atom is placed at the
            # head so web playback can start immediately. Doing it here
            # avoids a second full-file read+write copy pass afterwards.
            "-movflags", "+faststart",
            str(out_path)]
    return cmd


def _build_xfade_command(cmd: list[str], timeline: TimelineArtifact,
                         transitions: list[dict], has_audio: bool,
                         vlabels: list[str], alabels: list[str],
                         out_path: Path, base_chains: list[str] | None = None,
                         style: StyleConfig | None = None) -> list[str]:
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

    post = _style_post_filters(style)
    if post:
        chains.append(f"[vcat]{','.join(post)}[vout]")
        vlabel_out = "[vout]"
    else:
        vlabel_out = "[vcat]"
    cmd += ["-filter_complex", ";".join(chains),
            "-map", vlabel_out, "-map", alabel_out,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", str(timeline.fps),
            "-c:a", "aac", "-b:a", "192k", "-max_muxing_queue_size", "9999",
            "-movflags", "+faststart",
            str(out_path)]
    return cmd


# --------------------------------------------------------------------------- #
# Live render progress (ffmpeg streams "time=HH:MM:SS.xx" status lines)
# --------------------------------------------------------------------------- #
_FFMPEG_TIME_RE = re.compile(r"time=(\d+):(\d{2}):(\d{2}(?:[.,]\d+)?)")


def parse_ffmpeg_time(line: str) -> float | None:
    """Finished-seconds from an ffmpeg status line ('... time=00:01:23.45')."""
    m = _FFMPEG_TIME_RE.search(line or "")
    if not m:
        return None
    h, mi, s = m.groups()
    try:
        return int(h) * 3600 + int(mi) * 60 + float(s.replace(",", "."))
    except ValueError:
        return None


def timeline_seconds(timeline: TimelineArtifact) -> float:
    """Approximate finished duration; mirrors recap_video.total_seconds
    (duplicated so adapters never import upward out of the adapter layer)."""
    return round(sum(e.duration_seconds for e in timeline.entries), 3)


def _run_ffmpeg_progress(cmd: list[str], total_s: float,
                         progress_cb: Callable[[float, str], None] | None,
                         timeout: int) -> tuple[int, str]:
    """Run ffmpeg with streamed stderr, reporting fraction done to progress_cb.

    Only used on the progress path; plain subprocess.run stays for every
    other caller. A watchdog timer kills ffmpeg past `timeout` and re-raises
    subprocess.TimeoutExpired, because the stderr read loop cannot enforce
    one itself on a silent hang; error contract matches render()'s.
    """
    timed_out = False

    def _kill() -> None:
        nonlocal timed_out
        timed_out = True
        try:
            proc.kill()
        except OSError:
            pass

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True,
                            errors="replace", shell=False)
    watchdog = threading.Timer(max(1, timeout), _kill)
    watchdog.daemon = True
    watchdog.start()
    tail: list[str] = []
    try:
        assert proc.stderr is not None
        for line in proc.stderr:
            tail.append(line.rstrip("\n"))
            del tail[:-20]
            if total_s > 0 and progress_cb is not None:
                done_s = parse_ffmpeg_time(line)
                if done_s is not None:
                    progress_cb(min(1.0, done_s / total_s), "")
        rc = proc.wait(timeout=60)
    finally:
        watchdog.cancel()
    if timed_out:
        raise subprocess.TimeoutExpired(cmd, timeout)
    return rc, "\n".join(tail[-20:])


def render(timeline: TimelineArtifact, out_path: Path,
           ffmpeg_exe: str = "ffmpeg", timeout: int = 3600,
           style: StyleConfig | None = None,
           progress_cb: Callable[[float, str], None] | None = None) -> None:
    cmd = build_command(timeline, out_path, ffmpeg_exe, style=style)
    if progress_cb is not None:
        # Streaming variant so the caller sees live percentage; identical
        # contract (rc!=0 -> RenderError, timeout -> TimeoutExpired).
        rc, tail = _run_ffmpeg_progress(cmd, timeline_seconds(timeline),
                                        progress_cb, timeout)
        if rc != 0:
            raise RenderError(cmd, tail)
        return
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
                          profile: dict | None = None,
                          style: StyleConfig | None = None
                          ) -> tuple[list[tuple[list[str], Path]], list[str], Path]:
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
        cmd = build_command(part, seg, ffmpeg_exe, style=style)
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
                  "-i", str(list_file), "-c", "copy",
                  # Pin the video timescale to what a direct (non-chunked)
                  # mp4 render gets from ffmpeg (fps*512; 15360 at 30fps):
                  # mpegts segments otherwise carry a 90k tbn, and mixing
                  # timebases breaks any later -c copy concat of chapter
                  # mp4s (observed: video duration inflated ~7x vs audio).
                  "-video_track_timescale", str(timeline.fps * 512),
                  # moov at the head for immediate web playback; the per-segment
                  # .ts files are stream copies, so this is the only place it
                  # matters for the final mp4.
                  "-movflags", "+faststart",
                  str(out_path)]
    return segs, concat_cmd, tmp


def render_chunked(timeline: TimelineArtifact, out_path: Path,
                   ffmpeg_exe: str = "ffmpeg",
                   chunk_size: int = 12,
                   profile: dict | None = None,
                   timeout: int = 3600,
                   style: StyleConfig | None = None,
                   progress_cb: Callable[[float, str], None] | None = None
                   ) -> None:
    """Render large timelines in chunks to bound memory usage.

    With progress_cb the per-segment ffmpeg runs stream their status lines so
    the reported fraction is real (seconds done across ALL segments, plus a
    'segment k/n' label); without it behaviour is the original blocking one.
    """
    segs, concat_cmd, tmp = build_command_chunked(
        timeline, out_path, ffmpeg_exe, chunk_size, profile, style=style)
    entries = timeline.entries
    chunk_secs: list[float] = []
    if progress_cb is not None:
        for i in range(0, len(entries), chunk_size):
            chunk_secs.append(round(
                sum(e.duration_seconds for e in entries[i:i + chunk_size]),
                3))
    total_s = sum(chunk_secs) or 1.0
    done_s = 0.0
    try:
        for k, (cmd, seg) in enumerate(segs):
            if progress_cb is not None:
                part = chunk_secs[k] if k < len(chunk_secs) else 0.0
                msg = f"segment {k + 1}/{len(segs)}"
                rc, tail = _run_ffmpeg_progress(
                    cmd, part,
                    # `_ignored` absorbs the (always-empty) message the stream
                    # runner passes positionally; the segment label must win.
                    lambda f, _ignored=None, _m=msg, _p=part, _d=done_s, _t=total_s:
                    progress_cb(min(1.0, (_d + f * _p) / _t), _m),
                    timeout)
                if rc != 0:
                    raise RenderError(cmd, tail)
                done_s += part
            else:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=timeout, shell=False,
                                      check=False)
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
        if progress_cb is not None:
            progress_cb(1.0, "concatenating")
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
                 ffmpeg_exe: str = "ffmpeg", timeout: int = 1800,
                 style: StyleConfig | None = None) -> None:
    """Draft render: lower resolution, faster encoding, for preview."""
    draft_timeline = TimelineArtifact(
        meta=timeline.meta,
        width=540,  # half width
        height=960,  # half height (9:16)
        fps=24,  # lower fps
        gap_seconds=timeline.gap_seconds,
        min_display_seconds=timeline.min_display_seconds,
        entries=timeline.entries)
    cmd = build_command(draft_timeline, out_path, ffmpeg_exe, style=style)
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
