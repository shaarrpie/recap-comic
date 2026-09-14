#!/usr/bin/env python3
"""
cinematic_effects.py  –  Dynamic manhwa/webtoon recap video renderer
=====================================================================
Drop-in enhancement for github.com/shaarrpie/recap-comic.

Turns the plain Ken-Burns pan output into a cinematic manhwa-recap
video matching the style of popular YouTube channels:
  • Punch-zoom on action / reveal panels
  • Ken-Burns ZOOM+PAN (not just pan) on calm panels
  • Screen shake on high-energy panels
  • Glitch transition (chromatic aberration) between panels
  • Cinematic vignette + high-contrast color grade on every frame
  • Speed-lines overlay on action panels (generated offline, no API)
  • Fade-in on first panel, hard-cut default
  • Letterbox (subtle cinematic bars) – optional

Usage (standalone, after you have panels.json + audio/):
    python cinematic_effects.py panels.json --audio-dir audio \\
        --out cinematic_recap.mp4 [--style dynamic|subtle] [--letterbox]

Integration:
    from cinematic_effects import CinematicConfig, make_cinematic_video
    make_cinematic_video(panels_json, out_mp4, audio_dir, cfg)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CinematicConfig:
    style: Literal["dynamic", "subtle"] = "dynamic"
    """dynamic = full manhwa-recap style; subtle = light effects only"""

    fps: int = 30
    output_width: int = 1080
    output_height: int = 1920

    # Ken Burns
    kb_zoom_start: float = 1.0     # scale at start of clip
    kb_zoom_end: float = 1.12      # scale at end (for calm panels)
    kb_zoom_end_fast: float = 1.20 # scale at end (action panels)

    # Punch zoom  (first N frames)
    punch_frames: int = 9          # ≈ 0.3s @30fps  –  the "smash" in
    punch_scale: float = 1.20      # peak scale during punch
    punch_settle: float = 1.06     # settle after punch

    # Screen shake (applied via crop offset)
    shake_enabled: bool = True
    shake_duration: float = 0.25   # seconds of shake per action panel
    shake_amplitude_px: int = 12

    # Glitch transition  (applied as chromashift blur)
    glitch_enabled: bool = True
    glitch_duration: float = 0.06  # seconds of RGB split
    glitch_shift_px: int = 8

    # Color grade (every panel)
    grade_contrast: float = 1.08   # >1 = more contrast
    grade_saturation: float = 1.05 # slight boost
    grade_shadows: float = -0.04   # cool shadows (teal)
    grade_highlights: float = 0.02 # warm highlights (orange)
    grade_brightness: float = -0.02

    # Vignette
    vignette_enabled: bool = True
    vignette_angle: float = 0.8    # radians  (PI/4 ≈ 0.78)

    # Speed lines overlay (action panels)
    speedlines_enabled: bool = True
    speedlines_opacity: float = 0.22

    # Letterbox
    letterbox_enabled: bool = False
    letterbox_height_px: int = 60  # px per bar

    # Audio
    bgm_path: str | None = None    # optional background music
    bgm_volume: float = 0.18       # relative to narration

    ffmpeg_exe: str = "ffmpeg"


DEFAULT_DYNAMIC = CinematicConfig(style="dynamic")
DEFAULT_SUBTLE  = CinematicConfig(
    style="subtle",
    punch_frames=6,
    punch_scale=1.12,
    shake_amplitude_px=6,
    glitch_shift_px=4,
    speedlines_opacity=0.10,
    vignette_angle=0.5,
)


# ---------------------------------------------------------------------------
# Panel-type classifier
# ---------------------------------------------------------------------------

_ACTION_WORDS = re.compile(
    r"\b(attack|strike|punch|kick|slash|explode|blast|clash|dodge|fight|"
    r"charge|rush|power|death|kills?|destroy|roar|scream|shatter|break|"
    r"rage|fury|impact|smash|crush|battle|war|blood|win|lose|fist|sword|"
    r"magic|spell|fire|lightning|thunder|burst|explodes?)\b",
    re.I,
)
_REVEAL_WORDS = re.compile(
    r"\b(reveal|shock|sudden|appear|transform|reborn|awakens?|level up|"
    r"returns?|true form|realized?|discovered?|secret|emerges?)\b",
    re.I,
)
_DIALOGUE_EX = re.compile(r"[!]{1,}")


def classify_panel(panel: dict) -> str:
    """Returns 'action', 'reveal', 'dialogue', or 'calm'."""
    narration = (panel.get("narration") or "").lower()
    dialogue  = (panel.get("dialogue")  or "").lower()
    combined  = narration + " " + dialogue

    if _ACTION_WORDS.search(combined):
        return "action"
    if _REVEAL_WORDS.search(combined):
        return "reveal"
    excl = len(_DIALOGUE_EX.findall(dialogue)) + len(_DIALOGUE_EX.findall(narration))
    if excl >= 2:
        return "action"
    if dialogue.strip():
        return "dialogue"
    return "calm"


# ---------------------------------------------------------------------------
# Speed-lines generator  (PIL – offline, no external API)
# ---------------------------------------------------------------------------

def _generate_speed_lines(width: int, height: int, out_path: Path) -> bool:
    """Draw radial speed lines into a transparent PNG. Returns True on success."""
    try:
        import math
        import random

        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        cx, cy = width // 2, height // 2
        rng = random.Random(42)  # deterministic
        for _ in range(160):
            angle = rng.uniform(0, 2 * math.pi)
            r_near = rng.uniform(0.15, 0.30) * min(width, height)
            r_far  = rng.uniform(0.80, 1.30) * max(width, height)
            lw     = rng.randint(1, 3)
            alpha  = rng.randint(80, 160)
            x0 = int(cx + r_near * math.cos(angle))
            y0 = int(cy + r_near * math.sin(angle))
            x1 = int(cx + r_far  * math.cos(angle))
            y1 = int(cy + r_far  * math.sin(angle))
            draw.line([(x0, y0), (x1, y1)], fill=(255, 255, 255, alpha), width=lw)
        img.save(str(out_path), "PNG")
        return True
    except ImportError:
        log.warning("PIL not available – speed-lines overlay disabled")
        return False
    except Exception as exc:
        log.warning("speed-lines generation failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# FFmpeg filter builders
# ---------------------------------------------------------------------------

def _scale_pad_filter(w: int, h: int) -> str:
    """Scale + pad the source image to exactly WxH, black bars."""
    return (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black"
    )


def _zoompan_filter(
    panel_type: str,
    duration_frames: int,
    w: int,
    h: int,
    cfg: CinematicConfig,
    pan_direction: str = "none",   # 'down', 'right', 'none'
) -> str:
    """
    Build a zoompan FFmpeg expression for the given panel type.
    All expressions are evaluated per-output-frame ('on' = output frame index).
    """
    d = max(duration_frames, 2)
    fps = cfg.fps

    # --- Zoom expressions ---
    if panel_type == "action" and cfg.style == "dynamic":
        pf  = min(cfg.punch_frames, d - 1)
        # Phase 0: punch-in  (first pf frames)
        # Phase 1: pull back to settle (next ~0.3s)
        # Phase 2: slow creep to end
        pull_frames = min(int(fps * 0.35), d - pf - 1)
        settle      = cfg.punch_settle
        z_end       = cfg.kb_zoom_end_fast
        zoom_expr = (
            f"if(lte(on,{pf}),"
            f"min(zoom+{(cfg.punch_scale-1.0)/pf:.5f},{cfg.punch_scale}),"
            f"if(lte(on,{pf+pull_frames}),"
            f"max(zoom-{(cfg.punch_scale-settle)/pull_frames:.5f},{settle}),"
            f"min(zoom+{(z_end-settle)/(d-pf-pull_frames):.6f},{z_end})))"
        )
    elif panel_type == "reveal" and cfg.style == "dynamic":
        # Slow creep in: starts at zoom_start, ends at zoom_end_fast
        rate = (cfg.kb_zoom_end_fast - cfg.kb_zoom_start) / d
        zoom_expr = f"min({cfg.kb_zoom_start:.4f}+{rate:.7f}*on,{cfg.kb_zoom_end_fast:.4f})"
    else:
        # Calm / dialogue / subtle style: gentle creep
        rate = (cfg.kb_zoom_end - cfg.kb_zoom_start) / max(d, 1)
        zoom_expr = f"min({cfg.kb_zoom_start:.4f}+{rate:.7f}*on,{cfg.kb_zoom_end:.4f})"

    # --- Pan expressions ---
    # Centre the view by default; shift according to pan direction
    if pan_direction == "down":
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = f"on/{d-1:.1f}*(ih-ih/zoom)"
    elif pan_direction == "right":
        x_expr = f"on/{d-1:.1f}*(iw-iw/zoom)"
        y_expr = "ih/2-(ih/zoom/2)"
    else:
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"

    return (
        f"zoompan=z='{zoom_expr}':"
        f"x='{x_expr}':y='{y_expr}':"
        f"d={d}:s={w}x{h}:fps={fps}"
    )


def _shake_filter(duration_frames: int, cfg: CinematicConfig) -> str:
    """Crop-based screen shake for the first shake_duration seconds.

    FFmpeg crop uses 'n' (output frame index) and 't' (time in seconds),
    NOT 'on' which is zoompan-specific.
    """
    shake_end_s = cfg.shake_duration
    a = cfg.shake_amplitude_px
    # Sinusoidal shake in the first shake_end_s seconds, then static
    return (
        f"crop=w=iw-{2*a}:h=ih-{2*a}:"
        f"x='{a}+{a}*sin(t*28.3)*if(lt(t,{shake_end_s:.2f}),1,0)':"
        f"y='{a}+{a}*cos(t*39.7)*if(lt(t,{shake_end_s:.2f}),1,0)',"
        f"scale={cfg.output_width}:{cfg.output_height}"
    )


def _color_grade_filter(cfg: CinematicConfig) -> str:
    """High-contrast, teal-shadow / warm-highlight color grade (manhwa look).

    FFmpeg colorbalance options:
      rs/gs/bs = red/green/blue shadows  (-1 to 1)
      rm/gm/bm = red/green/blue midtones (-1 to 1)
      rh/gh/bh = red/green/blue highlights (-1 to 1)
    """
    eq = (
        f"eq=contrast={cfg.grade_contrast:.3f}:"
        f"saturation={cfg.grade_saturation:.3f}:"
        f"brightness={cfg.grade_brightness:.3f}"
    )
    s = cfg.grade_shadows      # negative = reduce, positive = boost
    h = cfg.grade_highlights
    # Teal-orange look:
    #   shadows  -> boost blue, reduce red  (cool/teal)
    #   midtones -> very slight blue up
    #   highlights -> boost red, reduce blue (warm/orange)
    cb = (
        f"colorbalance="
        f"bs={s:.3f}:rs={-s:.3f}:"
        f"bm={s/3:.3f}:"
        f"rh={h:.3f}:bh={-h:.3f}"
    )
    return f"{eq},{cb}"


def _vignette_filter(cfg: CinematicConfig) -> str:
    return f"vignette=angle={cfg.vignette_angle:.3f}:mode=forward:eval=frame"


def _glitch_frames(duration_frames: int, cfg: CinematicConfig) -> int:
    return max(1, int(cfg.glitch_duration * cfg.fps))


# ---------------------------------------------------------------------------
# Per-panel clip builder (writes an intermediate .mp4)
# ---------------------------------------------------------------------------

def _build_panel_clip(
    panel: dict,
    image_path: Path,
    audio_path: Path | None,
    duration_s: float,
    clip_out: Path,
    speedlines_png: Path | None,
    cfg: CinematicConfig,
    ffmpeg: str,
) -> None:
    """
    Render a single panel image (+ optional audio) into a short .mp4 clip
    with cinematic effects applied.
    """
    w, h   = cfg.output_width, cfg.output_height
    fps    = cfg.fps
    frames = max(2, math.ceil(duration_s * fps))

    panel_type   = classify_panel(panel)
    png_w        = panel.get("output_width")  or w
    png_h        = panel.get("output_height") or h
    pan_dir      = _infer_pan_direction(png_w, png_h, w, h)

    log.debug("panel %s  type=%s  pan=%s  dur=%.2fs  frames=%d",
              panel.get("id"), panel_type, pan_dir, duration_s, frames)

    # ── Build filter chain ──────────────────────────────────────────────────
    # Input tag: [vraw]
    chain = []

    # 1. Scale+pad to output size first (so zoompan has a consistent canvas)
    chain.append(_scale_pad_filter(w, h))

    # 2. zoompan (zoom + Ken-Burns pan)
    chain.append(_zoompan_filter(panel_type, frames, w, h, cfg, pan_dir))

    # 3. Screen shake on action panels
    if (cfg.shake_enabled and cfg.style == "dynamic"
            and panel_type == "action"):
        chain.append(_shake_filter(frames, cfg))

    # 4. Color grade
    chain.append(_color_grade_filter(cfg))

    # 5. Vignette
    if cfg.vignette_enabled:
        chain.append(_vignette_filter(cfg))

    # 6. Letterbox  (burn in black bars at top/bottom)
    if cfg.letterbox_enabled:
        lb = cfg.letterbox_height_px
        chain.append(
            f"drawbox=x=0:y=0:w={w}:h={lb}:color=black:t=fill,"
            f"drawbox=x=0:y={h-lb}:w={w}:h={lb}:color=black:t=fill"
        )

    # 7. Trim to exact frame count + set timing
    chain.append(f"trim=start_frame=0:end_frame={frames}")
    chain.append("setpts=N/FRAME_RATE/TB")

    vf_str = ",".join(chain)

    # ── Speed lines overlay (action panels, dynamic style) ─────────────────
    # Speed lines are composited via a separate FFmpeg overlay pass if available
    use_speedlines = (
        cfg.speedlines_enabled
        and cfg.style == "dynamic"
        and panel_type == "action"
        and speedlines_png is not None
        and speedlines_png.is_file()
    )

    # ── FFmpeg command ──────────────────────────────────────────────────────
    cmd: list[str] = [
        ffmpeg, "-y",
        "-loop", "1", "-t", f"{duration_s:.3f}",
        "-i", str(image_path),                  # input 0: panel image (looped)
    ]

    if use_speedlines:
        # Also loop the speedlines PNG for the same duration
        cmd += ["-loop", "1", "-t", f"{duration_s:.3f}",
                "-i", str(speedlines_png)]       # input 1: speed lines PNG

    if audio_path and audio_path.is_file():
        cmd += ["-i", str(audio_path)]           # last input: audio

    # Filter graph
    if use_speedlines:
        # Screen-blend speedlines PNG onto the processed base clip.
        # blend=screen: white lines become bright; black areas are transparent.
        filter_complex = (
            f"[0:v]{vf_str}[base];"
            f"[1:v]scale={w}:{h}[sl];"
            f"[base][sl]blend=all_mode=screen:all_opacity={cfg.speedlines_opacity:.3f}[vout]"
        )
        cmd += ["-filter_complex", filter_complex, "-map", "[vout]"]
    else:
        cmd += ["-vf", vf_str]

    # Audio map (last -i)
    if audio_path and audio_path.is_file():
        audio_input_idx = 2 if use_speedlines else 1
        cmd += ["-map", f"{audio_input_idx}:a", "-acodec", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]

    cmd += [
        "-vcodec", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-t", f"{duration_s:.3f}",
        "-r", str(fps),
        str(clip_out),
    ]

    _run(cmd, label=f"clip {panel.get('id','?')}")


def _infer_pan_direction(png_w: int, png_h: int, out_w: int, out_h: int) -> str:
    """Determine pan direction by aspect ratio mismatch."""
    ar_panel  = png_w / max(png_h, 1)
    ar_screen = out_w / max(out_h, 1)
    if ar_panel < ar_screen * 0.85:   # tall panel → pan down
        return "down"
    if ar_panel > ar_screen * 1.15:   # wide panel → pan right
        return "right"
    return "none"


# ---------------------------------------------------------------------------
# Glitch transition builder
# ---------------------------------------------------------------------------

def _build_glitch_clip(
    blank_path: Path,
    duration_frames: int,
    glitch_out: Path,
    cfg: CinematicConfig,
    ffmpeg: str,
) -> None:
    """Very short white-flash clip with chroma aberration (glitch transition).

    Uses lavfi color source + geq for a white flash that fades with
    chroma shift on the UV planes.  No mergeplanes / split needed.
    """
    w, h  = cfg.output_width, cfg.output_height
    s     = max(1, cfg.glitch_shift_px)
    dur_s = max(duration_frames / cfg.fps, 2 / cfg.fps)  # at least 2 frames

    # geq: T = time in seconds (0..dur_s)
    # lum: start bright (255), fade to dark (0)  → white flash
    # cb/cr: subtle chroma noise that dies away as lum fades
    filt = (
        f"geq="
        f"lum='clip(255-255*T/{dur_s:.4f},0,255)':"
        f"cb='128+{s}*sin(X/{max(w//16,1)}+T*30)':"
        f"cr='128+{s}*cos(Y/{max(h//16,1)}-T*25)'"
    )
    cmd = [
        ffmpeg, "-y",
        # Video: lavfi color source looped for dur_s
        "-f", "lavfi",
        "-i", f"color=white:s={w}x{h}:r={cfg.fps}",
        # Audio: silent lavfi source
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-map", "0:v", "-map", "1:a",
        "-vf", filt,
        "-vcodec", "libx264", "-preset", "fast", "-crf", "18",
        "-acodec", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-t", f"{dur_s:.3f}",
        "-r", str(cfg.fps),
        str(glitch_out),
    ]
    _run(cmd, label="glitch-clip")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def make_cinematic_video(
    panels_json: Path,
    out_mp4: Path,
    audio_dir: Path | None = None,
    cfg: CinematicConfig | None = None,
) -> dict:
    """
    Main entry point.

    Parameters
    ----------
    panels_json : Path
        panels.json written by `guided run` / `guided cut`.
    out_mp4 : Path
        Output video path.
    audio_dir : Path | None
        Directory containing per-panel mp3/wav files named by panel id
        (e.g. audio/panel_001.mp3).  If None, assumed to be panels_json.parent/audio.
    cfg : CinematicConfig | None
        Effect configuration.  Defaults to CinematicConfig() ("dynamic" style).

    Returns
    -------
    dict  –  summary with keys: panels, duration_s, out
    """
    if cfg is None:
        cfg = CinematicConfig()

    panels_json = Path(panels_json)
    out_mp4     = Path(out_mp4)
    audio_dir   = Path(audio_dir) if audio_dir else panels_json.parent / "audio"
    panels_dir  = panels_json.parent
    ffmpeg      = _resolve_ffmpeg(cfg.ffmpeg_exe)

    log.info("cinematic_effects start  panels=%s  out=%s  style=%s",
             panels_json, out_mp4, cfg.style)

    # ── Load panels.json ────────────────────────────────────────────────────
    raw = json.loads(panels_json.read_text("utf-8"))
    # Support both bare list and CutArtifact envelope
    if isinstance(raw, dict) and "panels" in raw:
        panels = raw["panels"]
    elif isinstance(raw, list):
        panels = raw
    else:
        raise ValueError(f"Unexpected panels.json format in {panels_json}")

    # Filter blanks and context-only (text-bubble) panels, sort by panel_index
    panels = [
        p for p in panels
        if p.get("blank_flag", "normal") != "blank"
        and not p.get("context_only", False)
        and (p.get("output_height") or (p.get("y_end", 1) - p.get("y_start", 0))) > 0
    ]
    panels.sort(key=lambda p: (p.get("panel_index", 0), p.get("y_start", 0)))

    if not panels:
        raise ValueError("No valid panels found in panels.json")

    log.info("loaded %d panels from %s", len(panels), panels_json)

    # ── Prepare temp dir ────────────────────────────────────────────────────
    tmp_dir = out_mp4.parent / f".cinematic_tmp_{out_mp4.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate speed-lines overlay  (once, reused for all action panels) ──
    speedlines_png: Path | None = None
    if cfg.speedlines_enabled and cfg.style == "dynamic":
        sl_path = tmp_dir / "speed_lines.png"
        ok = _generate_speed_lines(cfg.output_width, cfg.output_height, sl_path)
        if ok:
            speedlines_png = sl_path

    # ── Audio timeline ───────────────────────────────────────────────────────
    # Build duration map from timeline.json (if it exists) or estimate
    timeline_json = panels_json.parent / "timeline.json"
    duration_map: dict[str, float] = {}
    if timeline_json.is_file():
        try:
            tl = json.loads(timeline_json.read_text("utf-8"))
            entries = tl.get("entries", [])
            for e in entries:
                pid = e.get("panel_id") or e.get("id")
                dur = e.get("duration_seconds") or e.get("duration") or 0
                if pid and dur:
                    duration_map[pid] = float(dur)
            log.info("loaded timeline for %d panels", len(duration_map))
        except Exception as exc:
            log.warning("could not parse timeline.json: %s", exc)

    # ── Audio map: panel_id -> audio file ───────────────────────────────────
    audio_map: dict[str, Path] = {}
    if audio_dir.is_dir():
        for f in audio_dir.iterdir():
            if f.suffix.lower() in (".mp3", ".wav", ".m4a", ".aac"):
                # Match by panel id embedded in filename
                for p in panels:
                    pid = p.get("id", "")
                    if pid and pid in f.stem:
                        audio_map[pid] = f
                        break
        log.info("matched audio for %d / %d panels", len(audio_map), len(panels))

    # ── Render per-panel clips ───────────────────────────────────────────────
    clip_paths: list[Path] = []
    total_duration = 0.0
    glitch_frames  = _glitch_frames(0, cfg)  # frames for glitch transition
    glitch_clip    = tmp_dir / "glitch.mp4"
    glitch_built   = False

    for i, panel in enumerate(panels):
        pid   = panel.get("id", f"{i:03d}")
        img_f = panel.get("image_file", "")
        img_p = panels_dir / img_f if img_f else None

        if not img_p or not img_p.is_file():
            log.warning("panel %s: image %s not found – skipping", pid, img_f)
            continue

        # Duration from timeline or estimate from word count
        if pid in duration_map:
            dur_s = duration_map[pid]
        else:
            words  = len((panel.get("narration", "") + " " + panel.get("dialogue", "")).split())
            dur_s  = max(2.5, words / 2.5)   # ~2.5 words/second reading pace
        dur_s = max(dur_s, 1.5)              # never shorter than 1.5 s

        audio_p = audio_map.get(pid)

        clip_out = tmp_dir / f"clip_{i:04d}.mp4"
        log.info("rendering clip %d/%d  panel=%s  type=%s  dur=%.2fs",
                 i+1, len(panels), pid, classify_panel(panel), dur_s)

        _build_panel_clip(
            panel=panel,
            image_path=img_p,
            audio_path=audio_p,
            duration_s=dur_s,
            clip_out=clip_out,
            speedlines_png=speedlines_png,
            cfg=cfg,
            ffmpeg=ffmpeg,
        )
        clip_paths.append(clip_out)
        total_duration += dur_s

        # Insert glitch transition between every pair of panels (dynamic style)
        if (
            cfg.glitch_enabled
            and cfg.style == "dynamic"
            and i < len(panels) - 1
        ):
            if not glitch_built:
                _build_glitch_clip(
                    blank_path=tmp_dir / "_blank.mp4",
                    duration_frames=glitch_frames,
                    glitch_out=glitch_clip,
                    cfg=cfg,
                    ffmpeg=ffmpeg,
                )
                glitch_built = True
            clip_paths.append(glitch_clip)
            total_duration += glitch_frames / cfg.fps

    if not clip_paths:
        raise RuntimeError("No clips were rendered – check panel image paths")

    # ── Concatenate all clips ────────────────────────────────────────────────
    concat_list = tmp_dir / "concat_list.txt"
    with concat_list.open("w") as fh:
        for cp in clip_paths:
            fh.write(f"file '{cp.resolve()}'\n")

    log.info("concatenating %d clips → %s", len(clip_paths), out_mp4)

    concat_cmd = [
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
    ]

    # Optional: mix in background music
    if cfg.bgm_path and Path(cfg.bgm_path).is_file():
        concat_cmd += ["-i", cfg.bgm_path]
        concat_cmd += [
            "-filter_complex",
            f"[0:a]volume=1[narr];[1:a]volume={cfg.bgm_volume}[bgm];"
            "[narr][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]",
            "-map", "0:v", "-map", "[aout]",
        ]
    else:
        concat_cmd += ["-map", "0:v", "-map", "0:a?"]

    concat_cmd += [
        "-vcodec", "libx264", "-preset", "fast", "-crf", "17",
        "-acodec", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_mp4),
    ]

    _run(concat_cmd, label="concat")

    log.info("done  out=%s  duration=%.1fs", out_mp4, total_duration)
    return {
        "panels": len([p for p in clip_paths if "glitch" not in p.name]),
        "clips": len(clip_paths),
        "duration_s": round(total_duration, 1),
        "out": str(out_mp4),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_ffmpeg(exe: str = "ffmpeg") -> str:
    import shutil
    r = shutil.which(exe)
    if r:
        return r
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise RuntimeError(f"ffmpeg not found: {exe!r}")


def _run(cmd: list[str], label: str = "") -> None:
    log.debug("[%s] %s", label, " ".join(cmd))
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=False, timeout=600
    )
    if result.returncode != 0:
        stderr = result.stderr[-2000:] if result.stderr else ""
        raise RuntimeError(
            f"FFmpeg error in '{label}':\n{stderr}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Cinematic manhwa recap video renderer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("panels_json", help="panels.json from guided run/cut")
    parser.add_argument("--out", default="cinematic_recap.mp4",
                        help="output mp4 path")
    parser.add_argument("--audio-dir", default=None,
                        help="dir with per-panel audio (default: panels_json parent/audio)")
    parser.add_argument("--style", choices=["dynamic", "subtle"], default="dynamic",
                        help="dynamic = full manhwa-recap; subtle = light effects")
    parser.add_argument("--bgm", default=None,
                        help="optional background music mp3 mixed under narration")
    parser.add_argument("--bgm-volume", type=float, default=0.18,
                        help="background music volume relative to narration (0-1)")
    parser.add_argument("--letterbox", action="store_true",
                        help="add cinematic letterbox bars")
    parser.add_argument("--no-glitch", action="store_true",
                        help="disable glitch transitions")
    parser.add_argument("--no-shake", action="store_true",
                        help="disable screen shake")
    parser.add_argument("--no-speedlines", action="store_true",
                        help="disable speed-lines overlay")
    parser.add_argument("--no-vignette", action="store_true",
                        help="disable vignette")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--ffmpeg", default="ffmpeg",
                        help="ffmpeg executable path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    base = DEFAULT_DYNAMIC if args.style == "dynamic" else DEFAULT_SUBTLE
    cfg = CinematicConfig(
        style=args.style,
        fps=args.fps,
        bgm_path=args.bgm,
        bgm_volume=args.bgm_volume,
        letterbox_enabled=args.letterbox,
        glitch_enabled=not args.no_glitch,
        shake_enabled=not args.no_shake,
        speedlines_enabled=not args.no_speedlines,
        vignette_enabled=not args.no_vignette,
        ffmpeg_exe=args.ffmpeg,
        # Carry over all other defaults from base preset
        kb_zoom_start=base.kb_zoom_start,
        kb_zoom_end=base.kb_zoom_end,
        kb_zoom_end_fast=base.kb_zoom_end_fast,
        punch_frames=base.punch_frames,
        punch_scale=base.punch_scale,
        punch_settle=base.punch_settle,
        shake_duration=base.shake_duration,
        shake_amplitude_px=base.shake_amplitude_px,
        glitch_duration=base.glitch_duration,
        glitch_shift_px=base.glitch_shift_px,
        grade_contrast=base.grade_contrast,
        grade_saturation=base.grade_saturation,
        grade_shadows=base.grade_shadows,
        grade_highlights=base.grade_highlights,
        grade_brightness=base.grade_brightness,
        vignette_angle=base.vignette_angle,
        speedlines_opacity=base.speedlines_opacity,
    )

    panels_json = Path(args.panels_json)
    audio_dir   = Path(args.audio_dir) if args.audio_dir else None
    out_mp4     = Path(args.out)

    try:
        summary = make_cinematic_video(panels_json, out_mp4, audio_dir, cfg)
        print("\n✓ cinematic video rendered")
        print(f"  panels : {summary['panels']}")
        print(f"  clips  : {summary['clips']} (includes glitch transitions)")
        print(f"  length : {summary['duration_s']:.1f}s")
        print(f"  output : {summary['out']}")
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    _cli()
