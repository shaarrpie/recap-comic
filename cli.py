# ruff: noqa: B008  # typer's Option()/Argument() calls in defaults ARE its supported API
# cli.py
"""CLI for the AI-guided panels & narration feature (manhwa long-strips).

Commands (group `guided`):
  guided plan  STRIP [--out-plan PLAN.json]   Phase 1 only: pre-read the
                       strip and print the per-panel narration plan. This is
                       the --dry-run contract of the feature request.
  guided cut   STRIP PLAN.json                 Phase 2 only: cut with an
                       existing plan JSON.
  guided run   STRIP                           Phase 1 + Phase 2;
                       --dry-run stops after Phase 1.

STRIP may be a PNG/JPG/WebP image or a CBZ/ZIP archive containing
panel images in reading order. Archives are extracted and stitched
into a single tall strip before processing.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import uuid
import zipfile
from pathlib import Path

import typer
from dotenv import load_dotenv
from PIL import Image

import guided_pipeline as gp
import strip_analyzer as sa
from adapters._logging import get_logger, setup_logging

_WRITE_ATOMIC = sa.write_atomic

_ARCHIVE_EXTS = {".zip", ".cbz"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _natural_sort_key(name: str) -> list[str | int]:
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", name)]


log = get_logger(__name__)


def _configure_logging(level: str) -> None:
    load_dotenv()
    setup_logging(level=os.environ.get("LOG_LEVEL", level))
    logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))


def _resolve_strip_path(strip: Path) -> Path:
    """If `strip` is a CBZ/ZIP archive, extract images and stitch them
    into a single tall PNG in a temp dir. Otherwise return `strip` as-is."""
    if strip.suffix.lower() not in _ARCHIVE_EXTS:
        return strip
    log.info("archive detected suffix=%s extracting=%s", strip.suffix, strip.name)
    with zipfile.ZipFile(strip, "r") as archive:
        names = sorted(
            (f for f in archive.namelist()
             if Path(f).suffix.lower() in _IMAGE_EXTS),
            key=_natural_sort_key,
        )
        if not names:
            raise typer.BadParameter(
                f"no images found in archive {strip.name}")
        images = []
        for name in names:
            with archive.open(name) as fh:
                images.append(Image.open(fh).convert("RGB"))
    total_h = sum(img.height for img in images)
    max_w = max(img.width for img in images)
    stitched = Image.new("RGB", (max_w, total_h))
    y = 0
    for img in images:
        if img.width < max_w:
            border = int(np.median(np.array(np.concatenate([
                np.array(img)[:8].ravel(), np.array(img)[-8:].ravel(),
                np.array(img)[:, :8].ravel(), np.array(img)[:, -8:].ravel()
            ]))))
            pad_img = Image.new("RGB", (max_w - img.width, img.height),
                                (border, border, border))
            stitched.paste(img, (0, y))
            stitched.paste(pad_img, (img.width, y))
        else:
            stitched.paste(img, (0, y))
        y += img.height
    tmp = Path(tempfile.gettempdir()) / f"recap-comic-{strip.stem}-{uuid.uuid4().hex[:8]}-stitched.png"
    stitched.save(tmp, "PNG")
    log.info("archive stitched images=%d out=%s", len(images), tmp)
    return tmp


app = typer.Typer(
    help="manhwa-recap: AI-guided panels & narration for long strips")
guided_app = typer.Typer(help="AI-guided panel segmentation & narration")
app.add_typer(guided_app, name="guided")


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "recap-comic"


_VALID_BACKENDS = {"gemini", "openai", "anthropic", "ollama", "local", "cloudflare", "fixture", "none"}
_VALID_TTS = {"edge", "kokoro", "none"}
_VALID_STYLES = {"recap", "literal"}


def _validate_backend(name: str) -> str:
    n = name.lower()
    if n not in _VALID_BACKENDS:
        raise typer.BadParameter(
            f"unknown backend {name!r}; choose from: {', '.join(sorted(_VALID_BACKENDS))}")
    if n == "local":
        n = "ollama"
    return n


def _validate_chunk_params(chunk_height: int, overlap: int) -> None:
    if overlap >= chunk_height:
        raise typer.BadParameter(
            f"--overlap ({overlap}) must be smaller than --chunk-height ({chunk_height})")
    if chunk_height < 100:
        raise typer.BadParameter(
            f"--chunk-height must be at least 100px, got {chunk_height}")


def _validate_tts(name: str) -> str:
    n = name.lower()
    if n not in _VALID_TTS:
        raise typer.BadParameter(
            f"unknown --tts {name!r}; choose from: {', '.join(sorted(_VALID_TTS))}")
    return n


def _validate_style(name: str) -> str:
    n = name.lower()
    if n not in _VALID_STYLES:
        raise typer.BadParameter(
            f"unknown --style {name!r}; choose from: {', '.join(sorted(_VALID_STYLES))}")
    return n


@guided_app.command("plan")
def guided_plan(
    strip: Path = typer.Argument(..., exists=True, dir_okay=False,
                                 help="tall strip image (PNG/JPG) or CBZ/ZIP archive"),
    out_plan: Path | None = typer.Option(
        None, "--out-plan", help="also write the plan JSON here"),
    backend: str = typer.Option(
        "gemini", "--backend",
        help="gemini|openai|anthropic|ollama|local|cloudflare|fixture|none"),
    model: str | None = typer.Option(
        None, "--model",
        help="vision model id (gemini defaults to gemini-2.5-flash; "
             "required for openai/anthropic/local)"),
    chunk_height: int = typer.Option(2000, "--chunk-height",
                                     help="reading-chunk height in px"),
    overlap: int = typer.Option(200, "--overlap",
                                help="chunk overlap in px (coordinate safe)"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
    debug_overlay: Path | None = typer.Option(
        None, "--debug-overlay",
        help="draw AI boundaries (red) + snapped gutters (green) + bubble "
             "boxes (blue) into this PNG"),
    chunk_dir: Path | None = typer.Option(
        None, "--chunk-dir",
        help="also save each chunk sent to the model as chunk_XX.png here "
             "(visual debug: what the model saw)"),
    force: bool = typer.Option(False, "--force"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Phase 1 only: pre-read the strip; print the panel-narration plan.

    This is the --dry-run behaviour: nothing is cut; only the optional
    --out-plan file (and the Phase-1 cache, if --cache-dir is given) is
    written. With --debug-overlay, also writes a visualization PNG showing
    every AI-proposed boundary (red), the gutter-snapped final boundary
    (green), panel IDs/confidence labels, and bubble boxes (blue). With
    --chunk-dir, also saves each chunk image the model received.
    """
    _configure_logging(log_level)
    strip = _resolve_strip_path(strip)
    backend = _validate_backend(backend)
    _validate_chunk_params(chunk_height, overlap)
    used_cache_dir = cache_dir or _default_cache_dir()
    log.info("guided_plan start strip=%s backend=%s model=%s chunk_height=%d overlap=%d",
             strip.name, backend, model, chunk_height, overlap)
    try:
        plan, _artifact, _used = gp.run_guided(
            strip, Path("."), backend_name=backend, model=model,
            chunk_height=chunk_height, overlap=overlap,
            cache_dir=used_cache_dir, chunk_dir=chunk_dir,
            out_plan=out_plan,
            force=force, dry_run=True)
    except (gp.VisionAnalysisError, FileNotFoundError, ValueError) as exc:
        if log.isEnabledFor(logging.DEBUG):
            log.exception("guided plan failed")
        else:
            typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:
        log.exception("unexpected error in guided plan")
        typer.echo(f"ERROR: unexpected error: {exc} "
                   "(see --log-level DEBUG for details)", err=True)
        raise typer.Exit(1) from exc
    if debug_overlay is not None:
        from debug_view import draw_overlay
        draw_overlay(strip, plan, out_path=debug_overlay)
    log.info("guided_plan complete panels=%d", len(plan.entries))
    typer.echo(plan.model_dump_json(indent=2))


@guided_app.command("cut")
def guided_cut(
    strip: Path = typer.Argument(..., exists=True, dir_okay=False,
                                 help="tall strip image or CBZ/ZIP archive"),
    plan: Path = typer.Argument(..., exists=True, dir_okay=False,
                                help="plan JSON from 'guided plan'"),
    out_dir: Path = typer.Option("guided_out", "--out-dir"),
    tolerance: int = typer.Option(
        80, "--tolerance",
        help="+/-px around each AI boundary to scan for a gutter"),
    max_panel_height: int = typer.Option(
        1600, "--max-panel-height",
        help="panels taller than this are split at internal gutters"),
    variance_threshold: float = typer.Option(6.0, "--variance-threshold"),
    report: Path | None = typer.Option(
        None, "--report",
        help="also write a self-contained HTML review page here"),
    force: bool = typer.Option(False, "--force"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Phase 2 only: cut the strip using an existing plan JSON."""
    _configure_logging(log_level)
    strip = _resolve_strip_path(strip)
    try:
        _plan, artifact, _used = gp.run_guided(
            strip, out_dir, backend_name="none", plan_path=plan,
            tolerance=tolerance, max_panel_height=max_panel_height,
            variance_threshold=variance_threshold, force=force,
            fallback=False)
        if artifact is None:
            raise RuntimeError("guided cut returned no artifact")
    except (gp.VisionAnalysisError, FileNotFoundError, ValueError) as exc:
        if log.isEnabledFor(logging.DEBUG):
            log.exception("guided cut failed")
        else:
            typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:
        log.exception("unexpected error in guided cut")
        typer.echo(f"ERROR: unexpected error: {exc} (see --log-level DEBUG for details)",
                    err=True)
        raise typer.Exit(1) from exc
    log.info("guided_cut complete panels=%d", len(artifact.panels))
    typer.echo(f"cut {len(artifact.panels)} panels into {out_dir}")
    typer.echo(f"sidecar: {Path(out_dir) / 'panels.json'}")
    if report is not None:
        from report import render_report
        render_report(artifact, report, out_dir)
        typer.echo(f"report: {report}")


@guided_app.command("run")
def guided_run(
    strip: Path = typer.Argument(..., exists=True, dir_okay=False,
                                 help="tall strip image or CBZ/ZIP archive"),
    out_dir: Path = typer.Option("guided_out", "--out-dir"),
    backend: str = typer.Option(
        "gemini", "--backend",
        help="gemini|openai|anthropic|ollama|local|cloudflare|fixture|none"),
    model: str | None = typer.Option(
        None, "--model",
        help="vision model id (gemini defaults to gemini-2.5-flash; "
             "required for openai/anthropic/local)"),
    plan_path: Path | None = typer.Option(
        None, "--plan", help="reuse an existing plan JSON from 'guided plan'"),
    chunk_height: int = typer.Option(2000, "--chunk-height"),
    overlap: int = typer.Option(200, "--overlap"),
    cache_dir: Path | None = typer.Option(
        None, "--cache-dir",
        help="Phase-1 cache dir (default: ~/.cache/recap-comic)"),
    out_plan: Path | None = typer.Option(
        None, "--out-plan",
        help="also write the final plan JSON here (always written to out-dir/plan.json)"),
    chunk_dir: Path | None = typer.Option(
        None, "--chunk-dir",
        help="also save each chunk sent to the model as chunk_XX.png here "
             "(visual debug: what the model saw)"),
    tolerance: int = typer.Option(80, "--tolerance",
        help="+/-px around each AI boundary to scan for gutter (default 80)"),
    max_panel_height: int = typer.Option(1600, "--max-panel-height"),
    variance_threshold: float = typer.Option(6.0, "--variance-threshold"),
    edge_threshold: float = typer.Option(30.0, "--edge-threshold"),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Phase 1 only: print the plan, do not cut anything"),
    fallback: bool = typer.Option(
        True, "--fallback/--no-fallback",
        help="fall back to the gutter detector on AI failure / low confidence"),
    force: bool = typer.Option(False, "--force"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Phase 1 (AI pre-read) + Phase 2 (guided dissection) in one command."""
    _configure_logging(log_level)
    strip = _resolve_strip_path(strip)
    backend = _validate_backend(backend)
    _validate_chunk_params(chunk_height, overlap)
    used_cache_dir = cache_dir or _default_cache_dir()
    log.info("guided_run start strip=%s backend=%s model=%s dry_run=%s",
             strip.name, backend, model, dry_run)
    try:
        plan, artifact, used = gp.run_guided(
            strip, out_dir, backend_name=backend, model=model,
            plan_path=plan_path, chunk_height=chunk_height, overlap=overlap,
            cache_dir=used_cache_dir, chunk_dir=chunk_dir, out_plan=out_plan,
            tolerance=tolerance,
            max_panel_height=max_panel_height,
            variance_threshold=variance_threshold,
            edge_threshold=edge_threshold, fallback=fallback,
            force=force, dry_run=dry_run)
    except (gp.VisionAnalysisError, FileNotFoundError, ValueError) as exc:
        if log.isEnabledFor(logging.DEBUG):
            log.exception("guided run failed")
        else:
            typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:
        log.exception("unexpected error in guided run")
        typer.echo(f"ERROR: unexpected error: {exc} (see --log-level DEBUG for details)",
                    err=True)
        raise typer.Exit(1) from exc
    if dry_run:
        log.info("guided_run dry_run complete")
        return
    log.info("guided_run complete panels=%d fallback=%s", len(artifact.panels), used)
    typer.echo(f"cut {len(artifact.panels)} panels into {out_dir}")
    typer.echo(f"sidecar: {Path(out_dir) / 'panels.json'}")


@guided_app.command("narrate")
def guided_narrate(
    plan: Path = typer.Argument(..., exists=True, dir_okay=False,
                                help="plan JSON from 'guided plan' or panels.json from 'guided cut'"),
    out: Path = typer.Option("narration.txt", "--out"),
    style: str = typer.Option(
        "recap", "--style", help="recap (flowing) | literal (verbatim)",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Produce a single narration script from a per-panel plan.

    Offline: no API call. 'recap' gives a flowing paragraph (TTS-friendly);
    'literal' gives the per-panel narrations verbatim, separated by blank
    lines. Also writes a sidecar <out>.index.json for TTS chunking.
    """
    _configure_logging(log_level)
    try:
        from narrator import narrate_plan
        script = narrate_plan(plan, out, style=style)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        if log.isEnabledFor(logging.DEBUG):
            log.exception("guided narrate failed")
        else:
            typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:
        log.exception("unexpected error in guided narrate")
        typer.echo(f"ERROR: unexpected error: {exc} "
                   "(see --log-level DEBUG for details)", err=True)
        raise typer.Exit(1) from exc
    if not script.strip():
        typer.echo("WARNING: narration is empty (fallback plan has no AI narration)", err=True)
    typer.echo(f"wrote {out} ({len(script)} chars)")
    typer.echo(f"index: {out.with_suffix('.index.json')}")


@guided_app.command("video")
def guided_video(
    panels: Path = typer.Argument(
        ..., exists=True, dir_okay=False,
        help="panels.json written by 'guided run' / 'guided cut'"),
    out: Path | None = typer.Option(
        None, "--out",
        help="output mp4 (default: <panels dir>/recap.mp4)"),
    tts: str = typer.Option(
        "edge", "--tts", help="edge (default, needs internet) | kokoro (offline) | none (silent)"),
    voice: str = typer.Option(
        "en-US-AriaNeural", "--voice",
        help="edge-tts / kokoro voice id"),
    rate: str = typer.Option("+0%", "--rate", help="speech rate, e.g. +10%"),
    pitch: str = typer.Option("+0Hz", "--pitch", help="speech pitch, e.g. -2Hz"),
    speed: float = typer.Option(1.0, "--speed", help="kokoro speed multiplier (edge-tts ignores this)"),
    kokoro_model_path: Path | None = typer.Option(
        None, "--kokoro-model-path",
        help="path to kokoro-v1.0.onnx (required for --tts kokoro)"),
    kokoro_voices_path: Path | None = typer.Option(
        None, "--kokoro-voices-path",
        help="path to voices-v1.0.bin (required for --tts kokoro)"),
    dialogue: bool = typer.Option(
        True, "--dialogue/--no-dialogue",
        help="also read each panel's dialogue after its narration"),
    gap: float = typer.Option(0.35, "--gap", help="silence after each panel (s)"),
    min_display: float = typer.Option(
        2.0, "--min-display", help="minimum seconds a panel stays on screen"),
    max_display: float = typer.Option(
        12.0, "--max-display", help="cap for SILENT panels (tts none)"),
    pan_speed: int = typer.Option(
        450, "--pan-speed", help="max pan speed in px/s (lower = slower)"),
    fps: int = typer.Option(30, "--fps"),
    ffmpeg: str = typer.Option("ffmpeg", "--ffmpeg", help="ffmpeg executable"),
    ffprobe: str = typer.Option("ffprobe", "--ffprobe", help="ffprobe executable"),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="build narration/audio/timeline/srt but do NOT render the mp4"),
    force: bool = typer.Option(False, "--force", help="ignore all caches"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Phase 3: panels.json -> recap.mp4 (9:16, narrated, captioned).

    Reads the per-panel narration, synthesises speech with edge-tts (or none),
    builds a drift-free timeline from MEASURED clip durations, pans each
    panel (Ken-Burns) and renders one mp4 with ffmpeg. Also writes recap.srt,
    timeline.json, audio/ and narration.json next to the mp4.
    """
    _configure_logging(log_level)
    from recap_video import VideoConfig, VideoError, make_recap_video

    tts = _validate_tts(tts)
    out_path = out or panels.parent / "recap.mp4"
    cfg = VideoConfig(
        tts=tts, voice=voice, rate=rate, pitch=pitch, speed=speed,
        include_dialogue=dialogue, gap_seconds=gap,
        min_display_seconds=min_display, max_display_seconds=max_display,
        max_pan_px_per_sec=pan_speed, fps=fps,
        ffmpeg_exe=ffmpeg, ffprobe_exe=ffprobe,
        kokoro_model_path=kokoro_model_path,
        kokoro_voices_path=kokoro_voices_path)
    try:
        summary = make_recap_video(panels, out_path, cfg,
                                   force=force, dry_run=dry_run)
    except (VideoError, FileNotFoundError, ValueError) as exc:
        if log.isEnabledFor(logging.DEBUG):
            log.exception("guided video failed")
        else:
            typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("unexpected error in guided video")
        typer.echo(f"ERROR: unexpected error: {exc} "
                   "(see --log-level DEBUG for details)", err=True)
        raise typer.Exit(1) from exc

    mins, secs = divmod(summary["total_seconds"], 60)
    typer.echo(f"panels: {summary['panels']}  spoken: {summary['spoken_panels']}"
               f"  voice: {summary['voice']}  length: {int(mins)}m{secs:04.1f}s")
    typer.echo(f"timeline: {summary['timeline']}")
    typer.echo(f"captions: {summary['srt']} ({summary['srt_cues']} cues)")
    if dry_run:
        typer.echo("dry run: mp4 not rendered")
    else:
        typer.echo(f"video: {summary['video']}")


if __name__ == "__main__":
    app()
