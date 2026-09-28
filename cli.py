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
  guided filter OUT_DIR                        Phase 2.5: remove blank
                        panels, demote text-only panels to context
                        (--apply overwrites panels.json).

STRIP may be a PNG/JPG/WebP image or a CBZ/ZIP archive containing
panel images in reading order. Archives are extracted and stitched
into a single tall strip before processing.
"""
from __future__ import annotations

import atexit
import contextlib
import io
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import typer
from dotenv import load_dotenv
from PIL import Image

import guided_pipeline as gp
import strip_analyzer as sa
from adapters._logging import get_logger, setup_logging
from guided_cutter import CutArtifact, CutPanel

_WRITE_ATOMIC = sa.write_atomic

_ARCHIVE_EXTS = {".zip", ".cbz"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Caps for stitched archives. A crafted or just huge CBZ used to load every
# image into RAM at once (images.append in the old code) and allocate the
# full stitched buffer before the later 80 MP guard ever ran — an OOM bomb.
_MAX_CBZ_ARCHIVE_BYTES = 200 * 1024 * 1024   # 200 MB
_MAX_CBZ_IMAGES = 600                          # panel count
_MAX_CBZ_IMAGE_PX = 60_000_000                 # 60 MP per page
_MAX_CBZ_STITCHED_PX = 80_000_000              # 80 MP, matches the later guard


def _natural_sort_key(name: str) -> list[str | int]:
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", name)]


log = get_logger(__name__)


def _force_utf8_console() -> None:
    """Manhwa dialogue/narration is Korean/Japanese; a cp1252 Windows
    console makes typer.echo raise UnicodeEncodeError mid-command (the
    plan JSON is printed AFTER it is safely on disk, so the run itself
    succeeded -- but the CLI exited non-zero). Reconfigure the console
    streams to UTF-8 whenever they are not already."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding in ("utf8", "none", ""):
            continue
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, OSError, ValueError):
            pass


def _configure_logging(level: str) -> None:
    _force_utf8_console()
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

        # First pass: get dimensions without fully decoding, and enforce caps.
        dimensions = []
        total_bytes = 0
        for name in names:
            with archive.open(name) as fh:
                data = fh.read()
            total_bytes += len(data)
            if total_bytes > _MAX_CBZ_ARCHIVE_BYTES:
                raise typer.BadParameter(
                    f"archive {strip.name} is too large to load into memory "
                    f"({total_bytes / 1024 / 1024:.0f} MB > "
                    f"{_MAX_CBZ_ARCHIVE_BYTES / 1024 / 1024:.0f} MB)")
            with Image.open(io.BytesIO(data)) as img:
                img.load()  # verify it's a valid image
                if img.width * img.height > _MAX_CBZ_IMAGE_PX:
                    raise typer.BadParameter(
                        f"image {name} in {strip.name} is too large "
                        f"({img.width}x{img.height} = "
                        f"{img.width * img.height / 1_000_000:.0f} MP > "
                        f"{_MAX_CBZ_IMAGE_PX / 1_000_000:.0f} MP)")
                dimensions.append((name, img.width, img.height))

        if len(dimensions) > _MAX_CBZ_IMAGES:
            raise typer.BadParameter(
                f"archive {strip.name} has too many images "
                f"({len(dimensions)} > {_MAX_CBZ_IMAGES})")

        total_h = sum(h for _, _, h in dimensions)
        max_w = max(w for _, w, _ in dimensions)
        if max_w * total_h > _MAX_CBZ_STITCHED_PX:
            raise typer.BadParameter(
                f"stitched strip {strip.name} would be too large to load "
                f"({max_w}x{total_h} = {max_w * total_h / 1_000_000:.0f} MP > "
                f"{_MAX_CBZ_STITCHED_PX / 1_000_000:.0f} MP)")
        stitched = Image.new("RGB", (max_w, total_h))
        y = 0

        # Second pass: decode and paste each image (the caps above bound
        # how much can ever be in RAM at once).
        for name, w, h in dimensions:
            with archive.open(name) as fh, Image.open(fh) as img:
                img.load()
                img_rgb = img.convert("RGB")
                if w < max_w:
                    border = int(np.median(np.array(np.concatenate([
                        np.array(img_rgb)[:8].ravel(), np.array(img_rgb)[-8:].ravel(),
                        np.array(img_rgb)[:, :8].ravel(), np.array(img_rgb)[:, -8:].ravel()
                    ]))))
                    pad_img = Image.new("RGB", (max_w - w, h),
                                        (border, border, border))
                    stitched.paste(img_rgb, (0, y))
                    stitched.paste(pad_img, (w, y))
                else:
                    stitched.paste(img_rgb, (0, y))
            y += h

    # Write the stitched strip into a per-run temp directory that atexit
# removes wholesale (the old single-file + atexit.unlink left the file
# behind on a hard kill, and a second archive in the same process leaked).
    tmp_dir = Path(tempfile.mkdtemp(
        prefix=f"recap-comic-{strip.stem}-", dir=tempfile.gettempdir()))
    tmp_path = tmp_dir / "stitched.png"
    stitched.save(tmp_path, "PNG")
    log.info("archive stitched images=%d out=%s", len(dimensions), tmp_path)

    atexit.register(_purge_stitched_temp, tmp_dir)

    return Path(tmp_path)


_STITCHED_TMP_RE = re.compile(r"^recap-comic-.+-(\d{10,})$")


def _purge_stitched_temp(tmp_dir: Path) -> None:
    """Best-effort removal of one stitched-archive temp dir."""
    with contextlib.suppress(OSError):
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _sweep_stitched_temp() -> None:
    """Drop stitched-archive temp dirs older than the run window.

    A hard kill or a crash used to leave a temp dir behind forever; this
    runs once at startup so they cannot accumulate.
    """
    tmp_root = Path(tempfile.gettempdir())
    now = time.time()
    for entry in tmp_root.iterdir():
        if not _STITCHED_TMP_RE.match(entry.name):
            continue
        try:
            age = now - entry.stat().st_mtime
        except OSError:
            continue
        if age > 3600:                       # 1 h
            with contextlib.suppress(OSError):
                shutil.rmtree(entry, ignore_errors=True)


_sweep_stitched_temp()


app = typer.Typer(
    help="manhwa-recap: AI-guided panels & narration for long strips")
guided_app = typer.Typer(help="AI-guided panel segmentation & narration")
app.add_typer(guided_app, name="guided")
download_app = typer.Typer(help="Download manhwa/webtoon chapter page images "
                                "from scanlation sites (Asura/Vortex/Drake "
                                "Scans, LeviScanner/Madara themes, generic).")
app.add_typer(download_app, name="download")


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "recap-comic"


_VALID_BACKENDS = {"agnes", "fixture", "deterministic", "cv", "manual", "none"}
# Removed providers resolve to Agnes (sole provider) so old commands,
# configs and saved settings keep working.
_LEGACY_BACKEND_ALIASES = {"xkiro", "qwen", "mistral", "gemini", "openai",
                           "anthropic", "ollama", "local", "cloudflare"}
_VALID_TTS = {"edge", "kokoro", "none"}
_VALID_STYLES = {"recap", "literal"}
_VALID_BLANK_SENS = {"low", "conservative", "high"}


def _validate_backend(name: str) -> str:
    n = name.lower()
    # Legacy provider names resolve to Agnes (sole provider) so old
    # commands, configs and saved settings keep working.
    if n in _LEGACY_BACKEND_ALIASES:
        return "agnes"
    if n not in _VALID_BACKENDS:
        raise typer.BadParameter(
            f"unknown backend {name!r}; choose from: {', '.join(sorted(_VALID_BACKENDS))}")
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


def _validate_blank_sensitivity(name: str) -> str:
    n = name.lower()
    if n not in _VALID_BLANK_SENS:
        raise typer.BadParameter(
            f"unknown --blank-sensitivity {name!r}; choose from: "
            f"{', '.join(sorted(_VALID_BLANK_SENS))}")
    return n


def _detect_blanks_for_debug(strip: Path, sensitivity: str):
    """Run the deterministic (no-AI) blank detector on a strip for debug
    overlays. Never raises: an empty list means 'nothing found'."""
    try:
        import numpy as np
        from PIL import Image

        from blank_detector import BLANK, BlankDetectorConfig, detect_blank_regions
        with Image.open(strip) as img:
            img.load()
            gray = np.asarray(img.convert("L"))
            rgb = np.asarray(img.convert("RGB"))
        regions = detect_blank_regions(
            gray, rgb=rgb, config=BlankDetectorConfig(preset=sensitivity))
        for r in regions:
            if r.verdict == BLANK:
                typer.echo(
                    f"Blank region detected\n  Y: {r.y_start}-{r.y_end}\n"
                    f"  Height: {r.height}px\n  Blank score: {r.score:.2f}\n"
                    f"  Reasons: {'; '.join(r.reasons)}")
        return regions
    except Exception as exc:  # noqa: BLE001 - debug tool; never fatal
        log.warning("blank detection for debug overlay failed: %s", exc)
        return []


@guided_app.command("plan")
def guided_plan(
    strip: Path = typer.Argument(..., exists=True, dir_okay=False,
                                 help="tall strip image (PNG/JPG) or CBZ/ZIP archive"),
    out_plan: Path | None = typer.Option(
        None, "--out-plan", help="also write the plan JSON here"),
    backend: str = typer.Option(
        "agnes", "--backend",
        help="agnes (Agnes 2.5 Flash + 2.0 Flash fallback, default)|fixture|deterministic (blank-row CV cut, no AI)|none"),
    model: str | None = typer.Option(
        None, "--model",
        help="vision model id (agnes defaults to agnes-2.5-flash; "
             "agnes-2.0-flash is the automatic fallback)"),
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
    blank_overlay: Path | None = typer.Option(
        None, "--debug-blank-overlay",
        help="run the offline blank-region detector and draw its verdicts "
             "(orange) into this PNG"),
    blank_sensitivity: str = typer.Option(
        "conservative", "--blank-sensitivity",
        help="blank detector sensitivity: low | conservative | high"),
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
    With --debug-blank-overlay, also runs the deterministic (no-AI)
    blank-region detector and writes its visualization.
    """
    _configure_logging(log_level)
    strip = _resolve_strip_path(strip)
    backend = _validate_backend(backend)
    _validate_chunk_params(chunk_height, overlap)
    _validate_blank_sensitivity(blank_sensitivity)
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
    if blank_overlay is not None:
        from debug_view import draw_blank_overlay
        regions = _detect_blanks_for_debug(strip, blank_sensitivity)
        draw_blank_overlay(strip, regions, out_path=blank_overlay)
        typer.echo(f"blank overlay: {blank_overlay} "
                   f"({sum(1 for r in regions if r.verdict == 'blank')} blank, "
                   f"{len(regions)} total candidates)")
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
    output_width: int = typer.Option(
        390, "--output-width",
        help="normalized panel PNG width in px (source crops stay full-res)"),
    min_output_height: int = typer.Option(
        760, "--min-output-height",
        help="minimum normalized panel PNG height in px (pads with black)"),
    max_output_height: int = typer.Option(
        800, "--max-output-height",
        help="maximum normalized panel PNG height in px (panels taller than "
             "this are kept full-res so the video can pan them, not cropped)"),
    no_normalize: bool = typer.Option(
        False, "--no-normalize-output",
        help="write legacy full-resolution panel crops instead of 390x[760,800]"),
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
            variance_threshold=variance_threshold,
            output_width=output_width,
            min_output_height=min_output_height,
            max_output_height=max_output_height,
            normalize_output=not no_normalize,
            force=force,
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
        # The report is a best-effort review artifact: a missing parent
        # directory or an unreadable panel PNG must not undo a successful
        # cut with a raw traceback. Failures here are reported and the cut
        # still exits 0 (the panels.json sidecar is the real output).
        try:
            from report import render_report
            render_report(artifact, report, out_dir)
            typer.echo(f"report: {report}")
        except Exception as exc:
            typer.echo(f"ERROR: report rendering failed: {exc}", err=True)
            if log.isEnabledFor(logging.DEBUG):
                log.exception("report rendering failed")


@guided_app.command("run")
def guided_run(
    strip: Path = typer.Argument(..., exists=True, dir_okay=False,
                                 help="tall strip image or CBZ/ZIP archive"),
    out_dir: Path = typer.Option("guided_out", "--out-dir"),
    backend: str = typer.Option(
        "agnes", "--backend",
        help="agnes (Agnes 2.5 Flash + 2.0 Flash fallback, default)|fixture|deterministic (blank-row CV cut, no AI)|none"),
    model: str | None = typer.Option(
        None, "--model",
        help="vision model id (agnes defaults to agnes-2.5-flash; "
             "agnes-2.0-flash is the automatic fallback)"),
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
    blank_sensitivity: str = typer.Option(
        "conservative", "--blank-sensitivity",
        help="deterministic (no-AI) blank-region removal sensitivity: "
             "low | conservative | high (default conservative)"),
    disable_blank: bool = typer.Option(
        False, "--no-blank-detection",
        help="disable the deterministic blank-region detector"),
    output_width: int = typer.Option(
        390, "--output-width",
        help="normalized panel PNG width in px (source crops stay full-res)"),
    min_output_height: int = typer.Option(
        760, "--min-output-height",
        help="minimum normalized panel PNG height in px (pads with black)"),
    max_output_height: int = typer.Option(
        800, "--max-output-height",
        help="maximum normalized panel PNG height in px (panels taller than "
             "this are kept full-res so the video can pan them, not cropped)"),
    no_normalize: bool = typer.Option(
        False, "--no-normalize-output",
        help="write legacy full-resolution panel crops instead of 390x[760,800]"),
    filter_panels: bool = typer.Option(
        True, "--filter/--no-filter",
        help="run the deterministic panel filter after the cut (default ON; "
             "matches the webapp): blank panels removed, text-only panels "
             "kept as context (no frame/narration). Four gates must pass, "
             "including a dialogue-content gate, so scene panels are never "
             "false-positived. --no-filter for raw cuts"),
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
    _validate_blank_sensitivity(blank_sensitivity)
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
            blank_sensitivity=None if disable_blank else blank_sensitivity,
            output_width=output_width,
            min_output_height=min_output_height,
            max_output_height=max_output_height,
            normalize_output=not no_normalize,
            filter_panels=filter_panels,
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
    assert artifact is not None  # dry_run returned above; artifact is set here
    log.info("guided_run complete panels=%d fallback=%s", len(artifact.panels), used)
    typer.echo(f"cut {len(artifact.panels)} panels into {out_dir}")
    typer.echo(f"sidecar: {Path(out_dir) / 'panels.json'}")


@guided_app.command("filter")
def guided_filter(
    panels_dir: Path = typer.Argument(..., exists=True, file_okay=False,
        help="out dir from 'guided run/cut' (panels.json + panel_*.png)"),
    apply: bool = typer.Option(
        False, "--apply",
        help="overwrite panels.json in place (backup kept as "
             "panels_original.json); default writes panels_filtered.json"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="print decisions only, write nothing"),
    quarantine: bool = typer.Option(
        False, "--quarantine",
        help="move removed-blank panel PNGs to _filtered_panels/ "
             "(apply mode only)"),
    strict: bool = typer.Option(
        False, "--strict", help="IQR fence k=1.0 (catches more text panels)"),
    loose: bool = typer.Option(
        False, "--loose", help="IQR fence k=2.5 (catches fewer)"),
    fixed: bool = typer.Option(
        False, "--fixed", help="skip adaptive calibration; fixed thresholds"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Phase 2.5: drop blank panels; demote text-only panels to context.

    Blank panels (blank_flag from the deterministic blank detector) are
    removed from panels.json. Text-bubble-only panels are kept with
    context_only=True: their dialogue stays for story context, but the
    narrator, TTS and video timeline skip them. Thresholds adapt to the
    session's own panels; deterministic, no AI.
    """
    _configure_logging(log_level)
    from panel_filter import FilterConfig, filter_panels, filter_panels_inplace
    ov: dict[str, object] = {}
    if strict and loose:
        raise typer.BadParameter("--strict and --loose are mutually exclusive")
    if strict:
        ov["iqr_k"] = 1.0
    elif loose:
        ov["iqr_k"] = 2.5
    if fixed:
        ov["min_panels_for_adaptive"] = 999_999
    cfg = FilterConfig().with_overrides(**ov) if ov else FilterConfig()
    try:
        if apply:
            if dry_run:
                raise typer.BadParameter(
                    "--apply and --dry-run are mutually exclusive")
            result = filter_panels_inplace(str(panels_dir), config=cfg,
                                            quarantine_pngs=quarantine)
            out_label = "panels.json (overwritten; backup at panels_original.json)"
        else:
            result = filter_panels(str(panels_dir), config=cfg,
                                   dry_run=dry_run)
            out_label = "(dry-run)" if dry_run else "panels_filtered.json"
    except FileNotFoundError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc
    th = result["thresholds"]
    typer.echo(f"calibration: {th['method']} (n={th['n_panels']})")
    typer.echo(f"panels: {result['total']} total")
    typer.echo(f"kept: {result['kept']} scene, "
               f"{result['context_only']} context-only (no frame), "
               f"{result['removed_blank']} blank removed")
    if result.get("rescued"):
        typer.echo(f"rescued: {result['rescued']} (empty-output guard)")
    typer.echo(f"output: {out_label}")


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


@guided_app.command("narrate-ai")
def guided_narrate_ai(
    panels_dir: Path = typer.Argument(..., exists=True, file_okay=False,
        help="out dir from 'guided run/cut' (panels.json + panel_*.png), "
             "typically produced by the deterministic --backend deterministic cut"),
    model: str | None = typer.Option(
        None, "--model",
        help="vision model id (default agnes-2.5-flash with agnes-2.0-flash "
             "fallback; bare names resolved automatically)"),
    force: bool = typer.Option(
        False, "--force", help="re-narrate even cached panels"),
    narration_concurrency: int = typer.Option(
        16, "--narration-concurrency", min=1, max=32,
        help="per-panel vision calls in flight at once (network-bound; "
             "panels are narrated independently, continuity comes from the "
             "whole-chapter script pass)"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """START button (CLI): AI narration for ALREADY-CROPPED panels.

    Cropping needs no AI; this fills narration/dialogue per panel PNG via
    Agnes (2.5 Flash -> 2.0 Flash). Panel geometry (y ranges, files) is
    never modified.
    """
    _configure_logging(log_level)
    try:
        from adapters import ai_narration as ain
        summary = ain.narrate_cropped_panels(
            panels_dir, model=model or "", force=force,
            concurrency=narration_concurrency)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        if log.isEnabledFor(logging.DEBUG):
            log.exception("guided narrate-ai failed")
        else:
            typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:
        log.exception("unexpected error in guided narrate-ai")
        typer.echo(f"ERROR: unexpected error: {exc} "
                   "(see --log-level DEBUG for details)", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"panels: {summary['panels']}  narrated: {summary['narrated']} "
               f"cached: {summary['cached']}  failed: {len(summary['failed'])}")
    for f in summary["failed"]:
        typer.echo(f"  kept old text: {f}", err=True)


@guided_app.command("title")
def guided_title(
    session_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False,
        help="cut session dir holding script.json (from 'guided narrate-ai')"),
    model: str | None = typer.Option(
        None, "--model", help="text model id (default agnes-2.5-flash)"),
    write: bool = typer.Option(
        True, "--write/--no-write",
        help="also save the primary title to <session_dir>/title.txt"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Generate a CTR YouTube title for the recap (AI-narrator metadata).

    Derives a click-worthy title (plus 2 alternatives) from the finished
    script.json narration WITHOUT rewriting the recap lines. Best-effort and
    offline-safe: needs an AGNES_API_KEY for the live call.
    """
    _configure_logging(log_level)
    import json

    from guided_cutter import CutArtifact
    from recap_script import generate_recap_title, load_script
    # Title is metadata: reuse the current-version script, but also accept an
    # existing (even older-version) script.json's text so a title can be minted
    # WITHOUT rewriting the recap lines.
    text = (load_script(session_dir) or {}).get("text") or ""
    if not text.strip():
        try:
            text = json.loads(
                (session_dir / "script.json").read_text("utf-8")
            ).get("text", "") or ""
        except (OSError, ValueError):
            text = ""
    if not text.strip():
        typer.echo("ERROR: no script.json with narration in this session; run "
                   "'guided narrate-ai <dir>' first.", err=True)
        raise typer.Exit(1)
    series_title = ""
    try:
        art = CutArtifact.model_validate_json(
            (session_dir / "panels.json").read_text("utf-8"))
        series_title = str(art.source or "")
    except Exception:  # noqa: BLE001 - series is just extra title context
        pass
    from adapters import ai_models as _ai
    res = generate_recap_title(
        text, series_title=series_title,
        api_key=_ai.api_key_from_env(), model=model or "")
    title = res.get("title")
    if not title:
        typer.echo("ERROR: title generation failed (model unavailable or "
                   "bad response).", err=True)
        raise typer.Exit(1)
    typer.echo(f"TITLE: {title}")
    for i, alt in enumerate(res.get("alternatives") or [], 1):
        typer.echo(f"  alt {i}: {alt}")
    if write:
        out = session_dir / "title.txt"
        out.write_text(title + "\n", "utf-8")
        typer.echo(f"saved: {out}")


@guided_app.command("video")
def guided_video(
    panels: Path = typer.Argument(
        ..., exists=True, dir_okay=False,
        help="panels.json written by 'guided run' / 'guided cut'"),
    out: Path | None = typer.Option(
        None, "--out",
        help="output mp4 (default: <panels dir>/recap.mp4)"),
    tts: str = typer.Option(
        "edge", "--tts", help="edge (cloud, default) | kokoro (offline) | none (silent)"),
    voice: str = typer.Option(
        "af_heart", "--voice",
        help="kokoro voice id (e.g. af_heart, am_adam, bf_emma)"),
    rate: str = typer.Option("+0%", "--rate", help="legacy option, ignored by kokoro"),
    pitch: str = typer.Option("+0Hz", "--pitch", help="legacy option, ignored by kokoro"),
    speed: float = typer.Option(1.0, "--speed", help="kokoro speed multiplier (0.5-2.0)"),
    kokoro_model_path: Path | None = typer.Option(
        None, "--kokoro-model-path",
        help="path to kokoro-v1.0.onnx (else KOKORO_MODEL_PATH or ./models/)"),
    kokoro_voices_path: Path | None = typer.Option(
        None, "--kokoro-voices-path",
        help="path to voices-v1.0.bin (else KOKORO_VOICES_PATH or ./models/)"),
    dialogue: bool = typer.Option(
        True, "--dialogue/--no-dialogue",
        help="also read each panel's dialogue after its narration"),
    gap: float = typer.Option(
        0.0, "--gap",
        help="silence after each panel (s); 0 keeps the narrator flowing so "
             "panels catch up to the voice instead of pausing on it"),
    min_display: float = typer.Option(
        2.0, "--min-display", help="minimum seconds a panel stays on screen"),
    max_display: float = typer.Option(
        12.0, "--max-display", help="cap for SILENT panels (tts none)"),
    pan_speed: int = typer.Option(
        450, "--pan-speed", help="max pan speed in px/s (lower = slower)"),
    pan_fit_speech: bool = typer.Option(
        False, "--pan-fit-speech/--no-pan-fit-speech",
        help="when a panel's pan would outlast its narration, speed the pan "
             "(bounded at 2x --pan-speed) to fit inside the speech window "
             "instead of holding silent frames after the voice stops "
             "(default off: classic pacing, pan floor always wins)"),
    fps: int = typer.Option(30, "--fps"),
    canvas: str = typer.Option(
        "9:16", "--canvas",
        help="output aspect: 9:16 (1080x1920 portrait, default) | 16:9 "
             "(1920x1080 landscape) | <W>x<H> explicit"),
    blur_background: bool = typer.Option(
        True, "--blur-background/--no-blur-background",
        help="manhwa-recap look: the panel floats on a blurred full-frame "
             "copy of itself (default on; disables the Ken-Burns pan since "
             "the whole panel is already visible)"),
    color_grade: bool = typer.Option(
        False, "--color-grade/--no-color-grade",
        help="moody grade: darken + desaturate + cool blue-gray tint "
             "(default off)"),
    vignette: bool = typer.Option(
        True, "--vignette/--no-vignette",
        help="strong dark vignette around all 4 edges (default on)"),
    vignette_angle: str = typer.Option(
        "PI/2.5", "--vignette-angle",
        help="ffmpeg vignette angle expression; SMALLER = stronger "
             "(PI/3.5 mild, PI/2.5 default, PI/2.1 very aggressive)"),
    blur_sigma: float = typer.Option(
        40.0, "--blur-sigma",
        help="gblur sigma for the blurred background (default 40)"),
    render_preset: str = typer.Option(
        "veryfast", "--render-preset",
        help="libx264 speed preset: ultrafast|superfast|veryfast|faster|fast "
             "(faster = less CPU time, larger file)"),
    tts_concurrency: int = typer.Option(
        6, "--tts-concurrency", min=1, max=32,
        help="TTS clips synthesized in parallel (network-bound)"),
    motion_preset: str = typer.Option(
        "reference", "--motion-preset",
        help="editing style: reference (default; strict sequential camera "
             "cycle from reference_motion_preset.json — zoom in, pan down, "
             "pan up, zoom out, repeated) | none (geometry-driven "
             "automation)"),
    motion_preset_path: Path | None = typer.Option(
        None, "--motion-preset-path",
        help="override JSON for --motion-preset reference (custom template)"),
    motion_strength: float = typer.Option(
        1.0, "--motion-strength",
        help="pan-travel scale for the motion preset (1.0 = as measured, "
             "0 = static camera, zoom rhythm only)"),
    motion_report: bool = typer.Option(
        False, "--motion-report/--no-motion-report",
        help="print the per-shot camera plan (seg, duration, zoom, dx/dy, "
             "normalized movement, panel, confidence) and write "
             "motion_report.json next to the timeline"),
    speech_window: bool = typer.Option(
        True, "--speech-window/--no-speech-window",
        help="trim narration to whole sentences inside the target and stop a "
             "frame from lingering past its (natural-pace) voice; never speeds "
             "up or truncates speech"),
    speech_target: float = typer.Option(
        4.0, "--speech-target",
        help="trim budget aim per panel in seconds (sub-5s window)"),
    speech_max: float = typer.Option(
        4.6, "--speech-max",
        help="hard backstop per spoken panel in seconds (never cuts speech "
             "short, only trims trailing silence/padding; with the default "
             "0-gap the narrator is never sped up or paused)"),
    sfx_dir: Path | None = typer.Option(
        None, "--sfx-dir",
        help="sound-effect bank folder with transition/ action/ reveal/ "
             "subfolders (.wav/.mp3/.ogg/.m4a/.flac; an optional default/ "
             "fills any empty category) — enables automatic SFX mixing"),
    min_silent: float = typer.Option(
        1.0, "--min-silent",
        help="seconds a narration-less panel shows between its narrated "
             "neighbours (panels skipped by the chapter script stay visible "
             "as silent beats); 0 drops them so the timeline is speech-driven "
             "and the narrator never pauses"),
    sfx_volume: float = typer.Option(
        0.9, "--sfx-volume",
        help="global SFX loudness multiplier (per-kind volumes multiply "
             "on top: transition 0.5, action 0.9, reveal 0.6)"),
    ffmpeg: str = typer.Option("ffmpeg", "--ffmpeg", help="ffmpeg executable"),
    ffprobe: str = typer.Option("ffprobe", "--ffprobe", help="ffprobe executable"),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="build narration/audio/timeline/srt but do NOT render the mp4"),
    force: bool = typer.Option(False, "--force", help="ignore all caches"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Phase 3: panels.json -> recap.mp4 (narrated, captioned).

    Reads the per-panel narration, synthesises speech with kokoro (or none),
    builds a drift-free timeline from MEASURED clip durations, pans each
    panel (Ken-Burns) and renders one mp4 with ffmpeg. Also writes recap.srt,
    timeline.json, audio/ and narration.json next to the mp4.

    --canvas selects the output aspect (9:16 portrait default, 16:9 landscape).
    --blur-background/--vignette/--color-grade control the manhwa-recap look.
    """
    _configure_logging(log_level)
    from recap_video import VideoConfig, VideoError, make_recap_video

    def _parse_canvas(spec: str) -> tuple[int, int]:
        s = spec.strip().lower().replace(" ", "")
        named = {"9:16": (1080, 1920), "16:9": (1920, 1080),
                 "portrait": (1080, 1920), "landscape": (1920, 1080)}
        if s in named:
            return named[s]
        if "x" in s:
            try:
                w, h = (int(v) for v in s.split("x", 1))
                if w <= 0 or h <= 0:
                    raise ValueError
                return w, h
            except ValueError:
                pass
        raise ValueError(
            f"bad --canvas {spec!r}: use 9:16, 16:9, or <W>x<H> (e.g. 1920x1080)")

    try:
        canvas_w, canvas_h = _parse_canvas(canvas)
    except ValueError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc

    tts = _validate_tts(tts)
    mp = (motion_preset or "none").lower()
    if mp not in ("none", "reference"):
        typer.echo(
            f"ERROR: unknown --motion-preset {motion_preset!r}; "
            "choose 'none' or 'reference'", err=True)
        raise typer.Exit(1)
    if motion_strength < 0:
        typer.echo("ERROR: --motion-strength must be >= 0", err=True)
        raise typer.Exit(1)
    if speech_target <= 0 or speech_max <= 0 or speech_target > speech_max:
        typer.echo("ERROR: require 0 < --speech-target <= --speech-max",
                   err=True)
        raise typer.Exit(1)
    if sfx_dir is not None and not sfx_dir.is_dir():
        typer.echo(f"ERROR: --sfx-dir not found: {sfx_dir}", err=True)
        raise typer.Exit(1)
    out_path = out or panels.parent / "recap.mp4"
    cfg = VideoConfig(
        tts=tts,  # type: ignore[arg-type]
        voice=voice, rate=rate, pitch=pitch, speed=speed,
        include_dialogue=dialogue, gap_seconds=gap,
        min_display_seconds=min_display, max_display_seconds=max_display,
        min_silent=min_silent,
        max_pan_px_per_sec=pan_speed, pan_fit_speech=pan_fit_speech,
        fps=fps, canvas_w=canvas_w, canvas_h=canvas_h,
        blur_background=blur_background, color_grade=color_grade,
        vignette=vignette, vignette_angle=vignette_angle,
        blur_sigma=blur_sigma,
        render_preset=render_preset, tts_concurrency=tts_concurrency,
        motion_preset=mp, motion_preset_path=motion_preset_path,
        motion_strength=motion_strength,
        speech_window=speech_window, speech_target_seconds=speech_target,
        speech_max_seconds=speech_max,
        sfx_dir=sfx_dir, sfx_volume=sfx_volume,
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
    if summary.get("sfx"):
        typer.echo(f"sfx: {summary['sfx']} ({summary['sfx_events']} events)")
    if summary.get("motion_preset") and summary["motion_preset"] != "none":
        typer.echo(f"motion preset: {summary['motion_preset']}"
                   + (f" ({summary.get('motion_report')})"
                      if summary.get("motion_report") else ""))
    if motion_report and summary.get("motion_report"):
        try:
            import json as _json
            rows = _json.loads(
                Path(str(summary["motion_report"])).read_text("utf-8"))
            typer.echo("motion plan (seg/dur/zoom/dx/dy/kind/conf/panel):")
            for r in rows:
                typer.echo(
                    f"  seg {r.get('seg'):>2d} {r.get('duration_seconds', 0):5.2f}s "
                    f"zoom {r.get('zoom'):4.2f} "
                    f"dx {r.get('dx'):+6.1f} dy {r.get('dy'):+7.1f} "
                    f"{str(r.get('pan_kind')):8s} conf {r.get('confidence'):5.3f} "
                    f"{r.get('panel_id')}")
        except Exception:
            pass
    if dry_run:
        typer.echo("dry run: mp4 not rendered (preview timeline.json"
                   + (" + motion_report.json" if summary.get("motion_report") else "")
                   + "; use a draft render to preview camera movement)")
    else:
        typer.echo(f"video: {summary['video']}")


def _manual_crop(strip: Path, boundaries: list[int], out_dir: Path,
                 force: bool = False) -> CutArtifact:
    """Cut a strip at user-supplied Y boundaries (no AI, no plan.json).

    Boundaries are interior cut rows; 0 and the strip height are implied.
    """
    from guided_cutter import _check_image_size
    out = Path(out_dir)
    if force and out.exists():
        for prev in out.glob("panel_*.png"):
            prev.unlink()
        for stale in ("panels.json", "plan.json"):
            p = out / stale
            if p.exists():
                p.unlink()
    out.mkdir(parents=True, exist_ok=True)
    _check_image_size(strip)
    with Image.open(strip) as img:
        img.load()
        width, height = img.size
        rgb = img.convert("RGB")
    ys = sorted({0, height} | {max(0, min(height, int(b))) for b in boundaries})
    from itertools import pairwise
    ranges = list(pairwise(ys))
    saved: list[dict] = []
    for i, (y0, y1) in enumerate(ranges):
        if y1 - y0 < 30:
            continue
        dest_name = f"panel_{i + 1:03d}.png"
        rgb.crop((0, y0, width, y1)).save(out / dest_name, "PNG")
        saved.append({
            "id": f"{i + 1:03d}",
            "panel_index": i + 1,
            "y_start": y0,
            "y_end": y1,
            "narration": "",
            "dialogue": "",
            "panel_type": "panel",
            "confidence": 1.0,
            "image_file": dest_name,
        })
    artifact = CutArtifact(
        source=strip.name, width=width, height=height,
        plan_hash="manual",
        config={"mode": "manual", "boundaries": boundaries},
        panels=[CutPanel(**p) for p in saved],
    )
    (out / "panels.json").write_text(
        artifact.model_dump_json(indent=2) + "\n", "utf-8")
    return artifact


@guided_app.command("manual")
def manual(
    strip: Path = typer.Argument(..., exists=True, dir_okay=False,
                                 help="tall strip image (PNG/JPG/WebP)"),
    boundaries: list[int] = typer.Option(
        ..., "--at",
        help="interior Y cut rows, e.g. --at 800 --at 1600 --at 2400"),
    out_dir: Path = typer.Option("guided_out", "--out-dir"),
    force: bool = typer.Option(False, "--force"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Manually cut a strip at user-supplied Y boundaries (no AI, no plan).

    Each --at flag is an interior cut row. 0 and the strip height are implied,
    so N boundaries produce N+1 panels.  Example:

        recap-comic guided manual strip.png --at 800 --at 1600 --at 2400
    """
    _configure_logging(log_level)
    strip = _resolve_strip_path(strip)
    try:
        artifact = _manual_crop(strip, boundaries, out_dir, force=force)
    except Exception as exc:
        log.exception("manual crop failed")
        typer.echo(f"ERROR: {exc} (see --log-level DEBUG for details)", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"cut {len(artifact.panels)} panels into {out_dir}")
    typer.echo(f"sidecar: {Path(out_dir) / 'panels.json'}")
    for p in artifact.panels:
        typer.echo(f"  {p.panel_index:02d}  y=[{p.y_start},{p.y_end}]  "
                   f"{p.y_end - p.y_start}px  {p.image_file}")


# --------------------------------------------------------------------------- #
# Website image downloader (website_downloader.py)
# --------------------------------------------------------------------------- #
def _download_progress(done: int, total: int, msg: str) -> None:
    typer.echo(f"  [{done}/{total}] {msg}")


@download_app.command("chapter")
def download_chapter(
    url: str = typer.Argument(...,
                              help="chapter reader page URL (Asura/Vortex/"
                                   "Drake Scans, any LeviScanner/Madara or "
                                   "generic site)"),
    out: Path = typer.Option(..., "--out", "-o", file_okay=False,
                             help="folder for numbered page images "
                                  "(page_001.webp ...)"),
    concurrency: int = typer.Option(
        4, "--concurrency", "-c", min=1, max=16,
        help="parallel image downloads (lower on weak connections)"),
    force: bool = typer.Option(
        False, "--force", help="re-download pages that already exist"),
    inspect: bool = typer.Option(
        False, "--inspect",
        help="print detected image URLs and exit (download nothing) -- "
             "verify the scraper on an unknown site"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Download every page image of ONE chapter reader page.

    Feeds the recap pipeline directly: point scripts/cut_pages_and_merge.py
    (or `guided run` on a CBZ) at the output folder afterwards.
    """
    _configure_logging(log_level)
    import website_downloader as wd
    try:
        if inspect:
            session = wd.make_session()
            html = wd.fetch_html(url, session)
            for u in wd.extract_image_urls(html, url):
                typer.echo(u)
            return
        res = wd.download_chapter(url, out, concurrency=concurrency,
                                  force=force, on_progress=_download_progress)
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        log.exception("download chapter failed")
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"saved {res.page_count} pages -> {out}")
    if res.skipped:
        typer.echo(f"skipped {len(res.skipped)} existing (use --force to redo)")
    if res.failed:
        typer.echo(f"FAILED {len(res.failed)} page(s): "
                   + ", ".join(res.failed[:5]), err=True)
        raise typer.Exit(2)


@download_app.command("series")
def download_series(
    url: str = typer.Argument(..., help="series page URL (has a chapter list)"),
    out: Path = typer.Option(..., "--out", "-o", file_okay=False,
                             help="root folder; one sub-folder per chapter"),
    chapters: str = typer.Option(
        "last", "--chapters", "-c",
        help="which chapters: 'last' (default) | 'all' | a range like "
             "'1-10' | a list like '1,3,5-8'"),
    concurrency: int = typer.Option(4, "--concurrency", "-x", min=1, max=16),
    force: bool = typer.Option(False, "--force"),
    list_only: bool = typer.Option(
        False, "--list", help="print the detected chapter list and exit"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Download a range of chapters from a series page (one folder each).

    Parses the series chapter list, selects the requested range, then reuses
    the `chapter` downloader per chapter. Site chapter lists are matched
    loosely (any link that smells like a chapter) so it survives theme drift.
    """
    _configure_logging(log_level)
    import website_downloader as wd
    try:
        session = wd.make_session()
        links = wd.select_chapters(
            wd.parse_chapter_links(wd.fetch_html(url, session), url),
            "all" if list_only else chapters)
        if list_only:
            for c in links:
                typer.echo(f"  {c.number if c.number is not None else '?':>6}  "
                           f"{c.label or c.url}  {c.url}")
            return
        results = wd.download_series(url, out, chapters=chapters,
                                     session=session, concurrency=concurrency,
                                     force=force, on_progress=_download_progress)
    except Exception as exc:  # noqa: BLE001
        log.exception("download series failed")
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc
    total = sum(r.page_count for r in results)
    typer.echo(f"downloaded {len(results)} chapter(s), {total} pages -> {out}")
    failed = [f for r in results for f in r.failed]
    if failed:
        typer.echo(f"FAILED {len(failed)} page(s) across chapters", err=True)
        raise typer.Exit(2)


# Cinematic effects subcommand (cinematic_effects.py). Registered at module
# scope so the console script `recap-comic guided cinematic` works. Silently
# skipped if the module is missing so cli.py stays usable without the
# cinematic feature.
try:
    from cli_cinematic_patch import add_cinematic_command
    add_cinematic_command(guided_app)
except ImportError:
    pass


if __name__ == "__main__":
    app()
