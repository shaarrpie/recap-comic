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

import logging
import os
import re
import tempfile
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

        # First pass: get dimensions without fully decoding
        dimensions = []
        for name in names:
            with archive.open(name) as fh, Image.open(fh) as img:
                img.load()  # verify it's a valid image
                dimensions.append((name, img.width, img.height))

        total_h = sum(h for _, _, h in dimensions)
        max_w = max(w for _, w, _ in dimensions)
        stitched = Image.new("RGB", (max_w, total_h))
        y = 0

        # Second pass: decode and paste each image
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

    # Create temp file with cleanup registration
    import atexit
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".png",
        prefix=f"recap-comic-{strip.stem}-",
        dir=tempfile.gettempdir()
    )
    os.close(tmp_fd)
    stitched.save(tmp_path, "PNG")
    log.info("archive stitched images=%d out=%s", len(dimensions), tmp_path)

    # Register cleanup - best effort, won't run on hard kill
    def _cleanup(path: Path = Path(tmp_path)) -> None:
        path.unlink(missing_ok=True)

    atexit.register(_cleanup)

    return Path(tmp_path)


app = typer.Typer(
    help="manhwa-recap: AI-guided panels & narration for long strips")
guided_app = typer.Typer(help="AI-guided panel segmentation & narration")
app.add_typer(guided_app, name="guided")


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "recap-comic"


_VALID_BACKENDS = {"xkiro", "qwen", "mistral", "gemini", "openai", "anthropic", "ollama", "local", "cloudflare", "fixture", "deterministic", "cv", "manual", "none"}
_VALID_TTS = {"edge", "kokoro", "none"}
_VALID_STYLES = {"recap", "literal"}
_VALID_BLANK_SENS = {"low", "conservative", "high"}


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
        "xkiro", "--backend",
        help="xkiro (Qwen3.5-397B-A17B + Mistral Medium 3.5 fallback, default)|qwen|mistral|gemini|openai|anthropic|ollama|local|cloudflare|fixture|deterministic (blank-row CV cut, no AI)|none"),
    model: str | None = typer.Option(
        None, "--model",
        help="vision model id (xkiro defaults to Qwen3.5-397B-A17B; "
             "gemini defaults to gemini-2.5-flash; "
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
        help="maximum normalized panel PNG height in px (center-crops)"),
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
        from report import render_report
        render_report(artifact, report, out_dir)
        typer.echo(f"report: {report}")


@guided_app.command("run")
def guided_run(
    strip: Path = typer.Argument(..., exists=True, dir_okay=False,
                                 help="tall strip image or CBZ/ZIP archive"),
    out_dir: Path = typer.Option("guided_out", "--out-dir"),
    backend: str = typer.Option(
        "xkiro", "--backend",
        help="xkiro (Qwen3.5-397B-A17B + Mistral Medium 3.5 fallback, default)|qwen|mistral|gemini|openai|anthropic|ollama|local|cloudflare|fixture|deterministic (blank-row CV cut, no AI)|none"),
    model: str | None = typer.Option(
        None, "--model",
        help="vision model id (xkiro defaults to Qwen3.5-397B-A17B; "
             "gemini defaults to gemini-2.5-flash; "
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
        help="maximum normalized panel PNG height in px (center-crops)"),
    no_normalize: bool = typer.Option(
        False, "--no-normalize-output",
        help="write legacy full-resolution panel crops instead of 390x[760,800]"),
    filter_panels: bool = typer.Option(
        False, "--filter/--no-filter",
        help="run the deterministic panel filter after the cut: blank panels "
             "removed, text-only panels kept as context (no frame/narration)"),
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
        help="vision model id (default Qwen3.5-397B-A17B with Mistral "
             "Medium 3.5 fallback; bare names resolved automatically)"),
    force: bool = typer.Option(
        False, "--force", help="re-narrate even cached panels"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """START button (CLI): AI narration for ALREADY-CROPPED panels.

    Cropping needs no AI; this fills narration/dialogue per panel PNG via
    Qwen -> Mistral. Panel geometry (y ranges, files) is never modified.
    """
    _configure_logging(log_level)
    try:
        from adapters import ai_narration as ain
        summary = ain.narrate_cropped_panels(
            panels_dir, model=model or "", force=force)
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
        tts=tts,  # type: ignore[arg-type]
        voice=voice, rate=rate, pitch=pitch, speed=speed,
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
