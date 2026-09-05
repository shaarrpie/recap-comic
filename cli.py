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

Backends (--backend): gemini (default) | openai | anthropic | ollama | fixture | none.
"fixture" is for offline tests; "none" forces the gutter-detector fallback.
API keys come ONLY from environment variables (GEMINI_API_KEY /
OPENAI_API_KEY / ANTHROPIC_API_KEY — see .env.example).
"""
from __future__ import annotations

import logging
from pathlib import Path

import typer

import guided_pipeline as gp
import guided_cutter as gc

log = logging.getLogger(__name__)

app = typer.Typer(
    help="manhwa-recap: AI-guided panels & narration for long strips")
guided_app = typer.Typer(help="AI-guided panel segmentation & narration")
app.add_typer(guided_app, name="guided")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s")


@guided_app.command("plan")
def guided_plan(
    strip: Path = typer.Argument(..., exists=True, dir_okay=False,
                                 help="tall strip image (PNG/JPG)"),
    out_plan: Path | None = typer.Option(
        None, "--out-plan", help="also write the plan JSON here"),
    backend: str = typer.Option(
        "gemini", "--backend",
        help="gemini|openai|anthropic|ollama|fixture|none"),
    model: str | None = typer.Option(
        None, "--model",
        help="vision model id (required for openai/anthropic/ollama)"),
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
    try:
        plan, _artifact, _used = gp.run_guided(
            strip, Path("."), backend_name=backend, model=model,
            chunk_height=chunk_height, overlap=overlap,
            cache_dir=cache_dir, chunk_dir=chunk_dir,
            force=force, dry_run=True)
    except (gp.VisionAnalysisError, FileNotFoundError, ValueError) as exc:
        if log.isEnabledFor(logging.DEBUG):
            log.exception("guided plan failed")
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc
        if out_plan is not None:
        _write_atomic_str(out_plan, plan.model_dump_json(indent=2) + "\n")
    if debug_overlay is not None:
        from debug_view import draw_overlay
        draw_overlay(strip, plan, out_path=debug_overlay)
    typer.echo(plan.model_dump_json(indent=2))


@guided_app.command("cut")
def guided_cut(
    strip: Path = typer.Argument(..., exists=True, dir_okay=False,
                                 help="tall strip image"),
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
    try:
        _plan, artifact, _used = gp.run_guided(
            strip, out_dir, backend_name="none", plan_path=plan,
            tolerance=tolerance, max_panel_height=max_panel_height,
            variance_threshold=variance_threshold, force=force)
        assert artifact is not None
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
    typer.echo(f"cut {len(artifact.panels)} panels into {out_dir}")
    typer.echo(f"sidecar: {Path(out_dir) / 'panels.json'}")
    if report is not None:
        from report import render_report
        render_report(artifact, out_dir, report)
        typer.echo(f"report: {report}")


@guided_app.command("run")
def guided_run(
    strip: Path = typer.Argument(..., exists=True, dir_okay=False,
                                 help="tall strip image"),
    out_dir: Path = typer.Option("guided_out", "--out-dir"),
    backend: str = typer.Option(
        "gemini", "--backend",
        help="gemini|openai|anthropic|ollama|fixture|none"),
    model: str | None = typer.Option(
        None, "--model",
        help="vision model id (required for openai/anthropic/ollama)"),
    plan_path: Path | None = typer.Option(
        None, "--plan", help="reuse an existing plan JSON from 'guided plan'"),
    chunk_height: int = typer.Option(2000, "--chunk-height"),
    overlap: int = typer.Option(200, "--overlap"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
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
    try:
        plan, artifact, used = gp.run_guided(
            strip, out_dir, backend_name=backend, model=model,
            plan_path=plan_path, chunk_height=chunk_height, overlap=overlap,
            cache_dir=cache_dir, tolerance=tolerance,
            max_panel_height=max_panel_height,
            variance_threshold=variance_threshold,
            edge_threshold=edge_threshold, fallback=fallback,
            force=force, dry_run=dry_run)
    except (gp.VisionAnalysisError, FileNotFoundError, ValueError,
            AssertionError) as exc:
        if log.isEnabledFor(logging.DEBUG):
            log.exception("guided run failed")
        else:
            typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc
    if dry_run:
        typer.echo(plan.model_dump_json(indent=2))
        return
    assert artifact is not None
    typer.echo(f"cut {len(artifact.panels)} panels into {out_dir} "
               f"(used_fallback={used})")
    typer.echo(f"sidecar: {Path(out_dir) / 'panels.json'}")


@guided_app.command("narrate")
def guided_narrate(
    plan: Path = typer.Argument(..., exists=True, dir_okay=False,
                                help="plan JSON from 'guided plan'"),
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
    typer.echo(f"wrote {out} ({len(script)} chars)")
    typer.echo(f"index: {out.with_suffix('.index.json')}")


if __name__ == "__main__":
    app()
