# guided_pipeline.py
"""Orchestrator for the AI-Guided Panels & Narration feature.

run_guided() wires Phase 1 (strip_analyzer: AI pre-read) to Phase 2
(guided_cutter: physical dissection):

- dry_run=True  -> Phase 1 only; the plan is returned/printed, nothing cut.
- fallback=True (default) -> if the AI call fails OR fewer than 70% of
  panels reach confidence 0.5, the pipeline falls back to the existing
  gutter detector (adapters/panels_opencv.py) and logs a warning (no AI
  narration in that case).
- Phase-1 results are cached keyed by the strip's SHA-256 + reading-config
  hash, so a cut re-run never re-spends tokens.
"""
from __future__ import annotations

import logging
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image

import strip_analyzer as sa
from guided_cutter import CutArtifact, CutterConfig, compute_strip_metrics, guided_cut

log = logging.getLogger(__name__)

LOW_CONF_THRESHOLD = 0.5
LOW_CONF_RATIO_LIMIT = 0.30


def _merge_narration_into_fallback(ai_plan: sa.PanelPlan,
                                   fallback_plan: sa.PanelPlan) -> sa.PanelPlan:
    """Copy narration/dialogue from `ai_plan` into `fallback_plan` based on
    largest Y-overlap, so the geometry comes from the gutter detector but the
    AI's text is preserved."""
    for fb in fallback_plan.entries:
        best_overlap = 0
        best_ai: sa.PanelPlanEntry | None = None
        for ai in ai_plan.entries:
            overlap = (min(fb.y_end, ai.y_end) - max(fb.y_start, ai.y_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_ai = ai
        if best_ai is not None and best_overlap > 0:
            fb.narration = best_ai.narration
            fb.dialogue = best_ai.dialogue
            fb.panel_type = best_ai.panel_type
            fb.confidence = best_ai.confidence
    return fallback_plan


# Re-export so callers can write `gp.VisionAnalysisError` without importing
# strip_analyzer directly. Defined in strip_analyzer to keep the analyzer
# module self-contained.
VisionAnalysisError = sa.VisionAnalysisError


def build_backend(name: str, api_key: str | None = None,
                  model: str | None = None, base_url: str | None = None,
                  timeout: int = 120, cf_account_id: str | None = None
                  ) -> sa.VisionBackend | None:
    """Backend factory. "none" means no AI call at all (offline fallback).
    "local" uses Ollama's POST /api/generate (llava/qwen2-vl)."""
    name = name.lower()
    if name == "none":
        log.debug("backend=none offline fallback")
        return None
    if name == "fixture":
        log.debug("backend=fixture offline test backend")
        return sa.FixtureVisionBackend()
    if name == "gemini":
        kw = {"api_key": api_key}
        if model:
            kw["model"] = model
        backend = sa.GeminiVisionBackend(**kw)
        log.info("backend=gemini model=%s", backend.model)
        return backend
    if name == "openai":
        if not model:
            raise ValueError("--model is required for the openai backend")
        kw = {"model": model, "api_key": api_key}
        if base_url:
            kw["base_url"] = base_url
        log.info("backend=openai model=%s", model)
        return sa.OpenAIVisionBackend(**kw)
    if name == "anthropic":
        if not model:
            raise ValueError("--model is required for the anthropic backend")
        kw = {"model": model, "api_key": api_key}
        if base_url:
            kw["base_url"] = base_url
        log.info("backend=anthropic model=%s", model)
        return sa.AnthropicVisionBackend(**kw)
    if name in ("local", "ollama"):
        kw = {"model": model or "llava", "timeout": timeout}
        if base_url:
            kw["base_url"] = base_url
        log.info("backend=ollama model=%s", kw["model"])
        return sa.OllamaVisionBackend(**kw)
    if name == "cloudflare":
        kw = {"api_key": api_key}
        if model:
            kw["model"] = model
        if cf_account_id:
            kw["account_id"] = cf_account_id
        log.info("backend=cloudflare model=%s", kw.get("model", sa.CloudflareWorkersAIBackend.DEFAULT_MODEL))
        return sa.CloudflareWorkersAIBackend(**kw)
    raise ValueError(
        f"unknown backend {name!r}; supported: gemini, openai, anthropic, "
        "ollama, cloudflare, fixture, none")


def low_confidence_ratio(plan: sa.PanelPlan) -> float:
    """Fraction of panels with confidence < LOW_CONF_THRESHOLD."""
    if not plan.entries:
        return 1.0
    low = sum(1 for e in plan.entries if e.confidence < LOW_CONF_THRESHOLD)
    return low / len(plan.entries)

def fallback_plan_from_gutter_detector(
    strip_path: Path, *, variance_threshold: float = 6.0,
    edge_threshold: float = 30.0, max_panel_height: int = 1600
) -> sa.PanelPlan:
    """A PanelPlan from plain pixel analysis (no AI, no narration).

    Uses the SAME dual-metric (variance + edge density) as Phase 2's gutter
    detection: rows with variance <= variance_threshold AND edge_density <=
    edge_threshold are gutter rows; contiguous runs that are at least 3 rows
    wide become gutters; panel cuts are placed at gutter midpoints. The
    topmost/bottommost runs are treated as page margins.
    Every per-panel confidence is 0.0 on purpose (fallback provenance).
    Panels taller than max_panel_height are split at internal gutters.
    """
    with Image.open(strip_path) as img:
        img.load()
        width, height = img.size
        gray = np.asarray(img.convert("L"))

    variance, edge_density = compute_strip_metrics(gray, use_edge_density=True,
                                                   blur_sigma=0.5)
    run_is_gutter = np.zeros(height, dtype=bool)
    for y in range(height):
        if variance[y] <= variance_threshold and edge_density[y] <= edge_threshold:
            run_is_gutter[y] = True
    min_gutter_width = 3
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for y in range(height):
        if run_is_gutter[y]:
            if start is None:
                start = y
        else:
            if start is not None:
                runs.append((start, y - 1))
                start = None
    if start is not None:
        runs.append((start, height - 1))
    gutters = [r for r in runs
               if r[0] > 0 and r[1] < height - 1
               and r[1] - r[0] + 1 >= min_gutter_width]
    if not gutters:
        raise sa.VisionAnalysisError(
            "fallback detected no usable gutters on this strip")

    cuts = [0] + [(g[0] + g[1]) // 2 for g in gutters]
    if cuts[-1] < height:
        cuts.append(height)
    entries: list[sa.PanelPlanEntry] = []
    for y0, y1 in pairwise(cuts):
        if y1 - y0 < 5:
            continue
        entries.append(sa.PanelPlanEntry(
            panel_index=len(entries) + 1, y_start=y0, y_end=y1,
            narration="", dialogue="", panel_type="unknown", confidence=0.0))
    plan = sa.PanelPlan(source=strip_path.name, width=width, height=height,
                        model="gutter-fallback", config_hash="fallback",
                        input_hash="fallback", provenance="fallback",
                        entries=entries)
    if max_panel_height and entries:
        from guided_cutter import CutPanel, CutterConfig, _split_panel
        dummy = CutPanel(
            id="fb", panel_index=0, y_start=0, y_end=0,
            narration="", dialogue="", panel_type="unknown",
            confidence=0.0, image_file="")
        new_entries: list[sa.PanelPlanEntry] = []
        idx = 1
        for e in entries:
            if e.y_end - e.y_start <= max_panel_height:
                e.panel_index = idx
                new_entries.append(e)
                idx += 1
            else:
                piece = dummy.model_copy(update={
                    "y_start": e.y_start,
                    "y_end": e.y_end,
                })
                pieces = _split_panel(gray, piece, frozenset(),
                                       CutterConfig(max_panel_height=max_panel_height))
                for pc in pieces:
                    new_entries.append(sa.PanelPlanEntry(
                        panel_index=idx, y_start=pc.y_start, y_end=pc.y_end,
                        narration="", dialogue="", panel_type="unknown",
                        confidence=0.0))
                    idx += 1
        plan = plan.model_copy(update={"entries": new_entries})
    return plan


def run_guided(
    strip: str | Path,
    out_dir: str | Path,
    *,
    backend: sa.VisionBackend | None = None,
    backend_name: str = "gemini",
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    cf_account_id: str | None = None,
    plan_path: str | Path | None = None,
    chunk_height: int = sa.DEFAULT_CHUNK_HEIGHT,
    overlap: int = sa.DEFAULT_CHUNK_OVERLAP,
    cache_dir: str | Path | None = None,
    chunk_dir: str | Path | None = None,
    tolerance: int = 80,
    max_panel_height: int = 1600,
    variance_threshold: float = 6.0,
    edge_threshold: float = 30.0,
    fallback: bool = True,
    force: bool = False,
    dry_run: bool = False,
    out_plan: str | Path | None = None,
    validate: bool = False,
) -> tuple[sa.PanelPlan, CutArtifact | None, bool]:
    """Phase 1 + Phase 2. Returns (plan, artifact, used_fallback).

    The artifact is None in dry-run mode. Pass backend=StubBackend() or
    backend_name="none"/"fixture" for offline use; the fallback takes over
    automatically when the AI path raises or yields a low-confidence plan.
    """
    strip = Path(strip)
    if not strip.is_file():
        raise FileNotFoundError(f"strip image not found: {strip}")
    log.info("run_guided start strip=%s backend=%s model=%s chunk_height=%d overlap=%d",
             strip.name, backend_name, model, chunk_height, overlap)

    plan: sa.PanelPlan | None = None
    plan_from_file = False
    if plan_path is not None:
        plan = sa.PanelPlan.model_validate_json(
            Path(plan_path).read_text("utf-8"))
        plan_from_file = True
        log.info("plan loaded from file panels=%d", len(plan.entries))
    elif backend is not None or backend_name.lower() != "none":
        use = backend if backend is not None else build_backend(
            backend_name, api_key=api_key, model=model,
            base_url=base_url, cf_account_id=cf_account_id)
        if use is None:
            raise sa.VisionAnalysisError("no vision backend available")
        try:
            log.info("Phase-1 starting analyze_strip")
            plan, _cached = sa.analyze_strip(
                strip, use, chunk_height=chunk_height, overlap=overlap,
                cache_dir=cache_dir, force=force, chunk_dir=chunk_dir)
            log.info("Phase-1 complete panels=%d provenance=%s cached=%s",
                     len(plan.entries), plan.provenance, _cached)
        except sa.VisionAnalysisError as exc:
            log.error("Phase-1 analysis failed: %s", exc)
            plan = None

    used_fallback = False
    if (plan is not None
            and not plan_from_file
            and plan.provenance != "fallback"
            and low_confidence_ratio(plan) > LOW_CONF_RATIO_LIMIT):
        bad = low_confidence_ratio(plan) * 100.0
        log.warning("AI plan confidence too low (%.0f%% of panels < %.1f); "
                    "falling back to gutter detector for geometry, "
                    "keeping AI narrations", bad, LOW_CONF_THRESHOLD)
        fallback = fallback_plan_from_gutter_detector(
            strip, variance_threshold=variance_threshold)
        plan = _merge_narration_into_fallback(plan, fallback)
        plan.provenance = "fallback"
        used_fallback = True
    if plan is None:
        if not fallback:
            raise sa.VisionAnalysisError(
                "no AI plan and fallback is disabled (--no-fallback)")
        plan = fallback_plan_from_gutter_detector(strip, variance_threshold=variance_threshold)
        used_fallback = True
        log.warning("using gutter-detector fallback (no AI narration)")

    assert plan is not None
    if dry_run:
        log.info("dry_run returning plan panels=%d", len(plan.entries))
        return plan, None, used_fallback

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_json_path = out / "plan.json"
    sa.write_atomic(plan_json_path, plan.model_dump_json(indent=2) + "\n")
    if out_plan is not None:
        sa.write_atomic(Path(out_plan), plan.model_dump_json(indent=2) + "\n")

    config = CutterConfig(tolerance=tolerance,
                          max_panel_height=max_panel_height,
                          variance_threshold=variance_threshold,
                          edge_threshold=edge_threshold)
    log.info("Phase-2 starting guided_cut")
    artifact = guided_cut(strip, plan, out_dir, config=config, force=force, validate=validate)
    log.info("Phase-2 complete panels=%d", len(artifact.panels))
    return plan, artifact, used_fallback
