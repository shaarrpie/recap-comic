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
from typing import Any

import numpy as np
from PIL import Image

import strip_analyzer as sa
from guided_cutter import CutArtifact, CutterConfig, guided_cut

log = logging.getLogger(__name__)

LOW_CONF_THRESHOLD = 0.5
LOW_CONF_RATIO_LIMIT = 0.30

_OFFLINE_BACKENDS = frozenset(
    {"none", "deterministic", "cv", "manual", "no-ai", "noai"})


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
    if name in ("none", "deterministic", "cv", "manual", "no-ai", "noai"):
        # Pure deterministic CV path: uniform blank-color rows -> gutters ->
        # cuts. No AI call at all; panels get geometry but empty narration.
        # AI narration (if wanted) runs afterwards, per cropped panel, via
        # adapters.ai_narration (button-triggered, never touches geometry).
        log.debug("backend=%s deterministic offline (no AI)", name)
        return None
    if name == "fixture":
        log.debug("backend=fixture offline test backend")
        return sa.FixtureVisionBackend()
    if name == "gemini":
        kw: dict[str, object] = {"api_key": api_key}
        if model:
            kw["model"] = model
        backend: Any = sa.GeminiVisionBackend(**kw)  # type: ignore[arg-type]
        log.info("backend=gemini model=%s", backend.model)
        return backend
    if name == "openai":
        if not model:
            raise ValueError("--model is required for the openai backend")
        kw = {"model": model, "api_key": api_key}
        if base_url:
            kw["base_url"] = base_url
        log.info("backend=openai model=%s", model)
        return sa.OpenAIVisionBackend(**kw)  # type: ignore[arg-type]
    if name == "anthropic":
        if not model:
            raise ValueError("--model is required for the anthropic backend")
        kw = {"model": model, "api_key": api_key}
        if base_url:
            kw["base_url"] = base_url
        log.info("backend=anthropic model=%s", model)
        return sa.AnthropicVisionBackend(**kw)  # type: ignore[arg-type]
    if name in ("local", "ollama"):
        kw = {"model": model or "llava", "timeout": timeout}
        if base_url:
            kw["base_url"] = base_url
        log.info("backend=ollama model=%s", kw["model"])
        return sa.OllamaVisionBackend(**kw)  # type: ignore[arg-type]
    if name in ("xkiro", "qwen", "qwen3.5", "qwen3_5"):
        from adapters import ai_models as _ai
        kw = {}
        if model:
            kw["primary_model"] = model
        if api_key:
            kw["api_key"] = api_key
        kw["base_url"] = base_url or _ai.DEFAULT_BASE_URL
        kw["timeout"] = timeout
        backend = sa.XkiroVisionBackend(**kw)  # type: ignore[arg-type]
        log.info("[AI] backend=xkiro primary=%s fallback=%s",
                 backend.primary_model, backend.fallback_model)
        return backend
    if name == "mistral":
        from adapters import ai_models as _ai
        kw = {"primary_model": model or _ai.FALLBACK_MODEL,
              "fallback_model": _ai.PRIMARY_MODEL}
        if api_key:
            kw["api_key"] = api_key
        kw["base_url"] = base_url or _ai.DEFAULT_BASE_URL
        kw["timeout"] = timeout
        backend = sa.XkiroVisionBackend(**kw)  # type: ignore[arg-type]
        log.info("[AI] backend=mistral primary=%s fallback=%s",
                 backend.primary_model, backend.fallback_model)
        return backend
    if name == "cloudflare":
        kw = {"api_key": api_key}
        if model:
            kw["model"] = model
        if cf_account_id:
            kw["account_id"] = cf_account_id
        log.info("backend=cloudflare model=%s", kw.get("model", sa.CloudflareWorkersAIBackend.DEFAULT_MODEL))
        return sa.CloudflareWorkersAIBackend(**kw)  # type: ignore[arg-type]
    raise ValueError(
        f"unknown backend {name!r}; supported: xkiro, qwen, mistral, "
        "gemini, openai, anthropic, ollama, cloudflare, fixture, "
        "deterministic (aliases: none, cv, manual), none")


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

    Two-layer detection:

    1. VALLEY DETECTION (primary, adaptive): local minima of the smoothed
       row-activity profile (variance + edge density + row-uniformity).
       A gutter is a property of the neighbourhood (a dip between two
       panel peaks), not of the whole strip, so no per-title threshold
       tuning — the fixed "variance <= 6" classification that preceded it
       found 1 gutter on a 12-panel strip and then blind-split at
       midpoints.
    2. Strict gutter-run classification (secondary, the legacy
       variance/edge thresholds) still refines each valley cut to the
       widest qualifying run when one exists nearby.

    Every per-panel confidence is 0.0 on purpose (fallback provenance).
    Oversized panels are split at internal valleys (never blind
    midpoints). blank_detector still flags suspicious pieces afterwards.
    """
    with Image.open(strip_path) as img:
        img.load()
        width, height = img.size
        gray = np.asarray(img.convert("L"))

    from guided_cutter import CutPanel, CutterConfig, _split_panel, find_valley_cuts
    cuts = find_valley_cuts(gray, max_panel_height=max_panel_height)
    if len(cuts) < 2:
        raise sa.VisionAnalysisError(
            "fallback detected no usable gutter runs on this strip")

    # find_valley_cuts already snaps each cut to the midpoint of its gutter
    # run, so no further refinement is needed.
    entries: list[sa.PanelPlanEntry] = []
    for y0, y1 in pairwise(cuts):
        if y1 - y0 < 5:
            continue
        entries.append(sa.PanelPlanEntry(
            panel_index=len(entries) + 1, y_start=y0, y_end=y1,
            narration="", dialogue="", panel_type="unknown", confidence=0.0))
    plan = sa.PanelPlan(source=strip_path.name, width=width, height=height,
                        model="gutter-run", config_hash="fallback",
                        input_hash="fallback", provenance="fallback",
                        entries=entries)
    # oversized panels: split at internal VALLEYS first; the legacy
    # splitter (which searches real gutter runs, midpoints only as an
    # explicitly-logged last resort) handles the rest.
    if max_panel_height and entries:
        from guided_cutter import CutPanel, CutterConfig, _split_panel
        dummy = CutPanel(  # type: ignore[call-arg]
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
                                       CutterConfig(max_panel_height=max_panel_height,
                                                    variance_threshold=variance_threshold))
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
    blank_sensitivity: str | None = "conservative",
    output_width: int = 390,
    min_output_height: int = 760,
    max_output_height: int = 800,
    normalize_output: bool = True,
    filter_panels: bool = False,
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
    blank_sensitivity=None disables the deterministic blank-region detector
    (low | conservative | high otherwise). Panel PNGs are normalized to
    exactly `output_width` px wide with height in
    [`min_output_height`, `max_output_height`] (see
    guided_cutter.normalize_panel_image); source coordinates are untouched.
    Pass normalize_output=False for legacy full-resolution crops.
    filter_panels=True runs panel_filter on the fresh panels.json right
    after the cut (Phase 2.5): blank panels are removed and text-only
    panels are demoted to context_only=True (kept for story context, no
    frame/narration). Deterministic; never re-runs the AI.
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
    elif backend is not None or backend_name.lower() not in _OFFLINE_BACKENDS:
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
        gutter_plan = fallback_plan_from_gutter_detector(
            strip, variance_threshold=variance_threshold,
            edge_threshold=edge_threshold, max_panel_height=max_panel_height)
        plan = _merge_narration_into_fallback(plan, gutter_plan)
        plan.provenance = "fallback"
        used_fallback = True
    if plan is None:
        if not fallback:
            raise sa.VisionAnalysisError(
                "no AI plan and fallback is disabled (--no-fallback)")
        plan = fallback_plan_from_gutter_detector(
            strip, variance_threshold=variance_threshold,
            edge_threshold=edge_threshold, max_panel_height=max_panel_height)
        used_fallback = True
        log.warning("using gutter-detector fallback (no AI narration)")

    if plan is None:
        raise RuntimeError("internal: plan is None after fallback resolution")
    if out_plan is not None and plan is not None:
        # --out-plan must work in dry-run mode too (Phase-1 cache/plan
        # export); the out-dir plan.json below is only for real runs.
        sa.write_atomic(Path(out_plan), plan.model_dump_json(indent=2) + "\n")
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
                          edge_threshold=edge_threshold,
                          blank_detection=blank_sensitivity is not None,
                          blank_preset=blank_sensitivity or "conservative",
                          output_width=output_width,
                          min_output_height=min_output_height,
                          max_output_height=max_output_height,
                          normalize_output=normalize_output,
                          preserve_boundaries=(plan.provenance == "fallback"))
    log.info("Phase-2 starting guided_cut")
    artifact = guided_cut(strip, plan, out_dir, config=config, force=force, validate=validate)
    log.info("Phase-2 complete panels=%d", len(artifact.panels))
    if filter_panels and artifact.panels:
        # Phase 2.5: deterministic panel-content filter. In-place on the
        # fresh artifact; failures never kill the run (panels pass through
        # unfiltered — the safe direction).
        try:
            from panel_filter import filter_panels_inplace
            fres = filter_panels_inplace(out)
            log.info("Phase-2.5 panel filter: kept=%d context_only=%d "
                     "removed_blank=%d [%s]",
                     fres["kept"], fres["context_only"],
                     fres["removed_blank"], fres["thresholds"]["method"])
            artifact = CutArtifact.model_validate_json(
                (Path(out) / "panels.json").read_text("utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("panel filter failed (continuing unfiltered): %s", exc)
    return plan, artifact, used_fallback
