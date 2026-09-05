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


# Re-export so callers can write `gp.VisionAnalysisError` without importing
# strip_analyzer directly. Defined in strip_analyzer to keep the analyzer
# module self-contained.
VisionAnalysisError = sa.VisionAnalysisError


def build_backend(name: str, api_key: str | None = None,
                  model: str | None = None, base_url: str | None = None,
                  timeout: int = 120) -> sa.VisionBackend | None:
    """Backend factory. "none" means no AI call at all (offline fallback).
    "local" uses Ollama's POST /api/generate (llava/qwen2-vl)."""
    name = name.lower()
    if name == "none":
        return None
    if name == "fixture":
        return sa.FixtureVisionBackend()
    if name == "gemini":
        return sa.GeminiVisionBackend(api_key=api_key,
                                       model=model or "gemini-2.5-flash")
    if name == "openai":
        if not model:
            raise ValueError("--model is required for the openai backend")
        return sa.OpenAIVisionBackend(model=model, api_key=api_key)
    if name == "anthropic":
        if not model:
            raise ValueError("--model is required for the anthropic backend")
        return sa.AnthropicVisionBackend(model=model, api_key=api_key)
    if name in ("local", "ollama"):
        return sa.OllamaVisionBackend(model=model or "llava",
                                      base_url=base_url, timeout=timeout)
    if name == "cloudflare":
        return sa.CloudflareWorkersAIBackend(api_key=api_key)
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
    strip_path: Path, *, variance_threshold: float = 6.0
) -> sa.PanelPlan:
    """A PanelPlan from plain pixel analysis (no AI, no narration).

    Uses the SAME row-variance metric as Phase 2's boundary refinement: rows
    with variance <= variance_threshold are gutter rows; contiguous runs that
    are at least 3 rows wide become gutters; panel cuts are placed at gutter
    midpoints. The topmost/bottommost runs are treated as page margins.
    Every per-panel confidence is 0.0 on purpose (fallback provenance).
    """
    with Image.open(strip_path) as img:
        img.load()
        width, height = img.size
        gray = np.asarray(img.convert("L"))

    variance, _ = compute_strip_metrics(gray, use_edge_density=False)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for y in range(height):
        if variance[y] <= variance_threshold:
            start = y if start is None else start
        elif start is not None:
            runs.append((start, y - 1))
            start = None
    if start is not None:
        runs.append((start, height - 1))
    min_gutter_width = 3
    gutters = [r for r in runs
               if r[0] > 0 and r[1] < height - 1  # not a page margin
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
    return sa.PanelPlan(source=strip_path.name, width=width, height=height,
                        model="gutter-fallback", config_hash="fallback",
                        input_hash="fallback", provenance="fallback",
                        entries=entries)


def run_guided(
    strip: str | Path,
    out_dir: str | Path,
    *,
    backend: sa.VisionBackend | None = None,
    backend_name: str = "gemini",
    api_key: str | None = None,
    model: str | None = None,
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
) -> tuple[sa.PanelPlan, CutArtifact | None, bool]:
    """Phase 1 + Phase 2. Returns (plan, artifact, used_fallback).

    The artifact is None in dry-run mode. Pass backend=StubBackend() or
    backend_name="none"/"fixture" for offline use; the fallback takes over
    automatically when the AI path raises or yields a low-confidence plan.
    """
    strip = Path(strip)
    if not strip.is_file():
        raise FileNotFoundError(f"strip image not found: {strip}")

    plan: sa.PanelPlan | None = None
    if plan_path is not None:
        plan = sa.PanelPlan.model_validate_json(
            Path(plan_path).read_text("utf-8"))
    elif backend is not None or backend_name.lower() != "none":
        use = backend if backend is not None else build_backend(
            backend_name, api_key=api_key, model=model)
        if use is None:
            raise sa.VisionAnalysisError("no vision backend available")
        try:
            plan, _cached = sa.analyze_strip(
                strip, use, chunk_height=chunk_height, overlap=overlap,
                cache_dir=cache_dir, force=force, chunk_dir=chunk_dir)
        except sa.VisionAnalysisError as exc:
            log.error("Phase-1 analysis failed: %s", exc)
            plan = None

    used_fallback = False
    # A plan that ALREADY came from the gutter detector (provenance="fallback")
    # has confidence 0.0 on every panel by design. Exempt it from the
    # low-confidence rule, otherwise we'd re-derive the same plan in a loop.
    if (plan is not None
            and plan.provenance != "fallback"
            and low_confidence_ratio(plan) > LOW_CONF_RATIO_LIMIT):
        bad = low_confidence_ratio(plan) * 100.0
        log.warning("AI plan confidence too low (%.0f%% of panels < %.1f); "
                    "falling back to the gutter detector", bad,
                    LOW_CONF_THRESHOLD)
        plan = None  # <-- actually discard the low-confidence AI plan
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
        return plan, None, used_fallback

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_json_path = out / "plan.json"
    sa._write_atomic(plan_json_path, plan.model_dump_json(indent=2) + "\n")
    if out_plan is not None:
        sa._write_atomic(Path(out_plan), plan.model_dump_json(indent=2) + "\n")

    config = CutterConfig(tolerance=tolerance,
                          max_panel_height=max_panel_height,
                          variance_threshold=variance_threshold,
                          edge_threshold=edge_threshold)
    artifact = guided_cut(strip, plan, out_dir, config=config, force=force)
    return plan, artifact, used_fallback
