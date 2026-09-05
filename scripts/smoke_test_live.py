# scripts/smoke_test_live.py
"""Live smoke test for the AI-guided manhwa cutter (Phase 1 only).

Usage:
    python scripts/smoke_test_live.py samples/real_strip_01.png \
        --backend gemini [--model gemini-2.5-flash] \
        --overlay smoke_overlay.png --plan-out smoke_plan.json

Reads GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY from .env (or the
environment) per backhchoice; also works with --backend ollama (local, free;
runs against http://localhost:11434 by default, OLLAMA_BASE_URL to override).

Prints the numbers requested for honest model-accuracy measurement:
  - number of panels found,
  - mean / median / max gutter-snap distance (px between the AI boundary and
    the refined cut),
  - percentage of panels below confidence 0.5,
  - token usage (best-effort capture per backend; "n/a" when the SDK did not
    surface it),
  - wall time for Phase 1.
Reports large snap distances as a warning - it does NOT tune thresholds to
hide them.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

import guided_pipeline as gp
import strip_analyzer as sa
from debug_view import draw_overlay
from guided_cutter import CutterConfig, build_cuts


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines). Never prints the values."""
    if not path.is_file():
        return
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def snap_stats(cuts: list) -> dict:
    """Mean/median/max of per-boundary snap distances across all panels."""
    dists = [int(d) for c in cuts for d in c.snap_distances]
    if not dists:
        return {"count": 0}
    return {
        "count": len(dists),
        "mean_px": round(statistics.mean(dists), 1),
        "median_px": round(statistics.median(dists), 1),
        "max_px": max(dists),
    }


def token_summary(usage_log: list[dict]) -> dict:
    """Best-effort aggregate of per-chunk token counts (keys vary by SDK)."""
    totals: dict[str, int] = {}
    for usage in usage_log:
        for key, value in usage.items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals or {"n/a": 0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("strip", help="path to the real manhwa strip image")
    ap.add_argument("--backend", default="gemini",
                    choices=["gemini", "openai", "anthropic", "ollama"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--chunk-height", type=int, default=2000)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--snap-tolerance", type=int, default=150)
    ap.add_argument("--variance-threshold", type=float, default=6.0)
    ap.add_argument("--edge-threshold", type=float, default=30.0)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--overlay", default="smoke_overlay.png")
    ap.add_argument("--plan-out", default="smoke_plan.json")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    _load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    strip = Path(args.strip)
    if not strip.is_file():
        print(f"ERROR: strip not found: {strip}")
        return 2
    try:
        backend = gp.build_backend(args.backend, model=args.model)
    except (ValueError, NotImplementedError) as exc:
        # ValueError: unknown backend / missing --model.
        # NotImplementedError: selected backend not yet wired up.
        print(f"ERROR: {exc}")
        return 2
    if backend is None:
        print("ERROR: no usable backend selected")
        return 2

    t0 = time.time()
    plan, used_cache = sa.analyze_strip(
        strip, backend, chunk_height=args.chunk_height, overlap=args.overlap,
        cache_dir=args.cache_dir)
    wall = time.time() - t0

    with Image.open(strip) as img:
        gray = np.asarray(img.convert("L"))
    config = CutterConfig(
        tolerance=args.snap_tolerance,
        variance_threshold=args.variance_threshold,
        edge_threshold=args.edge_threshold)
    cuts = build_cuts(gray, plan, config=config)
    stats_ = snap_stats(cuts)
    low_pct = (sum(1 for e in plan.entries if e.confidence < 0.5)
               / max(1, len(plan.entries))) * 100.0
    tokens = token_summary(getattr(backend, "usage_log", None) or [])

    Path(args.plan_out).write_text(plan.model_dump_json(indent=2), "utf-8")
    overlay_path = draw_overlay(strip, plan, cuts, Path(args.overlay))

    report = {
        "backend": getattr(backend, "name", type(backend).__name__),
        "strip": strip.name,
        "width": plan.width,
        "height": plan.height,
        "panels": len(plan.entries),
        "wall_seconds": round(wall, 2),
        "cache_hit": used_cache,
        "snap_stats": stats_,
        "low_conf_pct": round(low_pct, 1),
        "token_usage": tokens,
        "overlay": str(overlay_path),
        "plan": str(Path(args.plan_out)),
    }
    print(json.dumps(report, indent=2))
    if stats_.get("count") and stats_["max_px"] > args.snap_tolerance:
        print("WARNING: max snap distance exceeds --snap-tolerance; inspect "
              "the overlay - do NOT tune thresholds to hide model error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
