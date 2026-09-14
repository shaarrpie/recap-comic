#!/usr/bin/env python
"""Regenerate broken webapp sessions in place (offline replay of the UI flow).

Fixes applied to the cutter (normalize no longer center-crops mega-panels)
and the continuation merge (output geometry carried through) only affect
FUTURE runs. Sessions segmented before the fix still hold 390x800 center-
crops of full-bleed pages, so their recap.mp4 shows ~10-30% of the art.

This script replays the webapp's own pipeline for an existing chain:

1. For every session in the continuation chain (oldest first), re-run
   segmentation IN PLACE: gp.run_guided with the session's cached plan.json
   (no new AI spend), force=True so the broken panel PNGs are rewritten
   full-resolution, then panel_filter.filter_panels_inplace (Phase 2.5).
   Existing review state is untouched: ids/boundaries are identical
   (same plan + same cutter config except normalization), and
   panels_edit.json / review.json are keyed by panel id.

2. Drive a real 'generate' job for the LATEST session via
   webapp.pipeline.run_job (same stages, same order, same store), exactly
   like POST /api/run does: apply_confirmed -> merge_continuation ->
   apply_order -> gemini_narration -> build_script -> render_video ->
   save_outputs -> create_editor_project.

Usage:
    python scripts/regen_webapp_chain.py SESSION [--tts edge|none] [--keep-jobs]

Examples:
    python scripts/regen_webapp_chain.py 135ab2689a88
    python scripts/regen_webapp_chain.py 135ab2689a88 --tts none
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("regen_webapp_chain")


def _chain_sessions(session: str) -> list[str]:
    """Continuation chain oldest-first, EXCLUDING the session itself
    (same walk as webapp.panel_api.chain_for)."""
    from webapp.panel_api import chain_for
    return chain_for(session)


def _recut_session(session: str) -> None:
    """Re-run segmentation for one session in place (cached plan, force)."""
    import guided_pipeline as gp
    from panel_filter import filter_panels_inplace
    from webapp.main import OUTPUT_DIR

    d = OUTPUT_DIR / session
    strip = d / "strip.webp"
    if not strip.is_file():
        strip = d / "strip.png"
    plan = d / "plan.json"
    if not strip.is_file() or not plan.is_file():
        log.warning("session %s: strip/plan missing; skipped re-cut", session)
        return
    log.info("re-cutting %s (%s, plan cache)", session, strip.name)
    _plan, artifact, _used = gp.run_guided(
        strip, d, backend_name="none", plan_path=plan, force=True,
        validate=True)
    assert artifact is not None
    fres = filter_panels_inplace(d)
    log.info("session %s re-cut: %d panels (filter kept %d, removed %d)",
             session, len(artifact.panels), fres["kept"],
             fres["removed_blank"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session", help="latest session id of the chain")
    ap.add_argument("--tts", default="edge", choices=["edge", "none"])
    ap.add_argument("--skip-recut", action="store_true",
                    help="only replay the generate job (panels already ok)")
    args = ap.parse_args()

    session = args.session
    chain = _chain_sessions(session)
    log.info("chain: %s -> %s", " -> ".join(chain) if chain else "(none)",
             session)

    # 1. re-cut every strip in the chain (incl. the latest) in place
    if not args.skip_recut:
        for s in [*chain, session]:
            _recut_session(s)

    # 2. replay the webapp generate job for the latest session
    from webapp.main import OUTPUT_DIR
    from webapp import pipeline as pl
    from webapp.jobs import store

    d = OUTPUT_DIR / session
    cfg = {
        "session": session,
        "strip_file": "strip.webp"
                      if (d / "strip.webp").is_file() else "strip.png",
        "order": None,
        "tts": args.tts,
        "voice": "en-US-AriaNeural",
        "style": "recap",
        "rate": 0,
        "pitch": 0,
        "backend": "none",      # plan.json is the source; no AI call
        "model": "",
        "endpoint": "",
        "cf_account_id": "",
        "start_stage": None,
        "continue_from": chain[-1] if chain else None,
    }
    job = store.create("generate", cfg)
    log.info("replaying generate job=%s session=%s tts=%s continue_from=%s",
             job.id, session, args.tts, cfg["continue_from"])
    pl.run_job(job.id)  # synchronous: same stages the UI runs

    print(f"\njob {job.id}: status={job.status.value} "
          f"error={job.error or 'none'}")
    if job.status.value == "failed":
        for e in list(job._logs)[-14:]:
            print(f"  [{e['level']}] {e['msg']}")
        sys.exit(1)
    out = d / "recap.mp4"
    print(f"outputs: {out} "
          f"({'exists' if out.is_file() else 'MISSING'})")
    print(f"         {d / 'timeline.json'}")
    print(f"         {d / 'recap.srt'}")


if __name__ == "__main__":
    main()
