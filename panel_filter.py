# panel_filter.py
"""Deterministic panel-content filter for recap-comic (Phase 2.5).

Runs between guided_cut (Phase 2) and narration (Phase 3) and removes or
demotes panels whose crop carries no story value:

    * BLANK panels       -> removed from panels.json entirely (the cutter's
                            blank_detector already flags them via
                            blank_flag='blank'; this filter acts on that
                            signal and never re-derives blankness from
                            brightness).
    * TEXT-ONLY panels   -> kept with context_only=True: the story-context
                            reader still sees their dialogue, but narration,
                            TTS and the video timeline skip them (no frame).

Design rules (aligned with blank_detector.py's philosophy):

    * NEVER classify blank by darkness/brightness. blank_flag/blank_score
      written by guided_cutter are the single source of truth; a missing
      field means 'keep' (fail-safe).
    * Text-only detection requires MULTIPLE independent signals: low
      saturation + high white dominance + edge structure + actual dialogue
      text on the panel (panel_type/dialogue from the plan). Pixels alone
      are the weakest evidence; a muted scene panel must never be
      silently dropped.
    * Adaptivity: thresholds are calibrated from the session's own panels
      (median-relative / percentile-based) so a dark-fantasy session does
      not get judged by a bright-sitcom ruler. Fewer than 6 scorable panels
      falls back to fixed defaults.
    * Fail-safe everywhere: unscorable/missing panels are kept; an empty
      result rescues the top-K least-blank panels; a filter never raises
      out of the pipeline (callers decide whether to propagate).

Two-tier output contract (panels.json stays schema-valid):

    Blank panels are dropped from the panels array. context_only=True is a
    REAL CutPanel field (guided_cutter.CutPanel + schemas/panels.schema.json)
    so it survives validation, re-serialization, and the confirmed artifact.
    A summary sidecar filter_summary.json records every decision with its
    evidence for the review UI; panels.json itself carries no extra keys.

Idempotence: filtering an already-filtered panels.json is a no-op
(removed panels are gone; context_only panels are never re-classified).
Re-running guided_cut regenerates panels.json and reverts the filter; the
sidecar records the input's mtime/size so stale filters are detectable.

Usage:
    python panel_filter.py OUT_DIR              # writes panels_filtered.json
    python panel_filter.py OUT_DIR --dry-run    # print decisions only
    python panel_filter.py OUT_DIR --apply      # overwrite panels.json
    python panel_filter.py OUT_DIR --strict     # IQR fence k=1.0 (more)
    python panel_filter.py OUT_DIR --loose      # IQR fence k=2.5 (fewer)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

FILTER_VERSION = 4

# Decisions ---------------------------------------------------------------
DECISION_KEEP = "keep"
DECISION_BLANK = "blank"          # removed from panels.json
DECISION_CONTEXT_ONLY = "context_only"  # kept, but no frame / no narration


@dataclass(frozen=True)
class FilterConfig:
    """All knobs in one place; nothing is tuned per-title by hand."""
    # Adaptive calibration
    min_panels_for_adaptive: int = 6
    iqr_k: float = 1.5            # Tukey fence on color_ratio (strict: 1.0)
    color_ratio_cap: float = 0.15  # adaptive threshold never above this
    fixed_text_color_ratio: float = 0.05  # fixed-mode fallback

    # Text-only gates (all must pass)
    white_dominance_pct: int = 65  # percentile of session white_of_content
    fixed_white_dominance: float = 0.80
    text_edge_floor_pct: int = 10  # percentile of session edge_density
    fixed_text_edge_floor: float = 0.003
    # B&W guard: median color_ratio below this -> session is greyscale,
    # the saturation gate is skipped (cannot separate text on B&W paper)
    bw_color_median_threshold: float = 0.02

    # blank_flag fallback when the field is absent entirely (very old
    # artifacts): use blank_score >= this
    blank_score_fallback: float = 0.70

    # Empty-output guard: never leave panels.json with zero real panels
    min_kept_top_k: int = 3

    # Pixel scoring knobs
    white_pixel_threshold: int = 215   # R,G,B all above = white
    dark_pixel_threshold: int = 25     # used only to trim black padding
    color_sat_gap: int = 25            # max-min channel gap for "colour"

    def with_overrides(self, **ov: Any) -> FilterConfig:
        valid = set(self.__dataclass_fields__)
        bad = set(ov) - valid
        if bad:
            raise ValueError(f"unknown filter overrides: {sorted(bad)}")
        return FilterConfig(**{**self.__dict__, **ov})


# --------------------------------------------------------------------------- #
# Pixel scoring — scores the UNPADDED source crop, never the letterboxed PNG
# --------------------------------------------------------------------------- #
def _score_array(arr: np.ndarray, cfg: FilterConfig) -> dict[str, float]:
    """white_of_content / color_ratio / edge_density for an RGB uint8 array.

    dark_ratio is intentionally absent: blankness is never derived from
    brightness (blank_detector.py: 'a black blank and a white blank behave
    identically'). Edge density is a Sobel-style gradient mean, normalized
    to [0, 1]: text strokes give moderate structured edges, a pure blank
    gives near-zero, real art gives high.
    """
    h, w = arr.shape[:2]
    total = h * w
    if total == 0:
        return {"white_of_content": 0.0, "color_ratio": 0.0,
                "edge_density": 0.0}

    flat = arr.reshape(-1, 3)
    white_mask = ((flat[:, 0] > cfg.white_pixel_threshold)
                  & (flat[:, 1] > cfg.white_pixel_threshold)
                  & (flat[:, 2] > cfg.white_pixel_threshold))
    white_of_content = float(white_mask.sum()) / total

    non_white = flat[~white_mask]
    if len(non_white) == 0:
        color_ratio = 0.0
    else:
        sat = (non_white.astype(np.int16).max(axis=1)
               - non_white.astype(np.int16).min(axis=1))
        color_ratio = float((sat > cfg.color_sat_gap).sum()) / len(non_white)

    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1]
            + 0.114 * arr[:, :, 2]).astype(np.float32)
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    edge_density = float(np.hypot(gx, gy).mean()) / 255.0

    return {"white_of_content": round(white_of_content, 5),
            "color_ratio": round(color_ratio, 5),
            "edge_density": round(edge_density, 6)}


def _trim_black_padding(arr: np.ndarray, dark_thr: int) -> np.ndarray:
    """Remove symmetric top/bottom black padding that normalize_panel_image
    (guided_cutter.py) adds to short panels. Symmetric-trim by first/last
    non-dark row; content with a genuinely dark top edge loses at most the
    padding rows it also has at the bottom."""
    if arr.size == 0:
        return arr
    row_max = arr.max(axis=(1, 2))
    rows = np.where(row_max > dark_thr)[0]
    if len(rows) == 0:
        return arr  # all-dark: leave as-is; blank_flag decides
    return arr[rows[0]:rows[-1] + 1]


def _score_panel(panel: dict[str, Any], strip_path: Path | None,
                 session_dir: Path, cfg: FilterConfig,
                 image_cache: dict[str, np.ndarray] | None = None,
                 ) -> tuple[dict[str, float], str]:
    """Score one panel entry. Returns (scores, method).

    Priority:
      A. raw source crop from the strip via y_start/y_end (full-res, no
         normalization artifacts). The strip is located next to panels.json
         as the artifact's `source` filename (guided_cutter writes the
         sidecar next to the cut PNGs; webapp sessions keep the uploaded
         strip in the session dir under its original name).
      B. the panel PNG with black padding trimmed off (390x[760,800]
         letterbox padding removed before scoring).
    """
    y_start = panel.get("y_start")
    y_end = panel.get("y_end")
    if strip_path is not None and y_start is not None and y_end is not None \
            and int(y_end) - int(y_start) > 0:
        try:
            if image_cache is not None and str(strip_path) in image_cache:
                arr = image_cache[str(strip_path)]
            else:
                with Image.open(strip_path) as img:
                    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
                if image_cache is not None:
                    image_cache[str(strip_path)] = arr
            y0, y1 = max(0, int(y_start)), min(arr.shape[0], int(y_end))
            if y1 > y0:
                return _score_array(arr[y0:y1], cfg), "strip_crop"
        except Exception as exc:  # noqa: BLE001 - scoring must never kill
            log.debug("strip crop scoring failed for %s: %s",
                      panel.get("id", "?"), exc)

    img_rel = panel.get("image_file", "")
    img_path = (Path(img_rel) if Path(img_rel).is_absolute()
                else session_dir / img_rel)
    if not img_path.is_file():
        return {}, "missing"
    try:
        with Image.open(img_path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        arr = _trim_black_padding(arr, cfg.dark_pixel_threshold)
        return _score_array(arr, cfg), "padding_stripped"
    except Exception as exc:  # noqa: BLE001
        log.debug("PNG scoring failed for %s: %s", img_path.name, exc)
        return {}, "error"


# --------------------------------------------------------------------------- #
# Adaptive calibration (text-only thresholds; blank uses cutter metadata)
# --------------------------------------------------------------------------- #
def _calibrate(scores: list[dict[str, float]],
               cfg: FilterConfig) -> dict[str, Any]:
    n = len(scores)
    if n < cfg.min_panels_for_adaptive:
        return {
            "method": "fixed",
            "n_panels": n,
            "text_color_threshold": cfg.fixed_text_color_ratio,
            "white_dominance_cutoff": cfg.fixed_white_dominance,
            "text_edge_floor": cfg.fixed_text_edge_floor,
            "is_bw_session": False,
        }

    col = np.array([s["color_ratio"] for s in scores], dtype=np.float64)
    whi = np.array([s["white_of_content"] for s in scores], dtype=np.float64)

    is_bw = float(np.median(col)) < cfg.bw_color_median_threshold
    if is_bw:
        # Greyscale session: saturation cannot separate text; rely on the
        # white + edge gates alone (see _is_text_only).
        text_color_threshold = cfg.fixed_text_color_ratio
    else:
        q1, q3 = (float(np.percentile(col, 25)),
                  float(np.percentile(col, 75)))
        iqr = q3 - q1
        text_color_threshold = min(max(0.0, q1 - cfg.iqr_k * iqr),
                                   cfg.color_ratio_cap)

    white_dominance_cutoff = min(
        float(np.percentile(whi, cfg.white_dominance_pct)) + 0.04, 0.97)
    # Edge floor is a small ABSOLUTE constant: its only job is to exclude
    # pure-blank crops (edge_density ~ 0). A percentile floor would adapt
    # upward in busy sessions and start rejecting real text panels, which
    # is the fail-unsafe direction (they would be kept, but then the gate
    # is meaningless). Text strokes give structured edges; blanks give none.
    text_edge_floor = cfg.fixed_text_edge_floor

    log.info("panel_filter adaptive n=%d color<%.4f white>%.3f edge>%.5f "
             "bw=%s", n, text_color_threshold, white_dominance_cutoff,
             text_edge_floor, is_bw)
    return {
        "method": "adaptive",
        "n_panels": n,
        "text_color_threshold": round(text_color_threshold, 5),
        "white_dominance_cutoff": round(white_dominance_cutoff, 5),
        "text_edge_floor": round(text_edge_floor, 6),
        "is_bw_session": bool(is_bw),
    }


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def _is_blank(panel: dict[str, Any], cfg: FilterConfig) -> bool:
    """blank_flag is a STRING ('normal' | 'suspicious' | 'blank'); only the
    exact 'blank' verdict removes a panel. Suspicious panels are kept for
    human review (blank_detector contract: never silently drop them)."""
    flag = str(panel.get("blank_flag", "normal") or "normal").lower()
    if flag == "blank":
        return True
    if "blank_score" in panel and flag == "normal":
        # legacy artifact without blank_flag: score threshold, fail-safe low
        try:
            return float(panel["blank_score"]) >= cfg.blank_score_fallback
        except (TypeError, ValueError):
            return False
    return False


def _is_text_only(panel: dict[str, Any], score: dict[str, float],
                  thr: dict[str, Any]) -> bool:
    """Four-signal text-only test. ALL gates must pass:

      1. content gate  — the panel actually carries dialogue text
                          (dialogue non-empty or panel_type is a text type).
                          Pixel signals alone are too weak to remove a
                          panel; this makes scene-panel false positives
                          structurally impossible.
      2. saturation    — color_ratio below threshold (skipped on B&W
                          sessions where it cannot discriminate).
      3. white         — white_of_content above cutoff (text pages are
                          mostly white).
      4. edges         — edge_density above floor (strokes present; a pure
                          blank is near-zero and would fail here too).
    """
    if not score:
        return False
    if panel.get("context_only"):
        return False  # already demoted; idempotent
    has_text = bool(str(panel.get("dialogue") or "").strip()) or \
        str(panel.get("panel_type") or "").lower() in TEXT_PANEL_TYPES
    if not has_text:
        return False
    color_ok = (bool(thr["is_bw_session"])
                or score["color_ratio"] < thr["text_color_threshold"])
    white_ok = score["white_of_content"] > thr["white_dominance_cutoff"]
    edge_ok = score["edge_density"] > thr["text_edge_floor"]
    return bool(color_ok and white_ok and edge_ok)


# panel_type values (strip_analyzer.PANEL_TYPES) that indicate text content
TEXT_PANEL_TYPES = frozenset({
    "transition_gutter",   # thin transition band between panels
    "unknown",             # fallback-provenance panels with dialogue
})


# --------------------------------------------------------------------------- #
# Core filter
# --------------------------------------------------------------------------- #
def _locate_strip(session_dir: Path, data: dict[str, Any]) -> Path | None:
    """Find the source strip next to panels.json.

    guided_cutter stores the strip's bare filename in artifact.source. The
    strip may sit in the session dir (webapp sessions), the session dir's
    parent (CLI 'guided cut': out_dir defaults elsewhere), or the CWD.
    """
    name = str(data.get("source") or "").strip()
    if not name or name in ("manual",):
        return None
    cands = [session_dir / name, session_dir.parent / name, Path.cwd() / name]
    for c in cands:
        if c.is_file():
            return c
    return None


def _renumber(panels: list[dict[str, Any]]) -> None:
    """Re-key panel_index 1..n over the kept set, patching merged_with.

    Keyed by panel id (split pieces 005a/005b share their parent's index —
    index-keyed maps collide). Blank panels are gone before this runs.
    """
    new_index = {p["id"]: i for i, p in enumerate(panels, start=1)}
    old_index: dict[int, int] = {}
    for i, p in enumerate(panels, start=1):
        old_index[int(p.get("panel_index", i))] = i
    for p in panels:
        p["panel_index"] = new_index[p["id"]]
        merged = p.get("merged_with")
        if isinstance(merged, list) and merged:
            p["merged_with"] = [
                old_index[m] for m in merged
                if m in old_index and old_index[m] != new_index[p["id"]]
            ]


def _quarantine_blank_pngs(removed: list[dict[str, Any]],
                           session_dir: Path) -> int:
    """Move blank panels' PNGs to _filtered_panels/ so out.glob('panel_*.png')
    does not see ghost frames (the cutter's cache-clean and any UI glob)."""
    import shutil
    q_dir = session_dir / "_filtered_panels"
    moved = 0
    for p in removed:
        img_rel = p.get("image_file", "")
        img_path = (Path(img_rel) if Path(img_rel).is_absolute()
                    else session_dir / img_rel)
        if img_path.is_file():
            q_dir.mkdir(exist_ok=True)
            try:
                shutil.move(str(img_path), str(q_dir / img_path.name))
                moved += 1
            except OSError as exc:
                log.warning("could not quarantine %s: %s", img_path.name, exc)
    return moved


def filter_panels(
    session_dir: str | Path,
    panels_file: str = "panels.json",
    out_file: str | None = None,
    *,
    config: FilterConfig | None = None,
    dry_run: bool = False,
    quarantine_pngs: bool = False,
) -> dict[str, Any]:
    """Score + classify all panels in <session_dir>/<panels_file>.

    Writes <out_file> (default panels_filtered.json; pass panels.json for
    in-place) plus a filter_summary.json sidecar. Returns a summary dict
    with per-panel annotated decisions. dry_run=True writes nothing.

    Blank panels are removed; text-only panels get context_only=True (a
    real CutPanel field). Neither decision ever touches user-confirmed
    panels (confirmed=True) — the filter runs BEFORE review in the
    pipeline, but stays review-safe when invoked manually.
    """
    cfg = config or FilterConfig()
    d = Path(session_dir)
    src = d / panels_file
    if not src.is_file():
        raise FileNotFoundError(f"panels file not found: {src}")
    data = json.loads(src.read_text("utf-8"))
    panels: list[dict] = list(data.get("panels", []))
    out_file = out_file or ("panels_filtered.json"
                            if panels_file == "panels.json" else panels_file)
    strip_path = _locate_strip(d, data)
    if strip_path is None:
        log.info("panel_filter: source strip not found next to %s; "
                 "scoring PNGs (padding-trimmed)", panels_file)
    image_cache: dict[str, np.ndarray] = {}

    # Pass 1: score every panel -------------------------------------------
    raw_scores: list[dict[str, float]] = []
    methods: list[str] = []
    for p in panels:
        score, method = _score_panel(p, strip_path, d, cfg,
                                     image_cache=image_cache)
        raw_scores.append(score)
        methods.append(method)

    thresholds = _calibrate([s for s in raw_scores if s], cfg)

    # Pass 2: classify ------------------------------------------------------
    out_panels: list[dict] = []
    removed_blank: list[dict] = []
    context_only_ids: set[str] = set()
    annotated: list[dict] = []

    for p, score, method in zip(panels, raw_scores, methods, strict=True):
        pid = str(p.get("id", "?"))
        entry = {**p, "_scores": score, "_score_method": method}

        if p.get("confirmed"):
            entry["_decision"] = DECISION_KEEP
            annotated.append(entry)
            out_panels.append(p)
            continue

        if _is_blank(p, cfg):
            entry["_decision"] = DECISION_BLANK
            annotated.append(entry)
            removed_blank.append(p)
            log.info("[X] %-18s blank (blank_flag=%s score=%.2f)",
                     pid, p.get("blank_flag", "?"), p.get("blank_score", 0))
            continue

        if method in ("missing", "error") or not score:
            # unscorable -> keep (fail safe; blank already handled above)
            entry["_decision"] = DECISION_KEEP
            annotated.append(entry)
            out_panels.append(p)
            continue

        if _is_text_only(p, score, thresholds):
            entry["_decision"] = DECISION_CONTEXT_ONLY
            annotated.append(entry)
            q = {**p, "context_only": True}
            out_panels.append(q)
            context_only_ids.add(pid)
            log.info("[C] %-18s context-only (white=%.3f color=%.4f "
                     "edge=%.5f)", pid, score["white_of_content"],
                     score["color_ratio"], score["edge_density"])
            continue

        entry["_decision"] = DECISION_KEEP
        annotated.append(entry)
        out_panels.append(p)

    # Empty-output guard: never write a panels.json with zero real panels --
    real_kept = [p for p in out_panels if not p.get("context_only")]
    rescued: set[str] = set()
    if not real_kept and panels:
        k = min(cfg.min_kept_top_k, len(out_panels) or len(panels))
        # rescue the least-blank panels (lowest blank_score) that are still
        # present in the output; fall back to the input list when output
        # is empty (everything was blank)
        pool = out_panels or panels
        candidates = sorted(
            pool, key=lambda p: float(p.get("blank_score", 0.5)))[:k]
        rescued = {str(c.get("id")) for c in candidates}
        for c in candidates:
            c.pop("context_only", None)
        for e in annotated:
            if e.get("id") in rescued and e["_decision"] != DECISION_KEEP:
                e["_decision"] = DECISION_KEEP + "_rescued"
        # dedupe by id, preserving order (a rescued panel may already be
        # present once from the classification loop)
        seen: set[str] = set()
        dedup: list[dict] = []
        for p in out_panels + [c for c in candidates
                               if c not in out_panels]:
            pid = str(p.get("id"))
            if pid in seen:
                continue
            seen.add(pid)
            dedup.append(p)
        out_panels = dedup
        context_only_ids -= rescued
        log.warning("all panels classified blank/context-only; rescued "
                    "top-%d by blank_score", len(rescued))

    _renumber(out_panels)

    kept = len([p for p in out_panels if not p.get("context_only")])
    summary = {
        "total": len(panels),
        "kept": kept,
        "context_only": len(out_panels) - kept,
        "removed_blank": len(removed_blank),
        "rescued": len(rescued),
        "thresholds": thresholds,
    }

    if dry_run:
        log.info("[dry-run] kept=%d context_only=%d removed=%d [%s]",
                 kept, summary["context_only"], summary["removed_blank"],
                 thresholds["method"])
        return {**summary, "panels": annotated}

    # Write output -----------------------------------------------------------
    src_stat = src.stat()
    clean = [{k: v for k, v in p.items() if not k.startswith("_")}
             for p in out_panels]
    out_path = d / out_file
    _atomic_write(out_path, json.dumps({**data, "panels": clean},
                                       indent=2, ensure_ascii=False))
    log.info("wrote %s kept=%d context_only=%d removed=%d [%s]",
             out_path.name, kept, summary["context_only"],
             summary["removed_blank"], thresholds["method"])

    # Sidecar (panels.json must stay schema-clean) --------------------------
    import hashlib
    sidecar = {
        "filter_version": FILTER_VERSION,
        "input_file": src.name,
        "input_mtime": src_stat.st_mtime,
        "input_size": src_stat.st_size,
        "out_file": out_path.name,
        "strip_found": strip_path is not None,
        **summary,
        "decisions": [
            {"id": e.get("id"), "decision": e.get("_decision"),
             "method": e.get("_score_method"), "scores": e.get("_scores")}
            for e in annotated
        ],
    }
    sidecar["filter_hash"] = hashlib.sha1(
        json.dumps(sidecar["decisions"], sort_keys=True).encode()
    ).hexdigest()[:12]
    _atomic_write(d / "filter_summary.json",
                  json.dumps(sidecar, indent=2, ensure_ascii=False))

    moved = 0
    if quarantine_pngs and removed_blank:
        moved = _quarantine_blank_pngs(removed_blank, d)
    return {**summary, "panels": annotated, "quarantined": moved}


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, "utf-8")
    tmp.replace(path)


def filter_panels_inplace(
    session_dir: str | Path,
    *,
    config: FilterConfig | None = None,
    quarantine_pngs: bool = False,
) -> dict[str, Any]:
    """Overwrite panels.json in place; first backup wins as
    panels_original.json. Returns filter_panels' summary.

    NOTE: guided_cut owns panels.json's plan_hash; re-running the cut
    regenerates the file and reverts this filter. The webapp runs the
    filter on every build (inside _segment_panels), so staleness is not
    an issue there; CLI users re-run 'guided filter' after re-cutting.
    """
    d = Path(session_dir)
    src = d / "panels.json"
    backup = d / "panels_original.json"
    if src.is_file() and not backup.is_file():
        backup.write_text(src.read_text("utf-8"), "utf-8")
        log.info("backed up panels.json -> panels_original.json")
    return filter_panels(
        session_dir, panels_file="panels.json", out_file="panels.json",
        config=config, quarantine_pngs=quarantine_pngs)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _install_cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Filter blank/text-only panels from a panels.json "
                    "(adaptive thresholds; deterministic).")
    ap.add_argument("session_dir",
                    help="directory containing panels.json + panel_*.png")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print decisions; write nothing")
    ap.add_argument("--apply", action="store_true",
                    help="Overwrite panels.json (backs up to "
                         "panels_original.json)")
    ap.add_argument("--quarantine", action="store_true",
                    help="Move removed-blank PNGs to _filtered_panels/ "
                         "(apply mode only)")
    ap.add_argument("--strict", action="store_true",
                    help="IQR fence k=1.0 (catches more text-only panels)")
    ap.add_argument("--loose", action="store_true",
                    help="IQR fence k=2.5 (catches fewer)")
    ap.add_argument("--fixed", action="store_true",
                    help="Skip adaptive calibration; use fixed thresholds")
    ap.add_argument("--min-panels", type=int, default=None,
                    help="Min panels for adaptive mode (default 6)")
    args = ap.parse_args()

    if args.apply and args.dry_run:
        ap.error("--apply and --dry-run are mutually exclusive")
    if args.strict and args.loose:
        ap.error("--strict and --loose are mutually exclusive")

    ov: dict[str, Any] = {}
    if args.strict:
        ov["iqr_k"] = 1.0
    elif args.loose:
        ov["iqr_k"] = 2.5
    if args.fixed:
        ov["min_panels_for_adaptive"] = 999_999
    if args.min_panels is not None:
        ov["min_panels_for_adaptive"] = args.min_panels
    cfg = FilterConfig().with_overrides(**ov) if ov else FilterConfig()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.apply:
        result = filter_panels_inplace(
            args.session_dir, config=cfg, quarantine_pngs=args.quarantine)
        out_label = "panels.json (overwritten; backup at panels_original.json)"
    else:
        result = filter_panels(args.session_dir, config=cfg,
                               dry_run=args.dry_run)
        out_label = "(dry-run)" if args.dry_run else "panels_filtered.json"

    th = result["thresholds"]
    print(f"\n{'=' * 62}")
    print(f"  Calibration : {th['method']} (n={th['n_panels']})")
    print(f"  Panels      : {result['total']} total")
    print(f"  Kept        : {result['kept']} scene")
    print(f"  Context-only: {result['context_only']} (no frame; dialogue kept)")
    print(f"  Removed     : {result['removed_blank']} blank")
    if result.get("rescued"):
        print(f"  Rescued     : {result['rescued']} (empty-output guard)")
    print(f"  Output      : {out_label}")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    _install_cli()
