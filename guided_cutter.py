# guided_cutter.py
"""Phase 2 — physical dissection of the strip, guided by the AI panel plan.

Why variance instead of brightness: a clean WHITE gutter and a clean BLACK
gutter are both "low-variance rows" — the surrounding art has high variance,
gutter rows are (nearly) uniform. Scanning for low-variance rows therefore
works for both gutter polarities, matching the border-colour detector in
adapters/panels_opencv.py.

Cutting rules (in priority order):
1. Every AI boundary is refined: within +/- tolerance px of the AI-suggested
   Y we snap the cut to the NEAREST low-variance row (the gutter). This
   corrects the model's approximate coordinates.
2. If two adjacent AI panels share NO low-variance row (continuous artwork)
   they are merged and their narrations concatenated.
3. A single AI panel taller than max_panel_height is split at the internal
   gutter closest to its midpoint; pieces are labelled <id>a / <id>b and each
   keeps the parent panel's narration.
4. Cuts are never placed inside a speech bubble: rows covered by the plan's
   bubble_boxes (plus a pad) are excluded from every cut-row search.

Output: one PNG per panel under out_dir plus a panels.json sidecar mapping
every file to its narration, dialogue, original Y range, type, confidence.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from strip_analyzer import PanelPlan, PanelPlanEntry

log = logging.getLogger(__name__)

# Image bomb guard: reject strips above ~80 MP (e.g. 800x100000). Legitimate
# manhwa strips are well under 10 MP. Setting this explicitly avoids Pillow's
# DecompressionBombWarning at import time and gives a clear error.
Image.MAX_IMAGE_PIXELS = 80_000_000


@dataclass
class CutterConfig:
    tolerance: int = 80
    max_panel_height: int = 1600
    variance_threshold: float = 6.0
    edge_threshold: float = 30.0  # max row edge-density (Sobel/Canny) for a gutter
    bubble_pad: int = 8
    use_edge_density: bool = True  # require gutters to be low on BOTH variance+edge


class CutPanel(BaseModel):
    id: str
    panel_index: int  # source AI panel index (integer part)
    y_start: int  # absolute in the source strip
    y_end: int
    narration: str
    dialogue: str
    panel_type: str
    confidence: float
    image_file: str
    split_of: str | None = None  # parent panel id when this is an a/b piece
    merged_with: list[int] = Field(default_factory=list)
    snap_distances: list[int] = Field(default_factory=list)  # AI->final snap px


class CutArtifact(BaseModel):
    source: str
    width: int
    height: int
    plan_hash: str
    config: dict[str, Any]
    panels: list[CutPanel]

def row_edge_density(gray: np.ndarray, y0: int, y1: int) -> np.ndarray:
    """Per-row mean absolute Sobel-X edge magnitude for rows [y0, y1).

    Why this helps: a clean gutter (white or black) has almost no horizontal
    edges, while screentone/gradient backgrounds have high edge density even
    when their per-row variance is also high. Requiring BOTH low variance AND
    low edge density prevents the variance-only metric from mis-firing on
    heavy screentone — the second signal rejects those rows.
    """
    import cv2
    edges = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    mag = np.abs(edges)
    return mag[y0:y1].mean(axis=1)


def compute_strip_metrics(gray: np.ndarray, use_edge_density: bool
                           ) -> tuple[np.ndarray, np.ndarray | None]:
    """Pre-compute per-row variance and (optionally) edge density for the WHOLE
    strip once. Returns (variances, edge_density_or_None).

    Why once per strip, not per boundary: build_cuts and _split_panel each
    call find_gutter_row for windows around many rows. Computing a 1-D
    variance array for the whole strip up front turns each per-boundary call
    into a cheap slice, and lets find_gutter_row do vectorised selection
    instead of a Python loop. For a 20,000px strip this is ~100x fewer
    numpy ops.
    """
    variances = gray.astype(np.float32).var(axis=1)
    edge_density: np.ndarray | None = None
    if use_edge_density:
        import cv2
        edges = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        edge_density = np.abs(edges).mean(axis=1)
    return variances, edge_density


def bubble_rows(plan: PanelPlan, pad: int = 0,
                 gray: np.ndarray | None = None) -> set[int]:
    """All strip rows occupied by any speech bubble (expanded by pad).

    Includes both the AI's bubble_boxes AND bubbles detected by the offline
    OpenCV bubble_detector (if `gray` is provided), so the never-cut-through
    rule is backed by two independent signals. The pixel detector is best-
    effort: if it fails, we fall back to the AI bubbles alone.
    """
    rows: set[int] = set()
    for e in plan.entries:
        for b in e.bubble_boxes:
            rows.update(range(max(0, b.y - pad),
                              min(plan.height, b.y + b.h + pad) + 1))
    if gray is not None:
        try:
            from bubble_detector import detect_bubbles
            for b in detect_bubbles(gray):
                rows.update(range(max(0, b.y - pad),
                                  min(plan.height, b.y + b.h + pad) + 1))
        except Exception as exc:  # noqa: BLE001 - backup signal; never fatal
            log.debug("pixel bubble detector failed (using AI bubbles only): %s",
                      exc)
    return rows


def find_gutter_row(
    gray: np.ndarray,
    center_y: int,
    *,
    tolerance: int,
    threshold: float,
    edge_threshold: float = 30.0,
    use_edge_density: bool = True,
    forbidden: frozenset[int] = frozenset(),
    require_threshold: bool = True,
) -> int | None:
    """Nearest low-variance row within `tolerance` of center_y.

    `forbidden` rows (speech bubbles) are never returned. With
    require_threshold=True, rows above the variance threshold are not
    considered gutters, so continuous artwork yields None (caller merges).
    When use_edge_density is True, a gutter must ALSO be below edge_threshold
    (mean Sobel magnitude) — this second signal prevents mis-firing on
    screentone/gradient backgrounds where variance alone is ambiguous.
    With require_threshold=False, the nearest allowed row is returned as a
    fallback so oversized panels can still be split (caller may log).
    """
    h = gray.shape[0]
    lo = max(0, center_y - tolerance)
    hi = min(h - 1, center_y + tolerance)
    if hi < lo:
        return None
    rows = np.arange(lo, hi + 1)
    var = gray[lo:hi + 1].astype(np.float32).var(axis=1)
    # Pre-compute edge density once for the search window if the dual-signal
    # mode is enabled. A gutter must be low on BOTH variance and edge density;
    # this prevents mis-firing on screentone/gradient backgrounds where
    # variance alone is ambiguous.
    edge: np.ndarray | None = None
    if use_edge_density:
        edge = row_edge_density(gray, lo, hi + 1)
    order = np.argsort(np.abs(rows - center_y))
    fallback: int | None = None
    for k in order:
        r = int(rows[int(k)])
        if forbidden and r in forbidden:
            continue
        is_gutter = float(var[int(k)]) <= threshold
        if use_edge_density and edge is not None:
            is_gutter = is_gutter and float(edge[int(k)]) <= edge_threshold
        if is_gutter:
            return r
        if not require_threshold and fallback is None:
            fallback = r
    return fallback if not require_threshold else None

def _emit(group: list[PanelPlanEntry], y0: int, y1: int,
          base_id: str, snap_distances: list[int] | None = None) -> CutPanel:
    """One CutPanel from a (possibly merged) group of AI entries.

    snap_distances: [top_snap, bottom_snap] in px; logs the AI->final
    snap distance so the smoke test can report model accuracy.
    """
    narration = " ".join(
        e.narration.strip() for e in group if e.narration.strip())
    dialogue = " ".join(
        e.dialogue.strip() for e in group if e.dialogue.strip())
    merged = [e.panel_index for e in group]
    return CutPanel(
        id=base_id,
        panel_index=group[0].panel_index,
        y_start=y0,
        y_end=y1,
        narration=narration,
        dialogue=dialogue,
        panel_type=group[0].panel_type,
        confidence=group[0].confidence,
        image_file=f"panel_{base_id}.png",
        merged_with=merged if len(merged) > 1 else [],
        snap_distances=snap_distances or [],
    )


def _split_panel(gray: np.ndarray, panel: CutPanel,
                 forbidden: frozenset[int],
                 config: CutterConfig) -> list[CutPanel]:
    """Split `panel` at internal gutters until every piece fits
    max_panel_height. Pieces are named <id>a / <id>b (recursively <id>aa...)
    and keep the parent's narration."""
    pieces: list[CutPanel] = []
    stack: list[tuple[int, int, str]] = [
        (panel.y_start, panel.y_end, panel.id)]
    while stack:
        y0, y1, frag_id = stack.pop()
        if (y1 - y0) <= config.max_panel_height:
            pieces.append(panel.model_copy(update={
                "id": frag_id,
                "split_of": panel.id if frag_id != panel.id else None,
                "y_start": y0,
                "y_end": y1,
                "image_file": f"panel_{frag_id}.png",
            }))
            continue
        mid = (y0 + y1) // 2
        row = find_gutter_row(
            gray, mid, tolerance=(y1 - y0) // 2,
            threshold=config.variance_threshold,
            forbidden=forbidden, require_threshold=False)
        if row is None or row <= y0 or row >= y1:
            row = mid
            log.warning("no usable gutter inside panel %s; splitting at "
                        "midpoint %d", frag_id, row)
        stack.append((y0, row, frag_id + "a"))
        stack.append((row, y1, frag_id + "b"))
    return sorted(pieces, key=lambda c: c.y_start)

def build_cuts(gray: np.ndarray, plan: PanelPlan, *,
               config: CutterConfig) -> list[CutPanel]:
    """AI-guided cut plan over the strip's pixel rows.

    Steps: refine AI boundaries (snap to nearest gutter), merge continuous
    art, emit one CutPanel per group, then split any oversized panels.
    """
    entries = sorted(plan.entries, key=lambda e: e.y_start)
    if not entries:
        raise ValueError("plan contains no panels; nothing to cut")
    forbidden = frozenset(bubble_rows(plan, pad=config.bubble_pad, gray=gray))

    # 1. Refine boundaries; a None gutter row means continuous art -> merge.
    groups: list[list[PanelPlanEntry]] = [[entries[0]]]
    cut_rows: list[int] = []
    snap_distances: list[int] = []  # |center - snapped_row| per boundary
    for a, b in pairwise(entries):
        center = (a.y_end + b.y_start) // 2
        row = find_gutter_row(
            gray, center, tolerance=config.tolerance,
            threshold=config.variance_threshold,
            edge_threshold=config.edge_threshold,
            use_edge_density=config.use_edge_density,
            forbidden=forbidden, require_threshold=True)
        if row is None:
            groups[-1].append(b)  # continuous art: merge the two panels
        else:
            groups.append([b])
            cut_rows.append(row)
            snap_distances.append(abs(row - center))

    tops = [entries[0].y_start] + cut_rows
    bottoms = cut_rows + [entries[-1].y_end]
    # snap_distances[i] is the snap for the boundary between groups[i] and
    # groups[i+1] (= bottom of groups[i] = top of groups[i+1]).
    panel_snaps: list[list[int]] = []
    for i in range(len(groups)):
        top_snap = snap_distances[i - 1] if 0 <= i - 1 < len(snap_distances) else 0
        bot_snap = snap_distances[i] if 0 <= i < len(snap_distances) else 0
        panel_snaps.append([top_snap, bot_snap])

    # 2. One CutPanel per (possibly merged) group.
    panels = [_emit(g, int(y0), int(y1), f"{g[0].panel_index:03d}", snaps)
              for g, y0, y1, snaps in zip(groups, tops, bottoms, panel_snaps, strict=True)]

    # 3. Split oversized panels at their internal gutters.
    final: list[CutPanel] = []
    for p in panels:
        if (p.y_end - p.y_start) <= config.max_panel_height:
            final.append(p)
        else:
            final.extend(_split_panel(gray, p, forbidden, config))
    return sorted(final, key=lambda c: (c.y_start, c.id))


def guided_cut(strip_path: str | Path, plan: PanelPlan, out_dir: str | Path,
               *, config: CutterConfig | None = None,
               force: bool = False) -> CutArtifact:
    """Cut the strip into per-panel images + a panels.json sidecar."""
    strip = Path(strip_path)
    if not strip.is_file():
        raise FileNotFoundError(f"strip image not found: {strip}")
    config = config or CutterConfig()
    out = Path(out_dir)
    if force and out.exists():
        # Clear existing outputs for this strip so stale panels from a
        # previous run don't linger alongside the new ones.
        for prev in out.glob("panel_*.png"):
            prev.unlink()
        panels_json = out / "panels.json"
        if panels_json.exists():
            panels_json.unlink()
        for prev in out.glob("debug_*"):
            prev.unlink()
    out.mkdir(parents=True, exist_ok=True)

    with Image.open(strip) as img:
        img.load()
        width, height = img.size
        rgb = img.convert("RGB")
        gray_arr = np.asarray(img.convert("L"))
    if (width, height) != (plan.width, plan.height):
        # The plan may have been generated from a resized copy of the same
        # strip (e.g. downscaled before upload to save tokens). Scale every
        # panel coordinate proportionally. If the aspect ratio diverges by
        # more than 5% we refuse — the plan is for a different image.
        aspect_plan = plan.width / plan.height
        aspect_strip = width / height
        if abs(aspect_plan - aspect_strip) / aspect_plan > 0.05:
            raise ValueError(
                f"plan aspect ratio ({plan.width}x{plan.height}) and strip "
                f"aspect ratio ({width}x{height}) differ by more than 5%; "
                "the plan is for a different image. Re-run 'guided plan' "
                "against this exact strip.")
        sx, sy = width / plan.width, height / plan.height
        log.warning("plan dimensions (%dx%d) differ from strip (%dx%d); "
                    "scaling panel coordinates by (%.3f, %.3f)",
                    plan.width, plan.height, width, height, sx, sy)
         for e in plan.entries:
             e.y_start = round(e.y_start * sy)
             e.y_end = round(e.y_end * sy)
             for b in e.bubble_boxes:
                 b.x = round(b.x * sx)
                 b.y = round(b.y * sy)
                 b.w = round(b.w * sx)
                 b.h = round(b.h * sy)
        plan.width, plan.height = width, height

    cuts = build_cuts(gray_arr, plan, config=config)
    for c in cuts:
        piece = rgb.crop((0, c.y_start, width, c.y_end))
        dest = out / c.image_file
        if dest.exists() and not force:
            log.info("panel file already exists, skipping: %s", dest)
            continue
        piece.save(dest, "PNG")

    plan_hash = hashlib.sha256(
        plan.model_dump_json().encode("utf-8")).hexdigest()
    artifact = CutArtifact(
        source=strip.name, width=width, height=height, plan_hash=plan_hash,
        config=asdict(config), panels=cuts)
    sidecar = out / "panels.json"
    tmp = sidecar.with_suffix(".json.tmp")
    tmp.write_text(artifact.model_dump_json(indent=2), "utf-8")
    tmp.replace(sidecar)  # atomic write
    return artifact
