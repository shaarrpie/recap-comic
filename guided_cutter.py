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


def _blur(gray: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Small Gaussian blur to reduce JPEG ringing before row statistics."""
    import cv2
    k = max(3, int(2 * sigma + 1) | 1)
    return cv2.GaussianBlur(gray, (k, k), sigma)


@dataclass
class CutterConfig:
    tolerance: int = 80
    max_panel_height: int = 1600
    variance_threshold: float = 6.0
    edge_threshold: float = 30.0  # max row edge-density (Sobel/Canny) for a gutter
    bubble_pad: int = 8
    use_edge_density: bool = True  # require gutters to be low on BOTH variance+edge
    min_gutter_run: int = 4       # minimum consecutive low-variance rows
    blur_sigma: float = 0.5       # pre-blur to tolerate JPEG noise


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


def compute_strip_metrics(gray: np.ndarray, use_edge_density: bool,
                           blur_sigma: float = 0.5) -> tuple[np.ndarray, np.ndarray | None]:
    """Pre-compute per-row variance and (optionally) edge density for the WHOLE
    strip once. Returns (variances, edge_density_or_None).

    A small Gaussian blur is applied before statistics to tolerate JPEG
    ringing. The returned 1-D arrays let find_gutter_row and _split_panel
    do cheap vectorised selection instead of recomputing per boundary.
    """
    blurred = _blur(gray, sigma=blur_sigma)
    variances = blurred.astype(np.float32).var(axis=1)
    edge_density: np.ndarray | None = None
    if use_edge_density:
        import cv2
        edges = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
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
    min_gutter_run: int = 4,
    blur_sigma: float = 0.5,
    variances: np.ndarray | None = None,
    edge_density: np.ndarray | None = None,
) -> int | None:
    """Nearest low-variance GUTTER RUN within `tolerance` of center_y.

    A gutter is a contiguous run of at least `min_gutter_run` rows whose
    variance (and edge density, if enabled) is below threshold. The cut is
    placed at the RUN'S MIDPOINT, not the nearest single row, so a wide
    gutter is never carried entirely to one side.

    `forbidden` rows (speech bubbles) are never returned. With
    require_threshold=True, the window must contain at least one valid run;
    otherwise None is returned (caller merges panels). When
    use_edge_density is True, a gutter must ALSO be below edge_threshold.
    With require_threshold=False, the midpoint of the nearest allowed run is
    returned as a fallback so oversized panels can still be split.
    """
    h = gray.shape[0]
    lo = max(0, center_y - tolerance)
    hi = min(h - 1, center_y + tolerance)
    if hi < lo:
        return None

    if variances is None:
        blurred = _blur(gray, sigma=blur_sigma)
        variances = blurred.astype(np.float32).var(axis=1)
    window_var = variances[lo:hi + 1]

    window_edge: np.ndarray | None = None
    if use_edge_density:
        if edge_density is not None:
            window_edge = edge_density[lo:hi + 1]
        else:
            window_edge = row_edge_density(_blur(gray, sigma=blur_sigma), lo, hi + 1)

    mask = window_var <= threshold
    if use_edge_density and window_edge is not None:
        mask = mask & (window_edge <= edge_threshold)

    forbidden_mask = np.zeros(hi - lo + 1, dtype=bool)
    for r in forbidden:
        if lo <= r <= hi:
            forbidden_mask[r - lo] = True
    mask = mask & ~forbidden_mask

    # Find contiguous runs of passing rows.
    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1
    runs = [(lo + s, lo + e) for s, e in zip(starts, ends, strict=True)
            if (e - s + 1) >= min_gutter_run]

    if not runs:
        if not require_threshold:
            return (lo + hi) // 2
        return None

    # Pick the run whose midpoint is nearest to center_y.
    best = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - center_y))
    return (best[0] + best[1]) // 2

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
                 config: CutterConfig,
                 variances: np.ndarray | None = None,
                 edge_density: np.ndarray | None = None) -> list[CutPanel]:
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
            edge_threshold=config.edge_threshold,
            use_edge_density=config.use_edge_density,
            forbidden=forbidden, require_threshold=False,
            min_gutter_run=config.min_gutter_run,
            blur_sigma=config.blur_sigma,
            variances=variances, edge_density=edge_density)
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
    log.debug("build_cuts start panels=%d forbidden_rows=%d", len(entries), len(forbidden))

    variances, edge_density = compute_strip_metrics(
        gray, use_edge_density=config.use_edge_density,
        blur_sigma=config.blur_sigma)

    # 1. Refine boundaries; a None gutter row means continuous art -> merge.
    groups: list[list[PanelPlanEntry]] = [[entries[0]]]
    cut_rows: list[int] = []
    snap_distances: list[int] = []
    for a, b in pairwise(entries):
        center = (a.y_end + b.y_start) // 2
        row = find_gutter_row(
            gray, center, tolerance=config.tolerance,
            threshold=config.variance_threshold,
            edge_threshold=config.edge_threshold,
            use_edge_density=config.use_edge_density,
            forbidden=forbidden, require_threshold=True,
            min_gutter_run=config.min_gutter_run,
            blur_sigma=config.blur_sigma,
            variances=variances, edge_density=edge_density)
        if row is None:
            groups[-1].append(b)
            log.debug("panel %d..%d merged (no gutter at %d)", a.panel_index, b.panel_index, center)
        else:
            groups.append([b])
            cut_rows.append(row)
            snap_distances.append(abs(row - center))

    tops = [entries[0].y_start] + cut_rows
    bottoms = cut_rows + [entries[-1].y_end]
    panel_snaps: list[list[int]] = []
    for i in range(len(groups)):
        top_snap = snap_distances[i - 1] if 0 <= i - 1 < len(snap_distances) else 0
        bot_snap = snap_distances[i] if 0 <= i < len(snap_distances) else 0
        panel_snaps.append([top_snap, bot_snap])

    panels = [_emit(g, int(y0), int(y1), f"{g[0].panel_index:03d}", snaps)
              for g, y0, y1, snaps in zip(groups, tops, bottoms, panel_snaps, strict=True)]
    log.debug("after merge/snap groups=%d", len(panels))

    # 2. Split oversized panels at their internal gutters.
    final: list[CutPanel] = []
    for p in panels:
        if (p.y_end - p.y_start) <= config.max_panel_height:
            final.append(p)
        else:
            pieces = _split_panel(gray, p, forbidden, config,
                                  variances=variances,
                                  edge_density=edge_density)
            log.info("panel %s split into %d pieces (height=%d)",
                     p.id, len(pieces), p.y_end - p.y_start)
            final.extend(pieces)
    log.info("build_cuts result panels=%d", len(final))
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
    sidecar = out / "panels.json"
    if not force and sidecar.exists():
        try:
            existing = CutArtifact.model_validate_json(sidecar.read_text("utf-8"))
            new_hash = hashlib.sha256(
                plan.model_dump_json().encode("utf-8")).hexdigest()
            new_cfg = asdict(config)
            if existing.plan_hash != new_hash or existing.config != new_cfg:
                log.info("plan/config changed since last run; clearing stale panels")
                for prev in out.glob("panel_*.png"):
                    prev.unlink()
                force = True
        except Exception as exc:  # noqa: BLE001 - corrupted sidecar; start fresh
            log.debug("could not read existing sidecar: %s", exc)
            force = True
    if force and out.exists():
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
    log.info("guided_cut start strip=%s size=%dx%d panels_in_plan=%d",
             strip.name, width, height, len(plan.entries))
    if (width, height) != (plan.width, plan.height):
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
        plan = plan.model_copy(deep=True)
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
    saved: list[CutPanel] = []
    for c in cuts:
        y0 = max(0, c.y_start)
        y1 = min(height, c.y_end)
        if y1 <= y0:
            log.warning("cut panel %s has empty/negative range [%d,%d]; skipping",
                        c.id, c.y_start, c.y_end)
            continue
        piece = rgb.crop((0, y0, width, y1))
        dest = out / c.image_file
        if dest.exists() and not force:
            log.info("panel file already exists, skipping: %s", dest)
            saved.append(c)
            continue
        piece.save(dest, "PNG")
        log.debug("saved panel %s y=[%d,%d] size=%dx%d", c.id, y0, y1, width, y1 - y0)
        saved.append(c)

    plan_hash = hashlib.sha256(
        plan.model_dump_json().encode("utf-8")).hexdigest()
    artifact = CutArtifact(
        source=strip.name, width=width, height=height, plan_hash=plan_hash,
        config=asdict(config), panels=saved)
    sidecar = out / "panels.json"
    tmp = sidecar.with_suffix(".json.tmp")
    tmp.write_text(artifact.model_dump_json(indent=2), "utf-8")
    tmp.replace(sidecar)
    log.info("guided_cut complete panels=%d sidecar=%s", len(saved), sidecar)
    return artifact
