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

Output-size policy (backward compatible): panel PNGs are normalized to
exactly 390px wide with height clamped to [760, 800]px via
normalize_panel_image() (aspect-preserving resize, then deterministic
center-crop / center-pad). Source geometry (y_start/y_end, artifact
width/height) and the full-resolution source crop are never altered;
only the PNG bytes written to disk are normalized. Set
CutterConfig(normalize_output=False) to restore legacy full-res crops.
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

try:
    from blank_detector import (
        BLANK,
        BlankDetectorConfig,
        BlankRegion,
        detect_blank_regions,
        score_crop,
    )
    _HAS_BLANK_DETECTOR = True
except ImportError:  # pragma: no cover - blank_detector ships with the project
    _HAS_BLANK_DETECTOR = False

log = logging.getLogger(__name__)


def _check_image_size(path: Path) -> None:
    with Image.open(path) as img:
        pixels = img.width * img.height
        if pixels > 80_000_000:
            raise ValueError(
                f"image too large for safe processing: {img.width}x{img.height} "
                f"({pixels / 1_000_000:.1f} MP); refusing to load")


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
    # Output PNG normalization. Source coordinates and source crops remain
    # full resolution; this only controls the panel image written to disk.
    # Policy: every panel PNG is exactly `output_width` px wide with its
    # height clamped to [min_output_height, max_output_height]. The source
    # crop is first resized to `output_width` (aspect-preserving, LANCZOS),
    # then center-cropped (if taller than max) or center-padded with black
    # (if shorter than min). Source geometry (y_start/y_end, artifact
    # width/height) is never altered by this step.
    output_width: int = 390
    min_output_height: int = 760
    max_output_height: int = 800
    normalize_output: bool = True  # False restores legacy full-res crops
    # Structure-first mode (fallback-provenance plans): a boundary is never
    # dropped just because no strict gutter run exists between two entries.
    # AI plans keep merge-on-continuous-art; valley/fallback plans keep every
    # detected boundary (colored/gradient gutters fail the strict run test).
    preserve_boundaries: bool = False
    # --- deterministic blank-region removal (NO AI) --------------------
    blank_detection: bool = True   # run the blank-region detector at all
    blank_preset: str = "conservative"  # low | conservative | high

    def __post_init__(self) -> None:
        if self.output_width <= 0:
            raise ValueError("output_width must be positive")
        if self.min_output_height <= 0:
            raise ValueError("min_output_height must be positive")
        if self.max_output_height < self.min_output_height:
            raise ValueError(
                "max_output_height must be greater than or equal to "
                "min_output_height")


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
    # Normalized PNG dimensions written by the cutter. Source coordinates and
    # the source strip dimensions remain unchanged in panels.json.
    output_width: int | None = None
    output_height: int | None = None
    # Width of THIS panel's source strip. None (default) for single-strip
    # artifacts; set for panels merged in from another strip whose strip
    # width differs from the artifact's own width (continuation sequences).
    strip_width: int | None = None
    split_of: str | None = None  # parent panel id when this is an a/b piece
    merged_with: list[int] = Field(default_factory=list)
    snap_distances: list[int] = Field(default_factory=list)  # AI->final snap px
    # --- deterministic blank analysis (NO AI; set by the blank detector) --
    blank_score: float = Field(0.0, ge=0.0, le=1.0)
    blank_flag: str = Field("normal")  # normal | suspicious | blank
    blank_reasons: list[str] = Field(default_factory=list)


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


def normalize_panel_image(piece: Image.Image, *,
                          output_width: int = 390,
                          min_output_height: int = 760,
                          max_output_height: int = 800) -> Image.Image:
    """Deterministically normalize one source crop to 390x[760,800].

    Step 1: aspect-preserving resize so the width is exactly `output_width`
    (LANCZOS; height rounded to the nearest int, minimum 1px). Step 2: if
    the resized height exceeds `max_output_height`, center-crop to the max;
    if it is below `min_output_height`, center-pad with black to the min.
    Otherwise the resized image is returned unchanged.

    Never touches source geometry or AI boundaries — it only reshapes the
    already-cropped PNG that is written to disk.
    """
    if piece.width <= 0 or piece.height <= 0:
        raise ValueError("cannot normalize an empty panel image")
    if output_width <= 0 or min_output_height <= 0:
        raise ValueError("output dimensions must be positive")
    if max_output_height < min_output_height:
        raise ValueError("max_output_height must be >= min_output_height")
    scale = output_width / float(piece.width)
    scaled_h = max(1, int(round(piece.height * scale)))
    resized = piece.resize((output_width, scaled_h), Image.LANCZOS)
    if scaled_h > max_output_height:
        top = (scaled_h - max_output_height) // 2
        return resized.crop((0, top, output_width, top + max_output_height))
    if scaled_h < min_output_height:
        canvas = Image.new("RGB", (output_width, min_output_height), (0, 0, 0))
        canvas.paste(resized, (0, (min_output_height - scaled_h) // 2))
        return canvas
    return resized


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


def row_mono_fraction(gray: np.ndarray, y: int, *,
                      tol: int = 8) -> float:
    """Fraction of the row's columns that are near-constant.

    A real gutter is uniform across the FULL strip width; a character's
    solid black hair is uniform only over its local extent. But many
    webtoon gutters are broken by a character standing in them or a border
    line, so requiring full-width uniformity rejects every real gutter on
    those strips and the cutter falls back to the nearest locally-flat run
    — which is how a cut ends up mid-character.

    Majority-width fixes that: a row counts as a gutter row when >=
    `mono_width_frac` of its columns are within `tol` of the row median.
    """
    row = gray[y].astype(np.int16)
    med = float(np.median(row))
    return float((np.abs(row - med) <= tol).mean())


def find_valley_cuts(
    gray: np.ndarray,
    *,
    min_panel_height: int = 120,
    max_panel_height: int = 1600,
    mono_width_frac: float = 0.70,
    forbidden: frozenset[int] = frozenset(),
) -> list[int]:
    """Adaptive panel cuts by finding GUTTER RUNS in the strip.

    A gutter is a horizontal band where most of the strip is ONE colour.
    That is the property we actually want to cut on, and it is NOT the same
    as low row variance: a character standing in a gutter makes the row half
    white / half black, so its variance is HIGH and a variance-based valley
    search walks straight past it. The majority-width test sees it
    correctly (60% white still passes at the default 0.70).

    So: find runs of rows that are uniform over >= mono_width_frac of the
    strip width, whose median colour matches the page background, and whose
    width is in the gutter range. Each run is a cut at its midpoint.

    A character's solid hair is uniform only over its local extent, so it
    still fails this test; speed-line art is streaky, not flat.

    Fails safe: no validated gutters -> fewer cuts (panels stay whole,
    user splits by hand) — NEVER a blind midpoint cut.
    """
    h = gray.shape[0]
    if h < 3 * min_panel_height:
        return []

    # --- background colour estimate ---------------------------------------
    # Real webtoon strips vary: clean strips have flat page margins
    # (estimable from the edges), but full-bleed strips have NO flat
    # margins while their gutters are still uniform runs somewhere in
    # the middle. Strategy:
    #   1. flat margin rows exist (>= 5) -> their median is the bg
    #   2. else: the colour that dominates all NEAR-MONO ROWS across the
    #      whole strip (weighted by row) — gutters are, by definition,
    #      the most common flat-band colour; margins-only estimation
    #      fails on full-bleed strips and rejects every real gutter.
    margin_h = max(8, h // 10)
    top_band = gray[:margin_h].astype(np.int16)
    bot_band = gray[h - margin_h:].astype(np.int16)
    def _flat_rows(band: np.ndarray) -> np.ndarray:
        return (band.max(axis=1) - band.min(axis=1)) <= 8
    top_flat = top_band[_flat_rows(top_band)]
    bot_flat = bot_band[_flat_rows(bot_band)]
    flats = np.concatenate([top_flat.reshape(-1),
                            bot_flat.reshape(-1)]) \
        if (len(top_flat) or len(bot_flat)) else np.array([])
    if len(flats) >= 5:
        bg = float(np.median(flats.astype(np.float64)))
    else:
        # dominant near-mono row colour across the whole strip
        row_span = gray.astype(np.int16)
        row_span = row_span.max(axis=1) - row_span.min(axis=1)
        mono_rows = np.where(row_span <= 8)[0]
        if len(mono_rows) >= 10:
            vals, counts = np.unique(gray[mono_rows], return_counts=True)
            bg = float(vals[np.argmax(counts)])
        else:
            # last resort: histogram mode
            hist, bin_edges = np.histogram(gray, bins=64)
            bg = float((bin_edges[np.argmax(hist)]
                        + bin_edges[np.argmax(hist) + 1]) / 2)

    # --- gutter-run scan --------------------------------------------------
    # A gutter is a horizontal band where most of the strip is ONE colour.
    # That is the property we actually want to cut on, and it is NOT the
    # same as low row variance: a character standing in a gutter makes the
    # row half white / half black, so its variance is HIGH and a
    # variance-based valley search walks straight past it. The majority-
    # width test sees it correctly (60% white still passes at 0.70).
    #
    # So: find runs of rows that are uniform over >= mono_width_frac of the
    # strip width, whose median colour matches the page background, and
    # whose width is in the gutter range. Each run is a cut at its midpoint.
    span = gray.astype(np.int16)
    row_med = np.median(span, axis=1, keepdims=True)
    mono_frac = (np.abs(span - row_med) <= 8).mean(axis=1)
    mono_rows = mono_frac >= mono_width_frac

    # runs of gutter rows
    runs: list[tuple[int, int]] = []
    i = 0
    while i < h:
        if bool(mono_rows[i]):
            j = i
            while j < h and bool(mono_rows[j]):
                j += 1
            if 3 <= (j - i) <= 120:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1

    # (b) background colour: a gutter run's median must match the page bg.
    validated: list[tuple[int, float]] = []
    for lo, hi in runs:
        band = gray[lo:hi + 1].astype(np.float64)
        med = float(np.median(band))
        if abs(med - bg) > 24:
            continue
        if any(lo <= r <= hi for r in forbidden):
            continue
        validated.append(((lo + hi) // 2, 0.0))

    # --- spacing + max-height handling ------------------------------------
    cuts: list[int] = [0]
    for cut, _score in sorted(validated):
        if cut - cuts[-1] >= min_panel_height:
            cuts.append(cut)
    if cuts[-1] < h - min_panel_height:
        cuts.append(h)
    elif cuts[-1] != h:
        cuts[-1] = h
    return cuts


def bubble_rows(plan: PanelPlan, pad: int = 0,
               gray: np.ndarray | None = None,
               max_box_width_frac: float = 0.4,
               max_box_height_frac: float = 0.15) -> set[int]:
    """All strip rows occupied by any speech bubble (expanded by pad).

    Includes both the AI's bubble_boxes AND bubbles detected by the offline
    OpenCV bubble_detector (if `gray` is provided), so the never-cut-through
    rule is backed by two independent signals. The pixel detector is best-
    effort: if it fails, we fall back to the AI bubbles alone.
    """
    rows: set[int] = set()
    max_w = plan.width * max_box_width_frac
    max_h = plan.height * max_box_height_frac
    for e in plan.entries:
        for b in e.bubble_boxes:
            if b.w > max_w or b.h > max_h:
                log.debug("ignoring oversized bubble box %s on panel %d",
                          b, e.panel_index)
                continue
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
    return CutPanel(  # type: ignore[call-arg]
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
    and keep the parent's narration.

    NEVER cuts blindly: when no structurally-valid gutter exists inside
    an oversized panel, the panel is kept WHOLE (returned unsplit) — a
    too-tall panel is recoverable by the user in Manual Crop, but a cut
    through mid-fight action is not. The old midpoint fallback is what
    produced "characters cut mid-fight" on action strips.
    """
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
            forbidden=forbidden, require_threshold=True,
            min_gutter_run=config.min_gutter_run,
            blur_sigma=config.blur_sigma,
            variances=variances, edge_density=edge_density)
        # BALANCED-split guard: both pieces must be a reasonable fraction of
        # the parent, otherwise a gutter found near an edge produces a tiny
        # sliver + an almost-unchanged oversized remainder, and the recursion
        # degenerates into dozens of lopsided slivers (panel_027babababbaa
        # style). Require each side to be at least min_piece tall AND at
        # least 25% of the parent so splits stay balanced.
        min_piece = 50
        quarter = (y1 - y0) // 4
        min_side = max(min_piece, quarter)
        if row is not None and not (y0 + min_side <= row <= y1 - min_side):
            log.debug("split row %d too close to edge for %s (need [%d,%d])",
                      row, frag_id, y0 + min_side, y1 - min_side)
            row = None
        if row is None or row <= y0 or row >= y1:
            # No validated gutter inside: KEEP THE PANEL WHOLE. A taller-
            # than-configured panel is a presentation choice; a blind cut
            # through the action is a corruption. The user can split it
            # deliberately in Manual Crop.
            log.info("no validated gutter inside oversized panel %s "
                     "(%dpx); keeping it whole", frag_id, y1 - y0)
            pieces.append(panel.model_copy(update={
                "id": frag_id,
                "split_of": panel.id if frag_id != panel.id else None,
                "y_start": y0,
                "y_end": y1,
                "image_file": f"panel_{frag_id}.png",
            }))
            continue
        stack.append((y0, row, frag_id + "a"))
        stack.append((row, y1, frag_id + "b"))
    pieces = sorted(pieces, key=lambda c: c.y_start)
    return pieces

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
        if a.y_end > b.y_start:
            merge_pt = (a.y_end + b.y_start) // 2
            log.warning("build_cuts: overlapping panels %d(y_end=%d) and %d(y_start=%d) "
                        "-> repairing to midpoint y=%d", a.panel_index, a.y_end,
                        b.panel_index, b.y_start, merge_pt)
            a.y_end = merge_pt
            b.y_start = merge_pt
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
            row = find_gutter_row(
                gray, center, tolerance=config.tolerance * 2,
                threshold=config.variance_threshold,
                edge_threshold=config.edge_threshold,
                use_edge_density=config.use_edge_density,
                forbidden=forbidden, require_threshold=True,
                min_gutter_run=config.min_gutter_run,
                blur_sigma=config.blur_sigma,
                variances=variances, edge_density=edge_density)
            if row is None:
                # Relaxed snap: colored/gradient gutters fail the strict
                # variance/edge thresholds but are still near-uniform runs.
                row = find_gutter_row(
                    gray, center, tolerance=config.tolerance * 2,
                    threshold=config.variance_threshold * 4,
                    edge_threshold=config.edge_threshold * 2,
                    use_edge_density=config.use_edge_density,
                    forbidden=forbidden, require_threshold=True,
                    min_gutter_run=config.min_gutter_run,
                    blur_sigma=config.blur_sigma,
                    variances=variances, edge_density=edge_density)
                if row is not None:
                    log.debug("relaxed snap for panels %d..%d at %d (+%dpx)",
                              a.panel_index, b.panel_index, row, abs(row - center))
            if row is None:
                if config.preserve_boundaries:
                    groups.append([b])
                    cut_rows.append(center)
                    snap_distances.append(0)
                    log.debug("no gutter at %d; boundary preserved "
                              "(structure-first)", center)
                else:
                    groups[-1].append(b)
                    log.debug("panel %d..%d merged (no gutter at %d)", a.panel_index, b.panel_index, center)
            else:
                groups.append([b])
                cut_rows.append(row)
                snap_distances.append(abs(row - center))
        else:
            groups.append([b])
            cut_rows.append(row)
            snap_distances.append(abs(row - center))

    tops = [entries[0].y_start] + cut_rows
    bottoms = cut_rows + [entries[-1].y_end]
    strip_h = gray.shape[0]
    first_top = tops[0]
    last_bottom = bottoms[-1]
    if first_top > 0:
        log.warning("build_cuts: first panel starts at y=%d, clamping top to 0 "
                    "(chop %dpx from strip top)", first_top, first_top)
    if last_bottom < strip_h:
        log.warning("build_cuts: last panel ends at y=%d, clamping bottom to %d "
                    "(dropped %dpx from strip bottom)", last_bottom, strip_h,
                    strip_h - last_bottom)
    tops[0] = 0
    bottoms[-1] = strip_h
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


def _apply_blank_regions(cuts: list[CutPanel],
                         regions: list[BlankRegion],
                         strip_height: int,
                         blank_score_threshold: float | None = None,
                         ) -> list[CutPanel]:
    """Shrink/remove cut panels that overlap deterministic blank regions.

    Rules (conservative; a panel is only dropped for verdict=blank):
    * a panel fully inside a blank region -> dropped (logged);
    * a panel partially blank -> trimmed to the content part; if both
      ends are blank the panel is split around the blank (kept as pieces);
    * verdict=suspicious regions never modify geometry — they only get a
      flag in blank_flag/blank_score so the review UI can surface them.
    """
    blanks = [r for r in regions if r.verdict == BLANK]
    if not blanks:
        # still record suspicious flags on the touching panels
        result: list[CutPanel] = []
        for c in cuts:
            for r in regions:
                if r.y_start < c.y_end and r.y_end > c.y_start:
                    c.blank_flag = "suspicious"
                    c.blank_score = max(c.blank_score, r.score)
                    break
            result.append(c)
        return result

    out: list[CutPanel] = []
    for c in cuts:
        overlaps = [r for r in blanks
                    if r.y_start < c.y_end and r.y_end > c.y_start]
        if not overlaps:
            out.append(c)
            continue
        # fraction of the panel covered by blank regions
        cover = sum(min(c.y_end, r.y_end) - max(c.y_start, r.y_start)
                    for r in overlaps)
        frac = cover / max(1, c.y_end - c.y_start)
        if frac >= 0.95:
            log.info("dropping panel %s: %.0f%% covered by blank region(s) "
                     "%s", c.id, frac * 100,
                     [f"{r.y_start}-{r.y_end}" for r in overlaps])
            continue
        # Trim blank bands off the panel's TOP/BOTTOM edges (keeping the
        # content). Blanks in the MIDDLE never change geometry - the panel
        # is flagged suspicious instead, because cutting a panel in two
        # (e.g. a deliberate white gap between two art halves) is riskier
        # than leaving it to human review.
        y0, y1 = c.y_start, c.y_end
        panel_h = y1 - y0
        slack = max(48, int(0.02 * panel_h))  # detector windowing slack
        min_keep = max(30, int(0.1 * panel_h))
        mid_blanks: list[BlankRegion] = []
        for r in sorted(overlaps, key=lambda r: r.y_start):
            top_blank = r.y_start <= y0 + slack
            bot_blank = r.y_end >= y1 - slack
            if top_blank and not bot_blank:
                y0 = max(y0, min(r.y_end, y1 - min_keep))
            elif bot_blank and not top_blank:
                y1 = min(y1, max(r.y_start, y0 + min_keep))
            elif top_blank and bot_blank:
                # blank spans the whole panel within slack; keep the middle
                y0 = max(y0, min(r.y_end, y1 - min_keep))
                y1 = min(y1, max(r.y_start, y0 + min_keep))
            else:
                mid_blanks.append(r)
        if mid_blanks:
            c.blank_flag = "suspicious"
            c.blank_score = max(c.blank_score, max(r.score for r in mid_blanks))
            log.info("panel %s overlaps a mid-panel blank region %s; kept "
                     "intact but flagged suspicious",
                     c.id, [f"{r.y_start}-{r.y_end}" for r in mid_blanks])
        if y1 - y0 < min(30, min_keep):
            log.info("dropping panel %s: content remainder %dpx too small "
                     "after blank trim", c.id, y1 - y0)
            continue
        if (y0, y1) != (c.y_start, c.y_end):
            log.info("trimmed panel %s from [%d,%d] to [%d,%d] "
                     "(blank regions removed)", c.id, c.y_start, c.y_end, y0, y1)
            c = c.model_copy(update={"y_start": y0, "y_end": y1})
        out.append(c)
    return out


def guided_cut(strip_path: str | Path, plan: PanelPlan, out_dir: str | Path,
               *, config: CutterConfig | None = None,
               force: bool = False,
               validate: bool = False) -> CutArtifact:
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

    _check_image_size(strip)
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

    # ------------------------------------------------------------------ #
    # DETERMINISTIC BLANK-REGION REMOVAL (NO AI)
    # First layer: blank regions detected on the strip shrink/exclude cut
    # ranges BEFORE cropping, so blank sections never become panels.
    # ------------------------------------------------------------------ #
    if config.blank_detection and _HAS_BLANK_DETECTOR:
        try:
            regions = detect_blank_regions(
                gray_arr, rgb=np.asarray(rgb),
                strip_height=height, strip_width=width,
                config=BlankDetectorConfig(preset=config.blank_preset))
            if regions:
                cuts = _apply_blank_regions(cuts, regions, height,
                                            blank_score_threshold=None)
        except Exception as exc:  # noqa: BLE001 - never kill a cut
            log.warning("blank-region detection failed (continuing without "
                        "it): %s", exc)

    # Post-segmentation validation layer (advisory; never deletes).
    if validate:
        try:
            from panel_validator import save_report, validate_panels
            vreport = validate_panels(
                gray_arr, cuts,
                ai_confidences={c.id: c.confidence for c in cuts})
            save_report(out, vreport)
        except Exception as exc:  # validation must never kill a cut
            log.warning("panel validation skipped: %s", exc)

    saved: list[CutPanel] = []
    min_panel_height = 30
    for c in cuts:
        y0 = max(0, c.y_start)
        y1 = min(height, c.y_end)
        if y1 <= y0:
            log.warning("cut panel %s has empty/negative range [%d,%d]; skipping",
                        c.id, c.y_start, c.y_end)
            continue
        if (y1 - y0) < min_panel_height:
            log.warning("cut panel %s is too thin (%dpx < %dpx); likely gutter "
                        "debris, skipping", c.id, y1 - y0, min_panel_height)
            continue
        piece = rgb.crop((0, y0, width, y1))

        # -------------------------------------------------------------- #
        # Second safety layer: post-crop blank scoring of the actual PNG.
        # A crop that is essentially uniform must not reach narration/
        # TTS/render; it is flagged (blank) or sent to review (suspicious).
        # -------------------------------------------------------------- #
        if config.blank_detection and _HAS_BLANK_DETECTOR:
            try:
                score, metrics = score_crop(np.asarray(piece))
                c.blank_score = score
                c.blank_reasons = [
                    f"{k}={v}" for k, v in metrics.items()]
                if score >= 0.90:
                    c.blank_flag = BLANK
                    log.info("panel %s scored BLANK (%.2f); dropping from "
                             "output (metrics: %s)", c.id, score, metrics)
                    continue  # never saved, never narrated
                if score >= 0.65:
                    c.blank_flag = "suspicious"
                    log.info("panel %s flagged suspicious blank score %.2f "
                             "(kept for review)", c.id, score)
            except Exception as exc:  # noqa: BLE001
                log.warning("post-crop blank scoring failed for %s: %s",
                            c.id, exc)

        dest = out / c.image_file
        if dest.exists() and not force:
            log.info("panel file already exists, skipping: %s", dest)
            saved.append(c)
            continue
        # Output-size policy: keep the full-resolution source crop for
        # geometry/scoring, then normalize ONLY the PNG written to disk to
        # exactly 390px wide with height in [760, 800]px. Source
        # coordinates (y_start/y_end, artifact width/height) are untouched.
        out_piece = piece
        if config.normalize_output:
            out_piece = normalize_panel_image(
                piece,
                output_width=config.output_width,
                min_output_height=config.min_output_height,
                max_output_height=config.max_output_height)
            c = c.model_copy(update={
                "output_width": out_piece.width,
                "output_height": out_piece.height})
        out_piece.save(dest, "PNG")
        log.debug("saved panel %s y=[%d,%d] source=%dx%d output=%dx%d",
                  c.id, y0, y1, piece.width, piece.height,
                  out_piece.width, out_piece.height)
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
