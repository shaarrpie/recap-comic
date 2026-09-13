# blank_detector.py
"""Deterministic (NO-AI) blank-region detection for manhwa strips.

Detects completely blank / nearly blank horizontal regions of a vertical
strip — white, black, gray, cream, pastel, or any nearly uniform colour —
using traditional image processing only:

    row variance  +  Sobel edge density  +  entropy  +  colour spread
    +  foreground occupancy (large-deviation pixel fraction)

Design rules (from the feature request):

* NEVER classify by brightness — a black blank and a white blank behave
  identically. All signals measure VARIATION and STRUCTURE, not level.
* A single weak signal is never enough: a region is BLANK only when
  several independent signals agree it is essentially structureless.
* Height thresholds are resolution-aware: the minimum blank height and
  the gutter/blank discriminator scale with the strip width so a 1600px
  phone-strip and an 800px webtoon are judged consistently.
* Small-but-meaningful content (a tiny character, one speech bubble,
  thin line art, sparkle effects) rescues a region: foreground occupancy
  is measured per sub-window so sparse content still registers, and
  uncertain regions are returned as SUSPICIOUS, never silently dropped.
* Fast: all metrics are row-wise / windowed vectorised statistics over
  a downscaled copy; the source image is never modified.

Scores
------
Each detected region carries ``blank_score`` in [0, 1] (1 = certainly
blank) and a human-readable ``reasons`` list, so every removal can be
explained and audited in the logs / review UI.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

# Quality-of-verdict constants used across the pipeline.
NORMAL = "normal"
SUSPICIOUS = "suspicious"
BLANK = "blank"


@dataclass(frozen=True)
class BlankDetectorConfig:
    """All thresholds in one place; every length is in px on the
    DOWNSCALED analysis grid unless the name says `orig`.

    Sensitivity presets exist so users do not need to touch raw CV
    numbers (see ``preset=`` on the detector):

    * conservative (default): a region must be very large AND very
      empty before BLANK; anything marginal -> SUSPICIOUS.
    * low: essentially never reports BLANK (review only).
    * high: trims more aggressively.
    """
    # --- analysis grid -------------------------------------------------
    analysis_width: int = 320    # downscale so metrics are resolution-independent
    window: int = 16             # row-window size (px, downscaled grid)

    # --- blank-run detection ------------------------------------------
    min_blank_height_frac: float = 0.04   # of strip height (min blank section)
    min_blank_height_px: int = 24         # absolute floor on the downscaled grid
    max_gutter_height_frac: float = 0.10  # taller than this = NOT a gutter
    merge_gap_windows: int = 2            # noisy windows tolerated inside a run

    # --- signal thresholds (downscaled grid units) --------------------
    max_variance: float = 12.0      # per-row-window grayscale variance
    max_edge_density: float = 6.0   # mean |Sobel| per window row
    max_entropy: float = 2.2        # window histogram entropy (bits; 8 = full)
    max_color_spread: float = 14.0  # per-window std of mean-RGB distance to mode
    min_fg_ratio: float = 0.02      # LARGE-deviation pixel fraction for "content"

    # --- scoring ---------------------------------------------------------
    # weights must sum to 1.0; blank_score = weighted mean of signal scores
    weights: tuple[float, float, float, float, float] = (
        0.25,  # variance
        0.25,  # edge density
        0.20,  # entropy
        0.15,  # colour spread
        0.15,  # foreground
    )
    blank_score_threshold: float = 0.90    # >= this AND empty -> BLANK
    suspicious_score_threshold: float = 0.65
    # content rescue: if any sub-window has fg_ratio above this, the
    # region has meaningful sparse content (bubble/character) -> not blank
    content_fg_ratio: float = 0.05
    content_area_frac: float = 0.02        # >=2% of windows with content

    # --- multi-scale -----------------------------------------------------
    # A second pass on a further-downscaled grid rejects regions that only
    # look empty because fine compression noise was below the first grid's
    # variance threshold but is still structure.
    multiscale: bool = True
    multiscale_factor: float = 0.5

    preset: str = "conservative"


_PRESETS: dict[str, dict[str, float]] = {
    # preset: (min_blank_height_frac, blank_score_threshold, content_fg_ratio)
    "low": {"min_blank_height_frac": 0.08, "blank_score_threshold": 0.95,
            "content_fg_ratio": 0.03},
    "conservative": {},  # defaults above
    "high": {"min_blank_height_frac": 0.025, "blank_score_threshold": 0.82,
             "content_fg_ratio": 0.07},
}


def _apply_preset(cfg: BlankDetectorConfig) -> BlankDetectorConfig:
    over = _PRESETS.get((cfg.preset or "conservative").lower())
    if not over:
        return cfg
    return BlankDetectorConfig(**{**cfg.__dict__, **over})


@dataclass
class BlankRegion:
    """One detected blank candidate in ORIGINAL strip coordinates."""
    y_start: int                       # original-strip pixel rows
    y_end: int
    height: int
    score: float                       # 0..1, higher = more certainly blank
    verdict: str                        # "blank" | "suspicious" | "normal"
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def log_line(self) -> str:
        m = self.metrics
        return (f"Y {self.y_start}-{self.y_end} (h={self.height}) "
                f"score={self.score:.2f} verdict={self.verdict} "
                f"var={m.get('variance', 0):.1f} "
                f"edge={m.get('edge_density', 0):.1f} "
                f"entropy={m.get('entropy', 0):.2f} "
                f"fg={m.get('fg_ratio', 0):.3f}")


# --------------------------------------------------------------------------- #
# metric helpers (all operate on the downscaled grayscale/RGB grid)
# --------------------------------------------------------------------------- #
def _downscale(gray: np.ndarray, rgb: np.ndarray | None,
               target_w: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Area-interpolation downscale so statistics are cheap and
    resolution-independent. Returns (small_gray, small_rgb|None)."""
    import cv2
    h, w = gray.shape[:2]
    if w <= target_w:
        return gray, rgb
    scale = target_w / w
    tw, th = target_w, max(1, round(h * scale))
    small = cv2.resize(gray, (tw, th), interpolation=cv2.INTER_AREA)
    srgb = None
    if rgb is not None:
        srgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
    return small, srgb


def _window_rows(v: np.ndarray, window: int) -> np.ndarray:
    """Mean of `v` over consecutive non-overlapping windows of rows."""
    n = len(v) // window * window
    if n == 0:
        return np.array([v.mean()] if len(v) else np.empty(0))
    trimmed = v[:n]
    return trimmed.reshape(-1, window).mean(axis=1)


def _window_entropy(small: np.ndarray, window: int, bins: int = 64) -> np.ndarray:
    """Per-row-window histogram entropy of the downscaled grayscale."""
    rows = small.shape[0] // window * window
    if rows == 0:
        rows = small.shape[0]
        window = max(1, rows)
    out = np.zeros(max(1, rows // window), dtype=np.float32)
    for i in range(len(out)):
        block = small[i * window:(i + 1) * window]
        hist, _ = np.histogram(block, bins=bins, range=(0, 256))
        p = hist / max(1, hist.sum())
        p = p[p > 0]
        out[i] = float(-(p * np.log2(p)).sum())
    return out


def _window_color_spread(srgb: np.ndarray | None, window: int) -> np.ndarray | None:
    """Per-window std of pixel distance to the window's modal colour.

    Captures colour variation that grayscale misses (e.g. pale colour
    washes with matching luminance). None when no RGB available.
    """
    if srgb is None:
        return None
    rows = srgb.shape[0] // window * window
    if rows == 0:
        rows = srgb.shape[0]
    n = max(1, rows // window)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        block = srgb[i * window:(i + 1) * window].reshape(-1, 3)
        if block.shape[0] == 0:
            continue
        med = np.median(block, axis=0)
        out[i] = float(np.sqrt(((block - med) ** 2).sum(axis=1)).std())
    return out


def _fg_ratio_windows(small: np.ndarray, window: int) -> np.ndarray:
    """Per-window fraction of pixels deviating strongly from THAT window's
    median (foreground occupancy). Per-window (not global) median is
    essential: in a mixed strip a white region deviates from the global
    median even though it is perfectly uniform. Sparse-but-real content
    (thin line character, small bubble) still produces a clearly non-zero
    ratio, unlike variance which it barely moves."""
    med = np.median(small, axis=1, keepdims=True)
    absdev = np.abs(small.astype(np.float32) - med.astype(np.float32))
    hot = (absdev > 28).astype(np.float32)
    return _window_rows(hot.mean(axis=1), window)


def _score_from_signals(var: float, edge: float, ent: float,
                        color: float, fg: float,
                        cfg: BlankDetectorConfig) -> float:
    """Weighted 0..1 blank score; each signal contributes in proportion
    to how far below its threshold it is (clamped to [0, 1])."""
    def closeness(value: float, threshold: float) -> float:
        if threshold <= 0:
            return 0.0
        return float(np.clip(1.0 - value / threshold, 0.0, 1.0))

    w_var, w_edge, w_ent, w_col, w_fg = cfg.weights
    return (w_var * closeness(var, cfg.max_variance)
            + w_edge * closeness(edge, cfg.max_edge_density)
            + w_ent * closeness(ent, cfg.max_entropy)
            + w_col * closeness(color, cfg.max_color_spread)
            + w_fg * closeness(fg, cfg.min_fg_ratio * 10))  # fg scaled: 0..~10%


# --------------------------------------------------------------------------- #
# detector
# --------------------------------------------------------------------------- #
def detect_blank_regions(
    gray: np.ndarray,
    rgb: np.ndarray | None = None,
    strip_height: int | None = None,
    strip_width: int | None = None,
    config: BlankDetectorConfig | None = None,
) -> list[BlankRegion]:
    """Detect blank/nearly-blank horizontal regions of a vertical strip.

    Parameters
    ----------
    gray : np.ndarray
        Full-resolution grayscale (H, W) uint8 of the strip. NOT modified.
    rgb : np.ndarray | None
        Optional full-resolution RGB (H, W, 3) uint8 for colour signals.
    strip_height / strip_width : int
        Size of the ORIGINAL strip (== gray.shape when omitted).
    config : BlankDetectorConfig | None
        Thresholds; None -> defaults (conservative preset).

    Returns
    -------
    list[BlankRegion]
        Candidate blank regions in ORIGINAL strip coordinates, sorted by
        y_start, each with score/verdict/reasons/metrics. Gutter-sized
        uniform runs are NOT returned (they are separators, not blanks).

    Determinism: no AI, no OCR, no randomness — same input always yields
    the same output.
    """
    cfg = _apply_preset(config or BlankDetectorConfig())
    h_orig = strip_height if strip_height is not None else gray.shape[0]

    small, srgb = _downscale(gray, rgb, cfg.analysis_width)
    h_small = small.shape[0]
    if h_small < 8:  # degenerate strip; nothing to analyse
        return []
    scale = h_orig / h_small  # downscaled row -> original row

    # --- per-window signal arrays ---------------------------------------
    var_w = _window_rows(small.astype(np.float32).var(axis=1), cfg.window)
    import cv2
    sobel = np.abs(cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3))
    edge_w = _window_rows(sobel.mean(axis=1), cfg.window)
    ent_w = _window_entropy(small, cfg.window)
    col_w = _window_color_spread(srgb, cfg.window)
    fg_w = _fg_ratio_windows(small, cfg.window)
    col_eff = col_w if col_w is not None else np.zeros_like(var_w)
    n_windows = len(var_w)
    if n_windows == 0:
        return []

    # --- blank-run mask ---------------------------------------------------
    # A window is "empty" when variance AND edges AND entropy all say so.
    empty = ((var_w <= cfg.max_variance)
             & (edge_w <= cfg.max_edge_density)
             & (ent_w <= cfg.max_entropy))
    # Tolerance for isolated noisy windows inside a run (morphological
    # closing over the run-length structure, implemented via runs).
    runs: list[tuple[int, int]] = []  # (start_window, end_window) inclusive
    start = None
    gaps = 0
    for i in range(n_windows):
        if empty[i]:
            if start is None:
                start = i
            gaps = 0
        elif start is not None:
            gaps += 1
            if gaps > cfg.merge_gap_windows:
                runs.append((start, i - gaps))
                start = None
    if start is not None:
        runs.append((start, n_windows - 1))

    # --- min-height + gutter discrimination -------------------------------
    min_h_orig = max(cfg.min_blank_height_px,
                     int(cfg.min_blank_height_frac * h_orig))
    max_gutter_h_orig = int(cfg.max_gutter_height_frac * h_orig)

    regions: list[BlankRegion] = []
    for w0, w1 in runs:
        y0 = int(w0 * cfg.window * scale)
        y1 = int((w1 + 1) * cfg.window * scale)
        y1 = min(y1, h_orig)
        height = y1 - y0
        if height < min_h_orig:
            continue  # tiny quiet gap: gutter debris, not a blank section

        # aggregate signals over the run
        var = float(var_w[w0:w1 + 1].mean())
        edge = float(edge_w[w0:w1 + 1].mean())
        ent = float(ent_w[w0:w1 + 1].mean())
        col = float(col_eff[w0:w1 + 1].mean())
        fg = float(fg_w[w0:w1 + 1].mean())

        # content rescue: any sub-window with clear foreground (line art,
        # bubble, character) inside the run => uncertain, never BLANK.
        content_windows = int((fg_w[w0:w1 + 1] >= cfg.content_fg_ratio).sum())
        content_frac = content_windows / max(1, (w1 - w0 + 1))
        reasons: list[str] = []
        if var <= cfg.max_variance:
            reasons.append("low variance")
        if edge <= cfg.max_edge_density:
            reasons.append("low edge density")
        if ent <= cfg.max_entropy:
            reasons.append("low entropy")
        if col <= cfg.max_color_spread:
            reasons.append("low colour spread")
        if fg < cfg.min_fg_ratio:
            reasons.append("low foreground occupancy")
        if content_frac >= cfg.content_area_frac:
            reasons.append(f"sparse content in {content_frac:.0%} of windows")

        score = _score_from_signals(var, edge, ent, col, fg, cfg)

        # Multi-scale confirmation: re-measure on a coarser grid; a truly
        # blank region stays empty, compression-noise "blanks" fall apart.
        if cfg.multiscale and score >= cfg.suspicious_score_threshold:
            sub = small[int(w0 * cfg.window):int((w1 + 1) * cfg.window)]
            if sub.size:
                f = cfg.multiscale_factor
                sub2 = sub if sub.shape[0] < 4 else sub[::max(1, int(1 / f))]
                var2 = float(sub2.astype(np.float32).var())
                ent2 = _window_entropy(sub2, max(2, sub2.shape[0]))
                if var2 > cfg.max_variance * 2 or (ent2.size and ent2[0] > cfg.max_entropy):
                    score *= 0.5
                    reasons.append("failed multiscale check (structure visible "
                                   "at coarse scale)")

        verdict = SUSPICIOUS
        if (score >= cfg.blank_score_threshold
                and content_frac < cfg.content_area_frac
                and height >= min_h_orig):
            verdict = BLANK
            reasons.append(f"blank run height {height}px >= min {min_h_orig}px")
        elif score < cfg.suspicious_score_threshold:
            verdict = NORMAL
            continue

        is_gutter_sized = height <= max_gutter_h_orig
        if is_gutter_sized and verdict == BLANK:
            # a *tall* uniform run is a blank section; a *short* one is a
            # normal panel separator -> keep out of the blank list, the
            # gutter detector already handles those boundaries.
            verdict = NORMAL
            continue

        regions.append(BlankRegion(
            y_start=y0, y_end=y1, height=height, score=round(score, 3),
            verdict=verdict, reasons=reasons,
            metrics={"variance": round(var, 2),
                     "edge_density": round(edge, 2),
                     "entropy": round(ent, 2),
                     "color_spread": round(col, 2),
                     "fg_ratio": round(fg, 4)}))

    for r in regions:
        if r.verdict == BLANK:
            log.info("blank region detected %s reasons=%s", r.log_line(),
                     "; ".join(r.reasons))
        else:
            log.debug("suspicious region %s", r.log_line())
    return regions


def blank_rows_mask(regions: list[BlankRegion], height: int,
                    verdicts: tuple[str, ...] = (BLANK,)) -> np.ndarray:
    """Boolean (height,) mask: True where a row belongs to a blank region
    with one of `verdicts`. Used to exclude rows from cut-row searches
    and to shrink oversized-panel split candidates."""
    mask = np.zeros(height, dtype=bool)
    for r in regions:
        if r.verdict in verdicts:
            mask[max(0, r.y_start):min(height, r.y_end)] = True
    return mask


def score_crop(rgb: np.ndarray, config: BlankDetectorConfig | None = None,
               ) -> tuple[float, dict[str, float]]:
    """Post-crop blank score for ONE panel image (H, W, 3).

    Returns (blank_score, metrics). Same multi-signal philosophy as the
    row detector, evaluated over the whole crop; used by the validator as
    the second safety layer so a blank crop can never reach narration/
    TTS/render without a flag.
    """
    import cv2
    cfg = _apply_preset(config or BlankDetectorConfig())
    if rgb.ndim != 3 or rgb.shape[0] == 0 or rgb.shape[1] == 0:
        return 1.0, {"variance": 0.0, "edge_density": 0.0,
                      "entropy": 0.0, "fg_ratio": 0.0}
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    scale = cfg.analysis_width / rgb.shape[1]
    if scale < 1.0:
        gray = cv2.resize(gray, (cfg.analysis_width,
                                 max(1, int(rgb.shape[0] * scale))),
                          interpolation=cv2.INTER_AREA)
    var = float(gray.astype(np.float32).var())
    edge = float(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).mean())
    n_win = max(1, gray.shape[0] // 4)
    ent_arr = _window_entropy(gray, max(8, n_win))
    ent = float(ent_arr.mean()) if ent_arr.size else 0.0
    fg = _fg_ratio_windows(gray, max(8, n_win))
    fg = float(fg.mean()) if fg.size else 0.0
    med = np.median(rgb.reshape(-1, 3), axis=0)
    col = float(np.sqrt(((rgb.reshape(-1, 3).astype(np.float32) - med) ** 2)
                        .sum(axis=1)).std())
    score = _score_from_signals(var, edge, ent, col, fg, cfg)
    return round(score, 3), {"variance": round(var, 2),
                             "edge_density": round(edge, 2),
                             "entropy": round(ent, 2),
                             "color_spread": round(col, 2),
                             "fg_ratio": round(fg, 4)}
