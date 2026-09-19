# motion_presets.py
"""Reference-video motion preset: normalized motion template layer (Phase 3 only).

The reference measurements are a *motion blueprint*, not pixel coordinates to
copy. This module adapts them to any panel/canvas size:

  normalized_dx = dx / reference_width
  normalized_dy = dy / reference_height

then scales by the render canvas and clamps to the panel's safe movement
range so no empty/black area is ever revealed.

Design rules (from the feature request):
- Never random: zoom/pan/direction come from the preset segment only.
- Preserve segment order + relative rhythm (weights, not absolute seconds;
  the caller scales to the measured narration total).
- Static vs moving via speed threshold on (dxps, dyps).
- Low matchScore damps travel distance only (never discards the segment).
- Works for both render paths: blur-background (overlay pan + per-panel
  push-in strength) and plain cover-crop (directional pan / zoom kind).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PRESET_PATH = Path(__file__).resolve().parent / "reference_motion_preset.json"

PRESET_NONE = "none"
PRESET_REFERENCE = "reference"
VALID_PRESETS = (PRESET_NONE, PRESET_REFERENCE)

# Tall/long panels read "too far away" when a contain-fit leaves them as a
# thin vertical strip (the artwork covers only a small fraction of the canvas
# width). Bring those closer with a fixed scale bump and reveal the extra
# top/bottom crop with a vertical pan only (never a zoom animation). Only
# genuinely narrow art triggers this -- a height-constrained panel whose
# contain-fit covers less than ``TALL_PANEL_WIDTH_FILL`` of the canvas width
# -- so wide/near-square panels and portrait renders stay byte-for-byte the
# same as before.
TALL_PANEL_SCALE = 1.2
TALL_PANEL_WIDTH_FILL = 0.5

# A "super tall" strip (aspect height/width >= SPLIT_MIN_ASPECT) is too thin
# to read even after the 1.2x bump. Split it into SPLIT_COLUMNS horizontal
# bands laid out side by side: the whole panel then shows at once at roughly
# SPLIT_COLUMNS x closer, with no long pan needed (fits a narration-length
# shot). Moderately tall panels keep the 1.2x + vertical-pan treatment.
SPLIT_MIN_ASPECT = 3.0
SPLIT_COLUMNS = 2

# Split-panel layout/animation knobs (surfaced in the resolve result so the
# renderer reads them from ``motion`` as the single source of truth).
# SPLIT_GAP_FRAC: blank (blurred-bg) gap between the side-by-side bands, as a
# fraction of canvas width. SPLIT_PAN_FRAC: extra height each band is scaled
# to so it can slowly pan vertically (bands drift in opposite directions).
SPLIT_GAP_FRAC = 0.05
SPLIT_PAN_FRAC = 0.15
# Vertical supersampling factor for the band pan: the crop pans in an image
# this many times taller, then downscales, so integer crop steps become
# sub-pixel final motion -> smooth, jitter-free pan. Higher = smoother but
# costlier; 3 is a good balance for slow narration-length pans.
SPLIT_SUPERSAMPLE = 3.0


@dataclass
class ReferenceSegment:
    seg: int
    t0: float
    t1: float
    dur: float
    zoom: float
    matchScore: float  # noqa: N815 - mirrors the reference JSON key verbatim
    dx: float
    dy: float
    dxps: float
    dyps: float


@dataclass
class MotionPreset:
    name: str
    reference_width: float
    reference_height: float
    static_speed_threshold: float = 10.0
    low_confidence_threshold: float = 0.85
    low_confidence_damping: float = 0.6
    segments: list[ReferenceSegment] | None = None

    def __post_init__(self) -> None:
        if self.segments is None:
            self.segments = []
        if self.reference_width <= 0 or self.reference_height <= 0:
            raise ValueError("reference dimensions must be positive")

    @property
    def total_duration(self) -> float:
        return sum(s.dur for s in (self.segments or []))

    def __len__(self) -> int:
        return len(self.segments or [])


def load_preset(path: str | Path | None = None) -> MotionPreset:
    """Load + validate the JSON motion template."""
    p = Path(path) if path else DEFAULT_PRESET_PATH
    data = json.loads(p.read_text("utf-8"))
    segs = [ReferenceSegment(seg=int(s["seg"]), t0=float(s["t0"]),
                             t1=float(s["t1"]), dur=float(s["dur"]),
                             zoom=float(s["zoom"]),
                             matchScore=float(s["matchScore"]),
                             dx=float(s["dx"]), dy=float(s["dy"]),
                             dxps=float(s["dxps"]), dyps=float(s["dyps"]))
            for s in data.get("segments", [])]
    if not segs:
        raise ValueError(f"motion preset {p} contains no segments")
    # Order contract: exact reference sequence must be preserved.
    segs.sort(key=lambda s: s.seg)
    return MotionPreset(
        name=str(data.get("name", "reference_v1")),
        reference_width=float(data.get("reference_width", 1312)),
        reference_height=float(data.get("reference_height", 2332)),
        static_speed_threshold=float(data.get("static_speed_threshold_px_per_s", 5.0)),
        low_confidence_threshold=float(data.get("low_confidence_threshold", 0.85)),
        low_confidence_damping=float(data.get("low_confidence_damping", 0.6)),
        segments=segs,
    )


def normalized_movement(seg: ReferenceSegment,
                        preset: MotionPreset) -> dict[str, float]:
    """Reference px -> normalized (resolution-independent) units."""
    return {
        "ndx": seg.dx / preset.reference_width,
        "ndy": seg.dy / preset.reference_height,
        "ndxps": seg.dxps / preset.reference_width,
        "ndyps": seg.dyps / preset.reference_height,
    }


def is_static(seg: ReferenceSegment, preset: MotionPreset) -> bool:
    """Speed close to zero => zoom-only shot, no panning."""
    speed = math.hypot(seg.dxps, seg.dyps)
    return speed < preset.static_speed_threshold


def confidence_damping(seg: ReferenceSegment, preset: MotionPreset) -> float:
    """High-confidence segs reproduce closely; low-confidence follow the same
    motion but damped so uncertain measurements are not absolute truth."""
    if seg.matchScore >= 0.95:
        return 1.0
    if seg.matchScore >= preset.low_confidence_threshold:
        return 0.8
    return preset.low_confidence_damping


def zoom_strength_for_seg(seg: ReferenceSegment) -> float:
    """Map reference zoom to the blur-foreground push-in strength.

    Reference: 1.00 normal, 0.92-0.93 slightly out, 1.13-1.14 in.
    Base 0.15 for normal framing preserves ordering (out < normal < in)
    without inventing pull-back behavior the blur chain cannot express.

    The multiplier is deliberately small (0.5) and the cap low (0.4) so
    push-ins/pull-outs stay slow and cinematic: a 1.14 reference zoom yields
    only a ~0.22 push-in. Tune by editing the base/multiplier/cap here
    rather than the per-segment zoom values.
    """
    return float(min(0.4, max(0.0, 0.15 + (seg.zoom - 1.0) * 0.5)))


def pan_kind_for_seg(seg: ReferenceSegment, preset: MotionPreset) -> str:
    """Directional kind from the dominant movement axis.

    Sign convention: +dx = right, +dy = down, -dy = up, -dx = left.
    Static (slow) segs become zoom_in / zoom_out / static by zoom value.
    """
    if is_static(seg, preset):
        if seg.zoom >= 1.1:
            return "zoom_in"
        if seg.zoom <= 0.95:
            return "zoom_out"
        return "static"
    if abs(seg.dy) >= abs(seg.dx):
        return "pan_down" if seg.dy > 0 else "pan_up"
    return "pan_right" if seg.dx > 0 else "pan_left"


def map_segments_to_panels(n_panels: int, preset: MotionPreset) -> list[int]:
    """Reference segment index (0-based) for each generated panel in order.

    STRICT sequential cycle: the pattern always starts at the first segment
    (beat 1) and repeats continuously — panel i uses segment ``i % s``. With
    the shipped 4-beat preset this yields the fixed camera rhythm
    zoom_in -> pan_down -> pan_up -> zoom_out, repeated for as many panels
    as there are (and truncated in order for fewer panels than beats). No
    even-sampling or mid-pattern start is used, so the sequence is never
    reordered or begun off-beat.
    """
    s = len(preset)
    if n_panels <= 0 or s == 0:
        return []
    return [i % s for i in range(n_panels)]


def resolve_for_panel(*, png_w: int, png_h: int, canvas_w: int, canvas_h: int,
                      seg: ReferenceSegment, preset: MotionPreset,
                      motion_strength: float = 1.0,
                      blur_background: bool = True) -> dict:
    """Adapt one reference segment to one concrete panel.

    Returns a debug/render dict with normalized movement, clamped canvas-px
    pan travels, PanSpec kind + cover geometry, and per-panel zoom strength
    for the blur foreground. Never proposes travel outside the safe range.
    """
    if png_w <= 0 or png_h <= 0:
        raise ValueError("panel must have positive size")
    norm = normalized_movement(seg, preset)
    damp = confidence_damping(seg, preset)
    strength = max(0.0, float(motion_strength)) * damp

    # Desired canvas-px travel (direction + relative amount preserved).
    desired_dx = norm["ndx"] * canvas_w * strength
    desired_dy = norm["ndy"] * canvas_h * strength

    # Per-panel push/pull strength for the blur foreground; overridden to 0
    # for tall/long panels (they reveal via pan only, never zoom).
    zs = zoom_strength_for_seg(seg)
    tall_panel = False
    panel_scale = 1.0
    split_columns = 0
    # Band-pan direction for split panels: band 0 pans (split_dir), band 1
    # pans (-split_dir). recap_video.build_ref_timeline flips this across
    # consecutive split panels; the default here is the first split panel.
    split_dir = 1

    # Cover geometry (same convention as recap_video.compute_pan): both dims
    # cover the canvas, capped at 4x, so clamping below can never go negative
    # for real panels and no empty area is ever revealed.
    scale = max(canvas_w / png_w, canvas_h / png_h)
    if scale > 4.0:
        scale = 4.0
    if blur_background:
        # Blur path: contain-fit floats on a full-frame blurred copy, so the
        # background always fills the frame. Pan travels via the overlay;
        # clamp to a conservative fraction of the canvas (the measured
        # reference never exceeds ~6% of the frame) to avoid excessive crop.
        import math as _math
        contain = min(canvas_w / png_w, canvas_h / png_h)
        if contain > 4.0:
            contain = 4.0
        # A "far away" panel is a thin strip after the contain-fit: its width
        # covers only a small slice of the canvas. Only height-constrained art
        # can be that narrow, so this leaves wide/portrait-filling panels on
        # the unchanged generic path.
        strip_fill = (png_w * contain) / canvas_w if canvas_w else 1.0
        if strip_fill < TALL_PANEL_WIDTH_FILL:
            tall_panel = True
            zs = 0.0
            if png_h / png_w >= SPLIT_MIN_ASPECT:
                # Super-tall strip: split into side-by-side bands so the whole
                # panel reads ~SPLIT_COLUMNS x closer at once (static, no pan).
                split_columns = SPLIT_COLUMNS
                panel_scale = 1.0
                kind = "static"
                travel_px = 0
                pan_x = pan_y = 0.0
                comp_w = png_w * split_columns
                comp_h = png_h / split_columns
                cc = min(canvas_w / comp_w, canvas_h / comp_h)
                if cc > 4.0:
                    cc = 4.0
                scaled_w = max(1, _math.ceil(comp_w * cc))
                scaled_h = max(1, _math.ceil(comp_h * cc))
            else:
                # Moderately tall: bring closer with a fixed scale bump and
                # reveal the extra top/bottom crop with a slow vertical pan
                # only (no zoom). Blurred side pillars stay.
                panel_scale = TALL_PANEL_SCALE
                # Alternate sweep direction across the 4-beat cycle so
                # consecutive tall panels do not repeat the same move.
                kind = "pan_down" if seg.seg % 2 == 0 else "pan_up"
                travel_px = int(round((panel_scale - 1.0) * canvas_h))
                pan_y = _math.copysign(
                    travel_px, 1.0 if kind == "pan_down" else -1.0)
                pan_x = 0.0
                scaled_w = max(1, _math.ceil(png_w * contain * panel_scale))
                scaled_h = max(1, _math.ceil(png_h * contain * panel_scale))
        else:
            max_dx = canvas_w * 0.08
            max_dy = canvas_h * 0.08
            pan_x = _math.copysign(min(abs(desired_dx), max_dx),
                                   desired_dx) if desired_dx else 0.0
            pan_y = _math.copysign(min(abs(desired_dy), max_dy),
                                   desired_dy) if desired_dy else 0.0
            if is_static(seg, preset):
                pan_x, pan_y = 0.0, 0.0
            # PanSpec stays meaningful for duration floors + editor display:
            # reuse the directional kind, with travel_px = dominant-axis px.
            kind = pan_kind_for_seg(seg, preset)
            travel_px = int(round(max(abs(pan_x), abs(pan_y))))
            scaled_w = max(1, _math.ceil(png_w * contain))
            scaled_h = max(1, _math.ceil(png_h * contain))
    else:
        import math as _math
        scaled_w = _math.ceil(png_w * scale)
        scaled_h = _math.ceil(png_h * scale)
        over_w = scaled_w - canvas_w
        over_h = scaled_h - canvas_h
        kind = pan_kind_for_seg(seg, preset)
        if kind in ("pan_down", "pan_up"):
            travel = min(abs(desired_dy), max(0, over_h - 2))
            # Static zoom segs must not pan: travel stays 0 via is_static.
            if is_static(seg, preset):
                travel = 0
                kind = pan_kind_for_seg(seg, preset)  # zoom_in/out/static
                # Re-derive: static kinds carry no travel.
                travel_px = 0
                pan_x, pan_y = 0.0, 0.0
            else:
                travel_px = int(round(travel))
                pan_x, pan_y = 0.0, _math.copysign(travel_px, desired_dy)
                if travel_px <= 2:
                    # Overflow too small to read as a pan: hold as zoom/static.
                    kind = ("zoom_in" if seg.zoom >= 1.1
                            else "zoom_out" if seg.zoom <= 0.95 else "static")
                    travel_px = 0
                    pan_x, pan_y = 0.0, 0.0
        elif kind in ("pan_right", "pan_left"):
            travel = min(abs(desired_dx), max(0, over_w - 2))
            if is_static(seg, preset):
                travel_px = 0
                kind = pan_kind_for_seg(seg, preset)
                pan_x, pan_y = 0.0, 0.0
            else:
                travel_px = int(round(travel))
                pan_x, pan_y = _math.copysign(travel_px, desired_dx), 0.0
                if travel_px <= 2:
                    kind = ("zoom_in" if seg.zoom >= 1.1
                            else "zoom_out" if seg.zoom <= 0.95 else "static")
                    travel_px = 0
                    pan_x, pan_y = 0.0, 0.0
        else:
            travel_px = 0
            pan_x, pan_y = 0.0, 0.0
        # Zoom margin for combined cases (e.g. seg 8 zoom_in + left drift):
        # tighten the crop slightly so a directional pan still reads tighter
        # than a normal pan without needing a new filter expression.
        if kind.startswith("pan_") and seg.zoom >= 1.1:
            extra = min(0.25, (seg.zoom - 1.0) * 0.5)
            scaled_w = _math.ceil(scaled_w * (1.0 + extra))
            scaled_h = _math.ceil(scaled_h * (1.0 + extra))
            over_w = scaled_w - canvas_w
            over_h = scaled_h - canvas_h
            if kind in ("pan_down", "pan_up"):
                travel_px = int(round(min(abs(desired_dy), max(0, over_h - 2))))
                pan_y = _math.copysign(travel_px, desired_dy) if travel_px else 0.0
            else:
                travel_px = int(round(min(abs(desired_dx), max(0, over_w - 2))))
                pan_x = _math.copysign(travel_px, desired_dx) if travel_px else 0.0

    return {
        "seg": seg.seg,
        "dur": seg.dur,
        "zoom": seg.zoom,
        "zoom_strength": zs,
        "dx": seg.dx,
        "dy": seg.dy,
        "dxps": seg.dxps,
        "dyps": seg.dyps,
        "ndx": norm["ndx"],
        "ndy": norm["ndy"],
        "ndxps": norm["ndxps"],
        "ndyps": norm["ndyps"],
        "matchScore": seg.matchScore,
        "damping": damp,
        "static": is_static(seg, preset),
        "kind": kind,
        "scaled_w": int(scaled_w),
        "scaled_h": int(scaled_h),
        "travel_px": int(travel_px),
        "pan_x_px": float(pan_x),
        "pan_y_px": float(pan_y),
        "tall_panel": bool(tall_panel),
        "panel_scale": float(panel_scale),
        "split_columns": int(split_columns),
        "split_dir": int(split_dir),
        "split_gap_frac": float(SPLIT_GAP_FRAC),
        "split_pan_frac": float(SPLIT_PAN_FRAC),
        "split_ss": float(SPLIT_SUPERSAMPLE),
    }


def rhythm_weights(preset: MotionPreset,
                   seg_indices: list[int]) -> list[float]:
    """Relative duration weights for mapped segments (sum-normalized)."""
    segs = preset.segments or []
    durs = [segs[i].dur for i in seg_indices]
    total = sum(durs) or 1.0
    return [d / total for d in durs]


def preview_table(preset: MotionPreset,
                  seg_indices: list[int] | None = None) -> str:
    """Human-readable camera plan (CLI --motion-report / dry-run log)."""
    segs = preset.segments or []
    idx = seg_indices if seg_indices is not None else list(range(len(segs)))
    lines = ["seg  dur   zoom  dx/dy        speed(px/s)   conf   kind",
             "--- ------ ----- ------------ ------------- ------ ----------"]
    for i in idx:
        s = segs[i]
        lines.append(
            f"{s.seg:>3d} {s.dur:5.2f}s {s.zoom:4.2f} "
            f"{s.dx:+6.1f}/{s.dy:+7.1f} {s.dxps:+6.1f}/{s.dyps:+6.1f} "
            f"{s.matchScore:5.3f}  {pan_kind_for_seg(s, preset)}")
    return "\n".join(lines)
