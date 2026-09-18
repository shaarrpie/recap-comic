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
    Base 0.3 for normal framing preserves ordering (out < normal < in)
    without inventing pull-back behavior the blur chain cannot express.
    """
    return float(min(0.8, max(0.0, 0.3 + (seg.zoom - 1.0) * 2.0)))


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

    - 1:1 when counts match (seg i -> shot i).
    - More panels than segments: cycle the pattern (preserves rhythm).
    - Fewer panels: even sampling across the pattern (first..last) instead
      of cutting important shots randomly.
    """
    s = len(preset)
    if n_panels <= 0 or s == 0:
        return []
    if n_panels == s:
        return list(range(s))
    if n_panels > s:
        return [i % s for i in range(n_panels)]
    if n_panels == 1:
        # Single panel: hold the middle of the pattern (a representative
        # vertical move) rather than the opening frame.
        return [s // 2]
    return [round(i * (s - 1) / (n_panels - 1)) for i in range(n_panels)]


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
        scaled_w = max(1, _math.ceil(png_w * min(canvas_w / png_w,
                                                canvas_h / png_h)))
        scaled_h = max(1, _math.ceil(png_h * min(canvas_w / png_w,
                                                canvas_h / png_h)))
        if min(canvas_w / png_w, canvas_h / png_h) > 4.0:
            scaled_w = max(1, _math.ceil(png_w * 4.0))
            scaled_h = max(1, _math.ceil(png_h * 4.0))
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
        "zoom_strength": zoom_strength_for_seg(seg),
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
