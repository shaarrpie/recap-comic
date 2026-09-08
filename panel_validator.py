# panel_validator.py
"""Post-segmentation panel validation layer.

Position in the pipeline (never deletes anything itself):

    RAW detections (guided_cutter / fallback detector)
            |
    validate_panels(gray, cuts)  ->  PanelReport
            |
    NORMAL panels  +  SUSPICIOUS (needs review)  +  INVALID (blocked)

User decisions (review.json) are authoritative downstream; the AI baseline
(panels.json) is preserved untouched.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("panel_validator")

NORMAL, SUSPICIOUS, INVALID = "normal", "suspicious", "invalid"


@dataclass(frozen=True)
class ValidationConfig:
    """ALL thresholds in one place. Conservative on purpose:
    only very strong signals yield INVALID; anything uncertain is
    SUSPICIOUS so the user decides."""
    min_height_px: int = 24
    min_area_px: int = 900
    min_aspect_hw: float = 0.10
    thin_rel_height: float = 0.015
    tall_aspect_hw: float = 6.0
    blank_variance: float = 4.0
    blank_edge_density: float = 9.0
    blank_fg_ratio: float = 0.02
    invalid_blank_signals: int = 3
    suspicious_blank_signals: int = 2
    dup_hamming_near: int = 6
    overlap_flag: float = 0.30
    overlap_dup: float = 0.85


@dataclass
class PanelVerdict:
    panel_id: str
    quality: str = NORMAL
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    ai_confidence: float | None = None
    duplicate_of: str | None = None
    user_decision: str | None = None


@dataclass
class PanelReport:
    detected: int = 0
    accepted: int = 0
    suspicious: int = 0
    rejected: int = 0
    duplicates: int = 0
    verdicts: list[PanelVerdict] = field(default_factory=list)

    def stats_line(self) -> str:
        return (f"Detected: {self.detected}  Accepted: {self.accepted}  "
                f"Suspicious: {self.suspicious}  Rejected: {self.rejected}  "
                f"Duplicates: {self.duplicates}")

    def effective_ids(self, ordered_ids: list[str]) -> list[str]:
        by = {v.panel_id: v for v in self.verdicts}
        out = []
        for pid in ordered_ids:
            v = by.get(pid)
            if v is None:
                out.append(pid)
                continue
            if v.user_decision == "delete":
                continue
            if v.user_decision == "keep":
                out.append(pid)
                continue
            if v.quality == INVALID:
                continue
            out.append(pid)
        return out


def effective_ids(report_dict: dict, ordered_ids: list[str]) -> list[str]:
    by = {v["panel_id"]: v for v in report_dict.get("verdicts", [])}
    out = []
    for pid in ordered_ids:
        v = by.get(pid)
        if v is None:
            out.append(pid)
            continue
        if v.get("user_decision") == "delete":
            continue
        if v.get("user_decision") == "keep":
            out.append(pid)
            continue
        if v.get("quality") == INVALID:
            continue
        out.append(pid)
    return out


def _region_metrics(gray: np.ndarray, y0: int, y1: int) -> dict[str, float]:
    """Multi-signal content metrics for gray[y0:y1]."""
    import cv2
    reg = gray[max(0, y0):y1].astype(np.float32)
    variance = float(reg.var())
    edges = cv2.Sobel(reg, cv2.CV_32F, 1, 0, ksize=3)
    edge_density = float(np.abs(edges).mean())
    frame = np.concatenate([reg[:4].ravel(), reg[-4:].ravel(),
                            reg[:, :4].ravel(), reg[:, -4:].ravel()])
    border = float(np.median(frame)) if frame.size else 127.0
    fg_ratio = float((np.abs(reg - border) > 28).mean())
    hist, _ = np.histogram(reg, bins=64, range=(0, 256))
    p = hist / max(1, hist.sum())
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    return {"variance": round(variance, 2), "edge_density": round(edge_density, 2),
            "fg_ratio": round(fg_ratio, 4), "entropy": round(entropy, 3),
            "height": int(y1 - y0), "width": int(reg.shape[1])}


def _dhash(gray: np.ndarray, y0: int, y1: int) -> int:
    """64-bit difference hash for cheap duplicate detection."""
    import cv2
    reg = gray[max(0, y0):y1]
    small = cv2.resize(reg, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for b in diff.ravel():
        bits = (bits << 1) | int(b)
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def validate_panels(gray: np.ndarray, panels: list, *,
                    config: ValidationConfig | None = None,
                    ai_confidences: dict[str, float] | None = None,
                    ) -> PanelReport:
    """Validate cut panels (objects with .id/.panel_id, .y_start, .y_end).
    `gray` is the strip's grayscale array. Returns a PanelReport."""
    cfg = config or ValidationConfig()
    strip_h = gray.shape[0]
    confs = ai_confidences or {}
    report = PanelReport(detected=len(panels))
    hashes: list[tuple[str, int]] = []
    prev: PanelVerdict | None = None
    prev_y: tuple[int, int] | None = None
    for p in panels:
        pid = (p.get("id") if isinstance(p, dict) else (getattr(p, "id", None) or getattr(p, "panel_id", None)))
        y0 = int(p["y_start"] if isinstance(p, dict) else p.y_start)
        y1 = int(p["y_end"] if isinstance(p, dict) else p.y_end)
        v = PanelVerdict(panel_id=pid, ai_confidence=confs.get(pid))
        if y1 <= y0 or y0 < 0 or y1 > strip_h:
            v.quality, v.reasons = INVALID, [
                f"invalid coordinates [{y0},{y1}] for strip height {strip_h}"]
        else:
            m = _region_metrics(gray, y0, y1)
            v.metrics = m
            h, w, area = m["height"], m["width"], m["height"] * m["width"]
            aspect = h / max(1, w)
            if h < cfg.min_height_px:
                v.quality = INVALID
                v.reasons.append(f"extremely thin crop ({h}px tall)")
            elif aspect < cfg.min_aspect_hw:
                v.quality = INVALID
                v.reasons.append(
                    f"horizontal strip: height/width {aspect:.3f} < {cfg.min_aspect_hw}")
            elif area < cfg.min_area_px:
                v.quality = INVALID
                v.reasons.append(f"tiny area ({area}px²)")
            else:
                if h < cfg.thin_rel_height * strip_h:
                    v.quality = SUSPICIOUS
                    v.reasons.append(f"thin relative to strip ({h}px of {strip_h}px)")
                if aspect > cfg.tall_aspect_hw:
                    v.quality = SUSPICIOUS
                    v.reasons.append(f"very tall crop (h/w {aspect:.1f})")
                flat = sum([m["variance"] < cfg.blank_variance,
                            m["edge_density"] < cfg.blank_edge_density,
                            m["fg_ratio"] < cfg.blank_fg_ratio])
                if flat >= cfg.invalid_blank_signals:
                    v.quality = INVALID
                    v.reasons.append("near-zero visual content")
                elif flat >= cfg.suspicious_blank_signals:
                    if v.quality != INVALID:
                        v.quality = SUSPICIOUS
                    v.reasons.append("mostly blank")
        if v.quality != INVALID and y1 > y0:
            hsh = _dhash(gray, y0, y1)
            for other_id, other_h in hashes:
                if _hamming(hsh, other_h) <= cfg.dup_hamming_near:
                    v.duplicate_of = other_id
                    report.duplicates += 1
                    if v.quality == NORMAL:
                        v.quality = SUSPICIOUS
                    v.reasons.append(f"possible duplicate of {other_id}")
                    break
            hashes.append((pid, hsh))
        if prev_y is not None and v.quality != INVALID and y1 > y0:
            inter = min(prev_y[1], y1) - max(prev_y[0], y0)
            union = max(prev_y[1], y1) - min(prev_y[0], y0)
            iou = inter / max(1, union)
            if iou >= cfg.overlap_dup:
                v.duplicate_of = v.duplicate_of or prev.panel_id
                v.quality = SUSPICIOUS if v.quality == NORMAL else v.quality
                v.reasons.append(f"~identical to previous (IoU {iou:.2f})")
            elif iou >= cfg.overlap_flag:
                v.quality = SUSPICIOUS if v.quality == NORMAL else v.quality
                v.reasons.append(f"overlaps previous (IoU {iou:.2f})")
        prev, prev_y = v, (y0, y1)
        report.verdicts.append(v)
    for v in report.verdicts:
        if v.quality == NORMAL:
            report.accepted += 1
        elif v.quality == SUSPICIOUS:
            report.suspicious += 1
        else:
            report.rejected += 1
    log.info("[PANEL_VALIDATION] %s", report.stats_line())
    return report


def save_report(session_dir, report: PanelReport) -> Path:
    out = Path(session_dir) / "panels_validation.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(
        {"stats": {k: getattr(report, k) for k in
                   ("detected", "accepted", "suspicious", "rejected", "duplicates")},
         "verdicts": [asdict(v) for v in report.verdicts]}, indent=2), "utf-8")
    tmp.replace(out)
    return out


def load_report(session_dir) -> dict | None:
    p = Path(session_dir) / "panels_validation.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception:
        return None


def review_path(session_dir) -> Path:
    return Path(session_dir) / "review.json"


def load_review(session_dir) -> dict:
    p = review_path(session_dir)
    if not p.is_file():
        return {"confirmed": False, "decisions": {}}
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception:
        return {"confirmed": False, "decisions": {}}


def save_review(session_dir, review: dict) -> None:
    p = review_path(session_dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(review, indent=2), "utf-8")
    tmp.replace(p)


def apply_decisions(report_dict: dict, review: dict) -> dict:
    decisions = review.get("decisions", {})
    for v in report_dict.get("verdicts", []):
        v["user_decision"] = decisions.get(v["panel_id"])
    return report_dict
