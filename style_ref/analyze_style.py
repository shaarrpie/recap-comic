"""Computational style profiler for a reference recap video.

No vision needed: measures color, letterboxing, caption placement, and motion
grammar directly from extracted frames, then emits a machine-readable profile.
"""
import glob
import json

import numpy as np
from PIL import Image

FRAMES = sorted(glob.glob("style_ref/frames/f_*.png"))   # 1 fps
MOTION = sorted(glob.glob("style_ref/motion/m_*.png"))   # 5 fps
FPS_M = 5.0


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def lum(img):
    return img @ np.array([0.299, 0.587, 0.114])


def detect_bars(img):
    """Return black-bar thickness on each edge (px)."""
    h, w, _ = img.shape
    L = lum(img)
    row_dark = (L < 12).mean(axis=1)
    col_dark = (L < 12).mean(axis=0)

    def run(counts, from_start):
        n = 0
        idx = 0 if from_start else len(counts) - 1
        step = 1 if from_start else -1
        while 0 <= idx < len(counts) and counts[idx] > 0.95:
            n += 1
            idx += step
        return n

    return {
        "top": run(row_dark, True),
        "bottom": run(row_dark, False),
        "left": run(col_dark, True),
        "right": run(col_dark, False),
    }


def color_stats(img):
    L = lum(img)
    mx = img.max(axis=2)
    mn = img.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
    return {
        "mean_rgb": [round(float(v), 1) for v in img.reshape(-1, 3).mean(axis=0)],
        "brightness": round(float(L.mean()), 1),
        "contrast": round(float(L.std()), 1),
        "saturation": round(float(sat.mean()), 3),
        "p5_lum": round(float(np.percentile(L, 5)), 1),
        "p95_lum": round(float(np.percentile(L, 95)), 1),
    }


def caption_region(img):
    """Locate text via horizontal-gradient density; return bbox + color."""
    h, w, _ = img.shape
    g = lum(img)
    gx = np.abs(np.diff(g, axis=1))
    strong = gx > 40  # strong vertical strokes -> horizontal gradient
    dens = strong.mean(axis=1)  # per-row stroke density

    thr = max(0.02, dens.max() * 0.25)
    rows = np.where(dens > thr)[0]
    if len(rows) < 5:
        return None

    # largest contiguous row band
    bands = []
    start = rows[0]
    prev = rows[0]
    for r in rows[1:]:
        if r - prev > 4:
            bands.append((start, prev))
            start = r
        prev = r
    bands.append((start, prev))
    band = max(bands, key=lambda b: b[1] - b[0])

    y0, y1 = int(band[0]), int(band[1])
    cols = np.where(strong[y0:y1 + 1].mean(axis=0) > thr * 0.5)[0]
    if len(cols) == 0:
        return None
    x0, x1 = int(cols[0]), int(cols[-1])

    region = img[y0:y1 + 1, x0:x1 + 1]
    Lr = lum(region)
    bright_mask = Lr > np.percentile(Lr, 90)
    dark_mask = Lr < np.percentile(Lr, 10)
    fg = region[bright_mask].mean(axis=0) if bright_mask.any() else None
    bg = region[dark_mask].mean(axis=0) if dark_mask.any() else None

    return {
        "bbox_px": [x0, y0, x1, y1],
        "bbox_frac": [round(x0 / w, 3), round(y0 / h, 3),
                      round(x1 / w, 3), round(y1 / h, 3)],
        "height_frac": round((y1 - y0) / h, 3),
        "centered_x": abs(((x0 + x1) / 2) / w - 0.5) < 0.12,
        "fg_color": [round(float(v)) for v in fg] if fg is not None else None,
        "bg_color": [round(float(v)) for v in bg] if bg is not None else None,
        "in_bottom_third": (y0 + y1) / 2 / h > 0.66,
        "in_top_third": (y0 + y1) / 2 / h < 0.33,
    }


def motion_grammar(paths):
    """Classify each inter-frame transition: static / motion / cut."""
    prev = load(paths[0])
    events = []
    for p in paths[1:]:
        cur = load(p)
        d = np.abs(lum(cur) - lum(prev)).mean()
        # pan estimate via phase correlation
        shift, _ = cv2_phase(prev, cur)
        # zoom estimate: correlation of prev vs center-scaled cur
        z = zoom_factor(prev, cur)
        if d < 2.0:
            kind = "static"
        elif d > 25.0:
            kind = "cut"
        else:
            kind = "motion"
        events.append({
            "t": round(len(events) / FPS_M, 2),
            "diff": round(float(d), 2),
            "kind": kind,
            "shift_px": [round(float(v), 1) for v in shift] if shift is not None else None,
            "zoom": round(float(z), 3) if z is not None else None,
        })
        prev = cur
    return events


def cv2_phase(a, b):
    try:
        import cv2
        ga = np.asarray(Image.fromarray(a.astype(np.uint8)).convert("L"), dtype=np.float32)
        gb = np.asarray(Image.fromarray(b.astype(np.uint8)).convert("L"), dtype=np.float32)
        (dx, dy), resp = cv2.phaseCorrelate(ga, gb)
        if resp < 0.15 or (abs(dx) < 0.5 and abs(dy) < 0.5):
            return None, resp
        return (dx, dy), resp
    except Exception:
        return None, 0.0


def zoom_factor(prev, cur):
    """Detect Ken Burns zoom: match cur against rescaled prev crops."""
    try:
        import cv2
        h, w = lum(prev).shape
        best = None
        for scale in (0.80, 0.90, 1.10, 1.20):
            ch, cw = int(h / scale), int(w / scale)
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            crop = prev[y0:y0 + ch, x0:x0 + cw]
            resized = np.asarray(Image.fromarray(crop.astype(np.uint8)).resize((w, h)))
            corr = float(np.corrcoef(lum(resized).ravel(), lum(cur).ravel())[0, 1])
            if best is None or corr > best[1]:
                best = (scale, corr)
        if best and best[1] > 0.75:
            return best[0]
    except Exception:
        pass
    return None


def main():
    per_frame = []
    for p in FRAMES:
        img = load(p)
        per_frame.append({
            "frame": int(p.split("_")[-1].split(".")[0]),
            "bars": detect_bars(img),
            "color": color_stats(img),
            "caption": caption_region(img),
        })

    motion = motion_grammar(MOTION)

    cuts = [e for e in motion if e["kind"] == "cut"]
    shots = []
    if cuts:
        start = 0.0
        for c in cuts:
            shots.append(round(c["t"] - start, 2))
            start = c["t"]
        shots.append(round(len(MOTION) / FPS_M - start, 2))

    all_brightness = [f["color"]["brightness"] for f in per_frame]
    all_sat = [f["color"]["saturation"] for f in per_frame]
    captions = [f["caption"] for f in per_frame if f["caption"]]
    caps_bottom = [c for c in captions if c["in_bottom_third"]]
    caps_top = [c for c in captions if c["in_top_third"]]

    profile = {
        "source": "Recording 2026-09-16 064103.mp4",
        "resolution": [1292, 706],
        "fps": 30,
        "duration_s": 25.5,
        "aspect": "16:9",
        "letterbox": per_frame[0]["bars"],
        "color_grading": {
            "brightness_mean": round(float(np.mean(all_brightness)), 1),
            "brightness_range": [round(float(min(all_brightness)), 1),
                                 round(float(max(all_brightness)), 1)],
            "saturation_mean": round(float(np.mean(all_sat)), 3),
            "mean_rgb_avg": [round(float(np.mean([f["color"]["mean_rgb"][i]
                             for f in per_frame])), 1) for i in range(3)],
        },
        "captions": {
            "frames_with_text": len(captions),
            "total_frames": len(per_frame),
            "bottom_third": len(caps_bottom),
            "top_third": len(caps_top),
            "centered_x": sum(1 for c in captions if c["centered_x"]),
            "height_frac_mean": round(float(np.mean([c["height_frac"] for c in captions])), 3)
                                 if captions else None,
            "fg_colors": [c["fg_color"] for c in captions[:6]],
            "bg_colors": [c["bg_color"] for c in captions[:6]],
        },
        "pacing": {
            "cut_count": len(cuts),
            "cut_times": [c["t"] for c in cuts],
            "shot_lengths": shots,
            "avg_shot_s": round(float(np.mean(shots)), 2) if shots else None,
            "motion_events": sum(1 for e in motion if e["kind"] == "motion"),
            "static_events": sum(1 for e in motion if e["kind"] == "static"),
            "zoom_events": sum(1 for e in motion if e.get("zoom")),
            "sample_shifts": [e["shift_px"] for e in motion
                              if e["shift_px"]][:8],
        },
        "per_frame": per_frame,
        "motion_series": motion,
    }

    with open("style_ref/style_profile.json", "w") as fh:
        json.dump(profile, fh, indent=2)

    print(json.dumps({k: v for k, v in profile.items()
                      if k not in ("per_frame", "motion_series")}, indent=2))


if __name__ == "__main__":
    main()
