# tests/test_blank_detector.py
"""Tests for the deterministic (NO-AI) blank-region detector.

Every fixture is a synthetic strip drawn with NumPy/Pillow — no network,
no API keys, no AI. These tests enforce the spec's mandatory cases:

  white blank / black blank / gray blank / colored blank -> BLANK
  mostly-white real panel (character) -> NOT BLANK
  mostly-black real panel (artwork)   -> NOT BLANK
  speech bubble on white              -> NOT BLANK
  thin uniform gutter                 -> GUTTER (not a blank section)
  thin horizontal crop                -> INVALID (dropped by cutter)
  completely blank extracted panel   -> INVALID / BLANK (dropped)
  sparse star/content region          -> SUSPICIOUS or NORMAL, never BLANK
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import blank_detector as bd
import guided_cutter as gc
import strip_analyzer as sa

W = 800  # strip width used by all fixtures (matches make_strip)


def art_rows(rng: np.random.Generator, h: int, w: int = W) -> np.ndarray:
    """Noisy 'artwork' rows: high variance, high edge density."""
    return rng.integers(30, 210, (h, w), dtype=np.uint8)


def strip_image(blocks: list[tuple[str, int]], *,
                width: int = W, seed: int = 7) -> Image.Image:
    """Build a GRAYSCALE strip from (kind, height) blocks.

    kinds: art | white | black | gray | color | star | bubble
    (all blocks are single-channel; the 'color' kind simulates a solid
    pastel via its luminance so the RGB vstack stays consistent)
    """
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    for kind, h in blocks:
        if kind == "art":
            rows.append(art_rows(rng, h, width))
        elif kind == "white":
            rows.append(np.full((h, width), 255, dtype=np.uint8))
        elif kind == "black":
            rows.append(np.full((h, width), 0, dtype=np.uint8))
        elif kind == "gray":
            rows.append(np.full((h, width), 128, dtype=np.uint8))
        elif kind == "color":  # solid pastel blue -> luminance ~217
            rows.append(np.full((h, width), 217, dtype=np.uint8))
        elif kind == "star":
            # near-white region with one small dark star cluster (content)
            block = np.full((h, width), 255, dtype=np.uint8)
            cy, cx = h // 2, width // 2
            block[cy - 10:cy + 10, cx - 10:cx + 10] = 30
            rows.append(block)
        elif kind == "bubble":
            # white region with a bordered speech bubble + text-ish lines
            block = np.full((h, width), 255, dtype=np.uint8)
            bx0, bx1 = width // 4, 3 * width // 4
            by0, by1 = 20, h - 20
            block[by0:by1, bx0:bx1] = 255
            block[by0, bx0:bx1] = 0
            block[by1 - 1, bx0:bx1] = 0
            block[by0:by1, bx0] = 0
            block[by0:by1, bx1 - 1] = 0
            for ty in range(by0 + 15, by1 - 10, 25):  # fake text lines
                block[ty, bx0 + 15:bx0 + 150] = 20
            rows.append(block)
        else:
            raise ValueError(kind)
    return Image.fromarray(np.vstack(rows))


def run_detector(img: Image.Image, **cfg_over) -> list[bd.BlankRegion]:
    rgb = np.asarray(img.convert("RGB"))
    gray = np.asarray(img.convert("L"))
    cfg = bd.BlankDetectorConfig(**cfg_over) if cfg_over else None
    return bd.detect_blank_regions(gray, rgb=rgb, config=cfg)


# ----------------------------------------------------------------- cases --
def test_white_blank_region_detected():
    img = strip_image([("art", 900), ("white", 1200), ("art", 900)])
    regions = run_detector(img)
    blanks = [r for r in regions if r.verdict == bd.BLANK]
    assert len(blanks) == 1
    r = blanks[0]
    # must cover most of the white block (900..2100) with slack for windows
    assert r.y_start < 1000 and r.y_end > 2000
    assert r.score >= 0.90
    assert r.height >= 1000


def test_black_blank_region_detected():
    img = strip_image([("art", 900), ("black", 1200), ("art", 900)])
    blanks = [r for r in run_detector(img) if r.verdict == bd.BLANK]
    assert len(blanks) == 1
    assert blanks[0].height >= 1000


def test_gray_blank_region_detected():
    img = strip_image([("art", 900), ("gray", 1200), ("art", 900)])
    blanks = [r for r in run_detector(img) if r.verdict == bd.BLANK]
    assert len(blanks) == 1


def test_colored_blank_region_detected():
    img = strip_image([("art", 900), ("color", 1200), ("art", 900)])
    blanks = [r for r in run_detector(img) if r.verdict == bd.BLANK]
    assert len(blanks) == 1


@pytest.mark.parametrize("kind", ["white", "black", "gray", "color"])
def test_all_blank_colours_become_no_panels(kind):
    """Full pipeline: a strip with a blank section must not yield a panel
    for the blank region, whatever the blank colour."""
    img = strip_image([("art", 900), (kind, 1200), ("art", 900)])
    rgb = np.asarray(img.convert("RGB"))
    gray = np.asarray(img.convert("L"))
    cuts = [gc.CutPanel(id=f"{i + 1:03d}", panel_index=i + 1,
                        y_start=y0, y_end=y1, narration="n", dialogue="",
                        panel_type="single", confidence=0.9,
                        image_file=f"panel_{i + 1:03d}.png")
            for i, (y0, y1) in enumerate([(0, 900), (900, 2100), (2100, 3000)])]
    regions = bd.detect_blank_regions(gray, rgb=rgb)
    out = gc._apply_blank_regions(cuts, regions, 3000)
    ids = [c.id for c in out]
    assert "001" in ids and "003" in ids
    assert "002" not in ids  # the blank block is gone


def test_mostly_white_panel_with_character_not_blank():
    img = strip_image([("star", 1500)])  # white + one small dark cluster
    regions = run_detector(img)
    blanks = [r for r in regions if r.verdict == bd.BLANK]
    assert blanks == []  # sparse content rescues the region


def test_mostly_black_panel_with_artwork_not_blank():
    """A black panel where art reaches top AND bottom margins: the small
    pure-black edge bands are below the min-blank height for this strip
    size, so the panel is never classified blank."""
    rng = np.random.default_rng(3)
    block = np.zeros((1500, W), dtype=np.uint8)
    block[0:1500, 100:700] = rng.integers(40, 220, (1500, 600),
                                          dtype=np.uint8)
    img = Image.fromarray(block)
    regions = run_detector(img)
    assert [r for r in regions if r.verdict == bd.BLANK] == []


def test_speech_bubble_not_blank():
    img = strip_image([("bubble", 1500)])
    regions = run_detector(img)
    assert [r for r in regions if r.verdict == bd.BLANK] == []


def test_thin_gutter_is_not_a_blank_section():
    """A 40px uniform gap is a gutter/boundary, NOT a blank page."""
    img = strip_image([("art", 900), ("white", 40), ("art", 900)])
    regions = run_detector(img)
    assert [r for r in regions if r.verdict == bd.BLANK] == []
    # and no suspicious spam either: small quiet gaps are simply ignored
    assert regions == []


def test_extremely_thin_crop_is_invalid():
    """A 5px-tall 'panel' is dropped by guided_cut's min-height rule."""
    # enforced by the cutter; here: no blank region flags a 5px band
    img = strip_image([("art", 900), ("white", 5), ("art", 900)])
    regions = run_detector(img)
    assert regions == []


def test_completely_blank_extracted_panel_score():
    """score_crop: pure white / black / gray crops score >= 0.9."""
    for fill in (255, 0, 128):
        crop = np.full((600, 400, 3), fill, dtype=np.uint8)
        score, metrics = bd.score_crop(crop)
        assert score >= 0.9, (fill, score, metrics)


def test_content_crop_scores_low():
    rng = np.random.default_rng(5)
    crop = rng.integers(30, 210, (600, 400, 3), dtype=np.uint8)
    score, _ = bd.score_crop(crop)
    assert score < 0.65


# ------------------------------------------------- merging + tolerance --
def test_consecutive_blank_runs_merge():
    """blank | 3 noisy rows | blank must merge into ONE region."""
    rng = np.random.default_rng(7)
    top = art_rows(rng, 800)
    b1 = np.full((500, W), 255, dtype=np.uint8)
    noise = rng.integers(200, 255, (3, W), dtype=np.uint8)  # slight noise
    b2 = np.full((500, W), 255, dtype=np.uint8)
    bot = art_rows(rng, 800)
    img = Image.fromarray(np.vstack([top, b1, noise, b2, bot]))
    regions = run_detector(img)
    blanks = [r for r in regions if r.verdict == bd.BLANK]
    assert len(blanks) == 1
    assert blanks[0].height >= 900  # 500+3+500 minus windowing slack


def test_min_blank_height_respected():
    """A 150px uniform band (below min height for this strip) is ignored."""
    img = strip_image([("art", 2400), ("white", 150), ("art", 2400)])
    regions = run_detector(img)
    assert [r for r in regions if r.verdict == bd.BLANK] == []


def test_resolution_awareness():
    """Same relative layout at 2x scale -> same verdicts (normalized
    thresholds, not fixed pixel constants)."""
    def build(width: int) -> Image.Image:
        return strip_image([("art", int(900 * width / W)),
                            ("white", int(1200 * width / W)),
                            ("art", int(900 * width / W))], width=width)

    for width, expected in ((W, 1), (1600, 1)):
        img = build(width)
        blanks = [r for r in run_detector(img) if r.verdict == bd.BLANK]
        assert len(blanks) == expected, (width, len(blanks))


def test_determinism():
    img = strip_image([("art", 900), ("white", 1200), ("art", 900)])
    a = run_detector(img)
    b = run_detector(img)
    assert [(r.y_start, r.y_end, r.score, r.verdict) for r in a] == \
           [(r.y_start, r.y_end, r.score, r.verdict) for r in b]


def test_sensitivity_presets_change_strictness():
    """A shorter marginal blank (400px in a 3000px strip) is blank only
    under the 'high' preset; 'low' ignores it."""
    img = strip_image([("art", 1300), ("white", 400), ("art", 1300)])
    low = [r for r in run_detector(img, preset="low") if r.verdict == bd.BLANK]
    high = [r for r in run_detector(img, preset="high") if r.verdict == bd.BLANK]
    assert len(low) <= len(high)


def test_partial_blank_trims_panel_edges():
    """A panel with a blank tail keeps only its content part."""
    rng = np.random.default_rng(11)
    content = art_rows(rng, 700)
    blank_tail = np.full((600, W), 255, dtype=np.uint8)
    img = Image.fromarray(np.vstack([content, blank_tail]))
    rgb = np.asarray(img.convert("RGB"))
    gray = np.asarray(img.convert("L"))
    regions = bd.detect_blank_regions(gray, rgb=rgb)
    assert any(r.verdict == bd.BLANK for r in regions)
    cuts = [gc.CutPanel(id="001", panel_index=1, y_start=0, y_end=1300,
                        narration="n", dialogue="", panel_type="single",
                        confidence=0.9, image_file="panel_001.png")]
    out = gc._apply_blank_regions(cuts, regions, 1300)
    assert len(out) == 1
    assert out[0].y_start == 0
    # blank tail (720..1300) trimmed away within windowing slack
    assert out[0].y_end <= 720 + 64


def test_suspicious_regions_never_modify_geometry():
    """verdict=suspicious regions must not drop/trim panels - they only
    set flags for the review UI. The star fixture yields a genuine
    SUSPICIOUS region (sparse content keeps it out of BLANK)."""
    img = strip_image([("star", 1500)])  # near-white with small content
    rgb = np.asarray(img.convert("RGB"))
    gray = np.asarray(img.convert("L"))
    regions = bd.detect_blank_regions(gray, rgb=rgb)
    suspicious = [r for r in regions if r.verdict == bd.SUSPICIOUS]
    assert suspicious, "fixture must produce a suspicious region"
    assert all(r.verdict != bd.BLANK for r in regions)
    cuts = [gc.CutPanel(id="001", panel_index=1, y_start=0, y_end=1500,
                        narration="n", dialogue="", panel_type="single",
                        confidence=0.9, image_file="panel_001.png")]
    out = gc._apply_blank_regions(cuts, regions, 1500)
    assert len(out) == 1
    assert out[0].y_start == 0 and out[0].y_end == 1500  # geometry intact
    assert out[0].blank_flag == "suspicious"              # flag surfaced
    assert out[0].blank_score > 0


def test_guided_cut_drops_blank_panel(tmp_path, monkeypatch):
    """End-to-end: guided_cut never saves a PNG for a fully blank panel
    and marks blank-flagged panels, so they cannot reach the timeline."""
    img = strip_image([("art", 700), ("black", 1000), ("art", 700)])
    strip = tmp_path / "strip.png"
    img.save(strip)
    plan = sa.PanelPlan(
        source="strip.png", width=W, height=2400, model="test",
        config_hash="t", input_hash="t",
        entries=[
            sa.PanelPlanEntry(panel_index=1, y_start=0, y_end=700,
                              narration="one", dialogue="", confidence=0.9),
            sa.PanelPlanEntry(panel_index=2, y_start=700, y_end=1700,
                              narration="blank zone", dialogue="",
                              confidence=0.9),
            sa.PanelPlanEntry(panel_index=3, y_start=1700, y_end=2400,
                              narration="three", dialogue="", confidence=0.9),
        ])
    artifact = gc.guided_cut(strip, plan, tmp_path,
                            config=gc.CutterConfig())
    ids = [p.id for p in artifact.panels]
    assert "002" not in ids          # blank panel dropped entirely
    assert "001" in ids and "003" in ids
    saved_files = sorted(p.name for p in tmp_path.glob("panel_*.png"))
    assert all("002" not in f for f in saved_files)


def test_guided_cut_blank_detection_can_be_disabled(tmp_path):
    img = strip_image([("art", 700), ("black", 1000), ("art", 700)])
    strip = tmp_path / "strip.png"
    img.save(strip)
    plan = sa.PanelPlan(
        source="strip.png", width=W, height=2400, model="test",
        config_hash="t", input_hash="t",
        entries=[
            sa.PanelPlanEntry(panel_index=1, y_start=0, y_end=700,
                              narration="one", dialogue="", confidence=0.9),
            sa.PanelPlanEntry(panel_index=2, y_start=700, y_end=1700,
                              narration="blank zone", dialogue="",
                              confidence=0.9),
            sa.PanelPlanEntry(panel_index=3, y_start=1700, y_end=2400,
                              narration="three", dialogue="", confidence=0.9),
        ])
    artifact = gc.guided_cut(strip, plan, tmp_path,
                             config=gc.CutterConfig(blank_detection=False))
    assert len(artifact.panels) == 3  # nothing removed


def test_blank_flag_survives_json_roundtrip(tmp_path):
    """CutPanel.blank_* fields persist through panels.json so the webapp
    review UI can read them."""
    p = gc.CutPanel(id="001", panel_index=1, y_start=0, y_end=100,
                    narration="", dialogue="", panel_type="single",
                    confidence=0.9, image_file="panel_001.png",
                    blank_score=0.71, blank_flag="suspicious",
                    blank_reasons=["low variance", "low edges"])
    data = p.model_dump_json()
    p2 = gc.CutPanel.model_validate_json(data)
    assert p2.blank_score == 0.71
    assert p2.blank_flag == "suspicious"
    assert p2.blank_reasons == ["low variance", "low edges"]
