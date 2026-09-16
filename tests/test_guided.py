# tests/test_guided.py
"""Offline tests for AI-Guided Panels & Narration (strip_analyzer +
guided_cutter + guided_pipeline).

All fixtures are synthetic strips drawn with Pillow/NumPy, so every test runs
with no API keys and no network; stub backends stand in for the vision model.
Synthetic fixtures prove the PLUMBING (coordinate mapping, gutter snapping,
merge/split logic, fallback, caching) — not real-world vision accuracy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import guided_cutter as gc
import guided_pipeline as gp
import strip_analyzer as sa
from adapters.schemas import BBox
from samples.make_sample_strip import make_sample_strip


def make_strip(height: int, *, width: int = 800,
               panels: list[tuple[int, int]] | None = None,
               gutters: list[tuple[int, int]] | None = None,
               bubble: tuple[int, int, int, int] | None = None,
               gutter_value: int = 255) -> Image.Image:
    """Synthetic strip: panels = art regions, gutters = uniform colour bands."""
    rng = np.random.default_rng(7)
    arr = np.full((height, width), 240, dtype=np.uint8)
    for y0, y1 in panels or []:
        arr[y0:y1] = rng.integers(30, 210, (y1 - y0, width), dtype=np.uint8)
    for y0, y1 in gutters or []:
        arr[y0:y1] = gutter_value
    if bubble is not None:
        x0, y0, x1, y1 = bubble
        arr[y0:y1, x0:x1] = 255
        arr[y0, x0:x1] = 0
        arr[y1 - 1, x0:x1] = 0
        arr[y0:y1, x0] = 0
        arr[y0:y1, x1 - 1] = 0
    return Image.fromarray(arr)


def plan_from(pairs: list[tuple[int, int]], *, height: int, width: int = 800,
              narration: str = "n", dialogue: str = "d",
              confidence: float = 0.9) -> sa.PanelPlan:
    entries = [
        sa.PanelPlanEntry(panel_index=i + 1, y_start=y0, y_end=y1,
                          narration=narration, dialogue=dialogue,
                          confidence=confidence)
        for i, (y0, y1) in enumerate(pairs)
    ]
    return sa.PanelPlan(source="strip.png", width=width, height=height,
                        model="test", config_hash="t", input_hash="t",
                        entries=entries)


class ErringBackend:
    """Vision backend stub that always raises (test the fallback path)."""
    name = "erring"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def analyze_chunk(self, image: Image.Image,
                      previous_context: str = ""
                      ) -> tuple[list[sa.PanelPlanEntry], list[str]]:
        raise self.exc


class FlakyBackend:
    """Vision backend stub that fails N times, then returns a canned plan."""
    name = "flaky"

    def __init__(self, fail_times: int,
                 entries: list[sa.PanelPlanEntry]) -> None:
        self.fail_times = fail_times
        self.entries = entries
        self.calls = 0

    def analyze_chunk(self, image: Image.Image,
                      previous_context: str = ""
                      ) -> tuple[list[sa.PanelPlanEntry], list[str]]:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ValueError("simulated model failure")
        return self.entries, []

def test_chunking_preserves_absolute_coordinates(tmp_path: Path) -> None:
    strip = tmp_path / "strip.png"
    make_strip(4500).save(strip)
    chunks = sa.make_chunks(Image.open(strip), chunk_height=2000, overlap=200)
    assert [base for base, _ in chunks] == [0, 1800, 3600]

    # chunk-local panels; one panel straddles chunks 0/1 (truncated vs full)
    backend = sa.FixtureVisionBackend([
        # chunk 0 (base 0)
        [sa.PanelPlanEntry(panel_index=1, y_start=100, y_end=900,
                           narration="A"),
         sa.PanelPlanEntry(panel_index=2, y_start=1700, y_end=2000,
                           narration="B-trunc")],
        # chunk 1 (base 1800); the seam panel view is LONGER here -> wins
        [sa.PanelPlanEntry(panel_index=1, y_start=0, y_end=400,
                           narration="B-full"),
         sa.PanelPlanEntry(panel_index=2, y_start=1200, y_end=2000,
                           narration="C")],
        # chunk 2 (base 3600)
        [sa.PanelPlanEntry(panel_index=1, y_start=200, y_end=800,
                           narration="D")],
    ])
    plan, used_cache = sa.analyze_strip(
        strip, backend, chunk_height=2000, overlap=200)
    assert used_cache is False
    ranges = [(e.y_start, e.y_end) for e in plan.entries]
    assert ranges == [(100, 900), (1700, 2200), (3000, 3800), (3800, 4400)]
    assert [e.panel_index for e in plan.entries] == [1, 2, 3, 4]
    assert [e.narration for e in plan.entries] == ["A", "B-full", "C", "D"]


def test_boundary_snaps_to_nearest_gutter_row() -> None:
    # true gutter rows 660..679 (white); AI offsets both boundaries by ~40px
    img = make_strip(900, panels=[(40, 660), (680, 900)], gutters=[(660, 680)])
    gray = np.asarray(img.convert("L"))
    plan = plan_from([(50, 640), (700, 890)], height=900)
    cuts = gc.build_cuts(gray, plan, config=gc.CutterConfig(tolerance=80))
    assert cuts[0].y_end == 669 and cuts[1].y_start == 669


def test_never_cut_through_speech_bubble() -> None:
    # the only gutter rows 660..679 sit under a bubble -> merge, not cut
    img = make_strip(940, panels=[(40, 660), (680, 940)], gutters=[(660, 680)])
    gray = np.asarray(img.convert("L"))
    plan = sa.PanelPlan(
        source="s.png", width=800, height=940, model="test",
        config_hash="t", input_hash="t",
        entries=[
            sa.PanelPlanEntry(panel_index=1, y_start=50, y_end=650,
                              narration="A", confidence=0.9,
                              bubble_boxes=[BBox(x=250, y=660, w=40, h=20)]),
            sa.PanelPlanEntry(panel_index=2, y_start=690, y_end=920,
                              narration="B", confidence=0.9)])
    cuts = gc.build_cuts(gray, plan, config=gc.CutterConfig(tolerance=100))
    assert len(cuts) == 1  # merged rather than cutting through the bubble
    assert cuts[0].merged_with == [1, 2]


def test_bubble_veto_no_cut_intersects_bbox() -> None:
    # Regression test (G3): every cut boundary row must avoid every bubble bbox.
    # The AI proposes panels [40,600] and [620,960] with a gutter at [600,620].
    # A bubble spans y=480..520 inside panel 1. The cutter must snap the cut
    # away from the bubble (or merge panels if the bubble spans the gutter).
    img = make_strip(1000, panels=[(40, 600), (620, 960)], gutters=[(600, 620)])
    gray = np.asarray(img.convert("L"))
    plan = sa.PanelPlan(
        source="s.png", width=800, height=1000, model="test",
        config_hash="t", input_hash="t",
        entries=[
            sa.PanelPlanEntry(panel_index=1, y_start=40, y_end=600,
                              narration="A", confidence=0.9,
                              bubble_boxes=[BBox(x=100, y=480, w=200, h=40)]),
            sa.PanelPlanEntry(panel_index=2, y_start=620, y_end=960,
                              narration="B", confidence=0.9)])
    cuts = gc.build_cuts(gray, plan, config=gc.CutterConfig(tolerance=80))
    # Collect all bubble y-ranges from the plan.
    bubble_ranges = []
    for e in plan.entries:
        for b in e.bubble_boxes:
            bubble_ranges.append((b.y, b.y + b.h))
    # The cut boundaries between adjacent panels are at cuts[i].y_end.
    # For a single merged panel there are no boundaries to check.
    if len(cuts) > 1:
        for i in range(len(cuts) - 1):
            boundary = cuts[i].y_end
            for by0, by1 in bubble_ranges:
                if by0 <= boundary < by1:
                    raise AssertionError(
                        f"cut boundary at y={boundary} intersects "
                        f"bubble bbox [{by0},{by1}]")


def test_tall_panel_split_at_internal_gutter() -> None:
    # panel 3: 1916..3700 (1784px) with internal white gutter 2800..2815
    img = make_strip(3700, panels=[(40, 1100), (1116, 1900), (1916, 3700)],
                     gutters=[(1100, 1116), (1900, 1916), (2800, 2816)])
    gray = np.asarray(img.convert("L"))
    plan = plan_from([(40, 1100), (1116, 1900), (1916, 3700)], height=3700,
                     narration="tall-scene")
    cuts = gc.build_cuts(gray, plan,
                         config=gc.CutterConfig(max_panel_height=1600))
    ids = [c.id for c in cuts]
    assert "003a" in ids and "003b" in ids
    pieces = [c for c in cuts if c.id.startswith("003")]
    assert len(pieces) == 2
    assert pieces[0].narration == "tall-scene"
    assert pieces[1].narration == "tall-scene"
    assert pieces[0].y_end == pieces[1].y_start  # contiguous
    assert all(p.split_of == "003" for p in pieces)


def test_merge_continuous_art_with_concatenated_narration() -> None:
    img = make_strip(1200, panels=[(40, 700), (700, 1200)])  # no gutter
    gray = np.asarray(img.convert("L"))
    plan = sa.PanelPlan(
        source="s.png", width=800, height=1200, model="test",
        config_hash="t", input_hash="t",
        entries=[
            sa.PanelPlanEntry(panel_index=1, y_start=50, y_end=690,
                              narration="first", dialogue="hi", confidence=0.8),
            sa.PanelPlanEntry(panel_index=2, y_start=710, y_end=1180,
                              narration="second", dialogue="", confidence=0.8)])
    cuts = gc.build_cuts(gray, plan, config=gc.CutterConfig(tolerance=80))
    assert len(cuts) == 1
    assert cuts[0].narration == "first second"
    assert cuts[0].dialogue == "hi"
    assert cuts[0].merged_with == [1, 2]


def test_outer_edges_clamped_to_strip_bounds() -> None:
    """F1: first panel y_start and last panel y_end must be clamped to 0 and
    strip height, not left as raw AI coordinates."""
    img = make_strip(1000, panels=[(50, 470), (520, 980)],
                     gutters=[(470, 520)])
    gray = np.asarray(img.convert("L"))
    plan = plan_from([(60, 460), (530, 970)], height=1000)
    cuts = gc.build_cuts(gray, plan, config=gc.CutterConfig(tolerance=80))
    assert len(cuts) == 2
    assert cuts[0].y_start == 0        # clamped from AI's 60
    assert cuts[-1].y_end == 1000      # clamped from AI's 970


def test_overlapping_ai_panels_are_repaired() -> None:
    """F2: overlapping AI boundaries (a.y_end > b.y_start) are repaired to
    their midpoint before the gutter search."""
    img = make_strip(1000, panels=[(40, 470), (470, 980)],
                     gutters=[(470, 490)])
    gray = np.asarray(img.convert("L"))
    plan = sa.PanelPlan(
        source="s.png", width=800, height=1000, model="test",
        config_hash="t", input_hash="t",
        entries=[
            sa.PanelPlanEntry(panel_index=1, y_start=40, y_end=500,
                              narration="A", confidence=0.9),
            sa.PanelPlanEntry(panel_index=2, y_start=480, y_end=980,
                              narration="B", confidence=0.9)])
    cuts = gc.build_cuts(gray, plan, config=gc.CutterConfig(tolerance=80))
    assert len(cuts) == 2
    assert cuts[0].y_end <= cuts[1].y_start  # no overlap after repair


def test_thin_panel_filtered_in_guided_cut(tmp_path: Path) -> None:
    """F8: panels thinner than min_panel_height are skipped, not saved as
    blank/white images."""
    strip = tmp_path / "strip.png"
    img = make_strip(1000, panels=[(40, 400), (500, 980)],
                     gutters=[(400, 500)])
    img.save(strip)
    # AI thinks there are 3 panels; the middle one (410..430) is just 20px
    # of gutter — it should be filtered out.
    plan = plan_from([(50, 390), (410, 430), (510, 950)], height=1000)
    artifact = gc.guided_cut(strip, plan, out_dir=tmp_path / "panels")
    ids = [p.id for p in artifact.panels]
    assert "002" not in ids  # thin panel (20px after snap) skipped
    assert len(artifact.panels) >= 2


def _panel_png_size(out_dir: Path, image_file: str) -> tuple[int, int]:
    with Image.open(out_dir / image_file) as img:
        img.load()
        return img.size


def test_panel_output_size_is_normalized(tmp_path: Path) -> None:
    """Output-size policy: PNGs are exactly 390px wide, height in [760, 800].

    Source coordinates stay full-resolution; only the PNG bytes change.
    """
    strip = tmp_path / "strip.png"
    make_strip(1700, panels=[(0, 800), (820, 1620)],
               gutters=[(800, 820)]).save(strip)
    out = tmp_path / "panels"
    plan = plan_from([(0, 800), (820, 1620)], height=1700)
    artifact = gc.guided_cut(strip, plan, out_dir=out)
    assert len(artifact.panels) == 2
    for p in artifact.panels:
        w, h = _panel_png_size(out, p.image_file)
        assert w == 390
        assert 760 <= h <= 800
        assert p.output_width == w
        assert p.output_height == h
        # Source geometry untouched: y range still matches the plan.
        assert p.y_end > p.y_start


def test_panel_output_size_clamps_tall_and_short(tmp_path: Path) -> None:
    """Short pads to 760; tall panels NEVER center-crop (full-res passthrough).

    A tall panel is a continuous-art mega-group the splitter kept whole;
    center-cropping it to 800px would discard most of the art, so the
    normalization keeps the FULL-resolution crop and the video stage pans
    top-to-bottom through it (recap_video.compute_pan contract).
    """
    strip = tmp_path / "strip.png"
    # Tall panel: 800px-wide crop, 1600px tall -> scaled to 390x780 (in bounds,
    # so the direct unit check below covers the true out-of-bounds paths).
    make_strip(2000, panels=[(0, 1600), (1620, 1720)],
                gutters=[(1600, 1620)]).save(strip)
    out = tmp_path / "panels"
    plan = plan_from([(0, 1600), (1620, 1720)], height=2000)
    artifact = gc.guided_cut(
        strip, plan, out_dir=out,
        config=gc.CutterConfig(normalize_output=True))
    assert len(artifact.panels) == 2
    for p in artifact.panels:
        w, h = _panel_png_size(out, p.image_file)
        assert w == 390
        assert 760 <= h <= 800

    # True out-of-bounds paths on the pure helper (deterministic, no I/O):
    # tall (4000px -> resized 1950px > 800) keeps the FULL-RES crop —
    # never a center crop, never art loss.
    tall = Image.new("RGB", (800, 4000), (120, 30, 30))
    assert gc.normalize_panel_image(tall).size == (800, 4000)
    short = Image.new("RGB", (800, 100), (30, 120, 30))
    assert gc.normalize_panel_image(short).size == (390, 760)
    # A tall panel under a high max_output_height still resizes (no crop
    # needed): 800x4000 -> 390x1950 at width 390.
    raw = gc.normalize_panel_image(tall, max_output_height=5000,
                                   min_output_height=1)
    assert raw.size == (390, 1950)


def test_panel_output_normalize_opt_out_keeps_fullres(tmp_path: Path) -> None:
    """normalize_output=False restores legacy full-resolution crops."""
    strip = tmp_path / "strip.png"
    make_strip(1700, panels=[(0, 800), (820, 1620)],
                gutters=[(800, 820)]).save(strip)
    out = tmp_path / "panels"
    plan = plan_from([(0, 800), (820, 1620)], height=1700)
    artifact = gc.guided_cut(
        strip, plan, out_dir=out,
        config=gc.CutterConfig(normalize_output=False))
    assert len(artifact.panels) == 2
    for p in artifact.panels:
        w, h = _panel_png_size(out, p.image_file)
        assert (w, h) == (800, p.y_end - p.y_start)
        assert p.output_width is None
        assert p.output_height is None


def test_mega_panel_keeps_full_art_for_video_pan(tmp_path: Path) -> None:
    """A continuous-art mega-group the splitter keeps whole must NOT be
    center-cropped: the panel PNG keeps the full art so the video stage
    pans top-to-bottom (recap_video.compute_pan: "Never centre-crops away
    content"). Regression for the webapp recap videos that showed only the
    middle ~10-30% of full-bleed action pages.

    Fixture: one 4000px continuous art region (no gutters inside), a
    normal panel below it, and a real gutter between them so the plan's
    two entries survive the merge step (continuous art between the two
    entries does not exist there).
    """
    strip = tmp_path / "strip.png"
    make_strip(4700, panels=[(0, 4000), (4120, 4600)],
                gutters=[(4000, 4120)]).save(strip)
    out = tmp_path / "panels"
    plan = plan_from([(0, 4000), (4120, 4600)], height=4700)
    artifact = gc.guided_cut(strip, plan, out_dir=out)
    mega = next(p for p in artifact.panels if p.y_end - p.y_start > 3000)
    # PNG on disk keeps the full-resolution crop: 800px wide, ALL of the
    # 4000px art (the boundary may snap into the adjacent gutter midpoint,
    # hence >=), not a 390x800 center crop.
    w, h = _panel_png_size(out, mega.image_file)
    assert w == 800
    assert h >= 4000
    # Recorded geometry matches the PNG bytes actually written.
    assert mega.output_width == 800
    assert mega.output_height == h
    # compute_pan gives a top-to-bottom pan over the whole art.
    from recap_video import compute_pan
    pan = compute_pan(mega.output_width, mega.output_height)
    assert pan.kind == "pan_down"
    assert pan.travel_px > 3000
    # The normal panel below still normalizes to 390x[760,800].
    normal = next(p for p in artifact.panels
                  if 3000 >= p.y_end - p.y_start > 100)
    nw, nh = _panel_png_size(out, normal.image_file)
    assert nw == 390
    assert 760 <= nh <= 800


def test_phase1_cache_avoids_recall(tmp_path: Path) -> None:
    strip = tmp_path / "strip.png"
    make_strip(2200).save(strip)
    backend = sa.FixtureVisionBackend([
        [sa.PanelPlanEntry(panel_index=1, y_start=0, y_end=1000, narration="x")],
        [sa.PanelPlanEntry(panel_index=1, y_start=200, y_end=900, narration="y")],
    ])
    cache = tmp_path / "cache"
    plan1, c1 = sa.analyze_strip(strip, backend, chunk_height=2000,
                                 overlap=200, cache_dir=cache)
    calls = backend.calls
    plan2, c2 = sa.analyze_strip(strip, backend, chunk_height=2000,
                                 overlap=200, cache_dir=cache)
    assert c1 is False and c2 is True
    assert backend.calls == calls  # backend not called again on cache hit
    assert plan1.entries[0].y_start == plan2.entries[0].y_start


def test_retry_on_invalid_then_success(tmp_path: Path) -> None:
    strip = tmp_path / "strip.png"
    make_strip(1500).save(strip)
    backend = FlakyBackend(fail_times=2, entries=[
        sa.PanelPlanEntry(panel_index=1, y_start=0, y_end=1000,
                          narration="ok", confidence=0.9)])
    plan, _ = sa.analyze_strip(strip, backend, chunk_height=2000, overlap=0,
                               attempts=3)
    assert [(e.y_start, e.y_end) for e in plan.entries] == [(0, 1000)]
    assert backend.calls == 3


def test_json_parse_rejects_prose_and_fenced_json() -> None:
    good = ('```json\n'
            '{"panels":[{"panel_index":1,"y_start":10,"y_end":500,'
            '"confidence":0.9}]}\n```')
    entries, _characters = sa.parse_entries_from_json(good, 1000)
    assert len(entries) == 1 and entries[0].y_end == 500
    with pytest.raises(ValueError):
        sa.parse_entries_from_json("sure, here is a prose recap...", 1000)
    entries, _ = sa.parse_entries_from_json(
        '{"panels":[{"panel_index":1,"y_start":10,"y_end":2000,'
        '"confidence":0.9}]}', 1000)
    assert entries[0].y_end == 1000  # clamped to chunk bounds


def test_fallback_on_backend_failure(tmp_path: Path) -> None:
    strip = tmp_path / "strip.png"
    make_strip(1200, panels=[(40, 300), (340, 600), (640, 1100)],
               gutters=[(300, 340), (600, 640)]).save(strip)
    out = tmp_path / "out"
    plan, artifact, used = gp.run_guided(
        strip, out, backend=ErringBackend(RuntimeError("api down")),
        fallback=True)
    assert used is True
    assert plan is not None  # fallback plan replaces the failed AI plan
    assert artifact is not None and len(artifact.panels) == 3
    assert (out / "panels.json").is_file()


def test_fallback_on_low_confidence(tmp_path: Path) -> None:
    """Bug A2 regression: an explicit plan file is NEVER discarded due to
    low confidence. The user trusted it enough to pass it on the CLI."""
    strip = tmp_path / "strip.png"
    make_strip(1200, panels=[(40, 300), (340, 600), (640, 1100)],
               gutters=[(300, 340), (600, 640)]).save(strip)
    plan_file = tmp_path / "low_conf.json"
    low = sa.PanelPlan(
        source="strip.png", width=800, height=1200, model="test",
        config_hash="t", input_hash="t",
        entries=[sa.PanelPlanEntry(panel_index=i + 1, y_start=y0, y_end=y1,
                                   narration="n", dialogue="d", confidence=0.3)
                 for i, (y0, y1) in enumerate([(40, 300), (340, 600),
                                               (640, 1100)])])
    plan_file.write_text(low.model_dump_json(), "utf-8")
    out = tmp_path / "out2"
    _, artifact, used = gp.run_guided(strip, out, backend_name="none",
                                      plan_path=plan_file, fallback=True)
    assert used is False, "explicit plan must not be discarded due to low confidence"
    assert artifact is not None and len(artifact.panels) == 3
    # AI narrations are preserved on the cut panels.
    assert all(p.narration == "n" for p in artifact.panels)


def test_fallback_segments_sample_strip(tmp_path: Path) -> None:
    """The variance-gutter fallback correctly splits the sample strip into
    its 5 logical bands (panel 3 contains an internal gutter)."""
    strip = tmp_path / "sample_strip.png"
    make_sample_strip(strip)
    plan = gp.fallback_plan_from_gutter_detector(strip)
    assert len(plan.entries) == 5
    assert [(e.y_start, e.y_end) for e in plan.entries] == [
        (0, 1107), (1107, 1907), (1907, 2807), (2807, 3707), (3707, 4400)]


def test_gutter_broken_by_character_still_detected(tmp_path: Path) -> None:
    """Majority-width gutter test (no AI): a gutter that is NOT uniform across
    the full strip width — a character stands in it, or a border line breaks
    it — must still be detected, otherwise the cutter falls back to the
    nearest locally-flat run and cuts mid-character."""
    strip = tmp_path / "strip.png"
    # Two art panels with a 16px white gutter between them, but a 200px-wide
    # black bar (simulating a character/border) crosses the gutter, so no
    # row is uniform across the full 800px width.
    img = make_strip(1200, panels=[(40, 600), (616, 1160)],
                     gutters=[(600, 616)])
    arr = np.array(img)
    arr[600:616, 300:500] = 0   # break the gutter over 200px of its width
    Image.fromarray(arr).save(strip)
    plan = gp.fallback_plan_from_gutter_detector(strip)
    assert len(plan.entries) == 2, (
        f"expected the broken gutter to be detected; got "
        f"{[(e.y_start, e.y_end) for e in plan.entries]}")
    assert plan.entries[0].y_end == plan.entries[1].y_start


def test_character_hair_is_not_cut(tmp_path: Path) -> None:
    """A locally-uniform block (solid black hair) that covers only a fraction
    of the strip width must NOT be mistaken for a gutter."""
    strip = tmp_path / "strip.png"
    # No real gutter anywhere; a 200px-wide solid black block sits in the
    # middle of otherwise-noisy art. The cutter must keep the strip whole
    # (no cut) rather than slicing through the hair.
    img = make_strip(1200, panels=[(40, 1160)])
    arr = np.array(img)
    arr[500:600, 300:500] = 0   # solid black block, 200px of 800px width
    Image.fromarray(arr).save(strip)
    plan = gp.fallback_plan_from_gutter_detector(strip)
    # No validated valley -> the strip stays one panel, never a blind cut.
    assert len(plan.entries) == 1, (
        f"expected the strip to stay whole (hair is not a gutter); got "
        f"{[(e.y_start, e.y_end) for e in plan.entries]}")


# --- New tests for Step 6 fixes and Step 1/5 additions ---


def test_fallback_provenance_exempt_from_low_conf_rule(tmp_path: Path) -> None:
    """A plan that ALREADY came from the gutter detector (provenance='fallback')
    has confidence 0.0 on every panel. It must NOT trigger the low-confidence
    fallback rule again, or we'd loop forever re-deriving the same plan."""
    strip = tmp_path / "strip.png"
    make_strip(1200, panels=[(40, 300), (340, 600), (640, 1100)],
               gutters=[(300, 340), (600, 640)]).save(strip)
    plan_file = tmp_path / "fallback.json"
    fb = sa.PanelPlan(
        source="strip.png", width=800, height=1200, model="gutter-fallback",
        config_hash="fallback", input_hash="fallback",
        provenance="fallback",
        entries=[sa.PanelPlanEntry(panel_index=i + 1, y_start=y0, y_end=y1,
                                   narration="", confidence=0.0)
                 for i, (y0, y1) in enumerate([(40, 300), (340, 600),
                                               (640, 1100)])])
    plan_file.write_text(fb.model_dump_json(), "utf-8")
    out = tmp_path / "out_fb"
    _, artifact, used = gp.run_guided(strip, out, backend_name="none",
                                      plan_path=plan_file, fallback=True)
    # The fallback-provenance plan is accepted as-is; no re-derivation loop.
    assert used is False
    assert artifact is not None and len(artifact.panels) == 3


def test_narrator_recap_and_literal(tmp_path: Path) -> None:
    """The narrator module produces both flowing-recap and verbatim-literal
    scripts from a plan, offline (no API call)."""
    from narrator import make_script
    plan = sa.PanelPlan(
        source="s.png", width=800, height=4000, model="test",
        config_hash="t", input_hash="t",
        entries=[
            sa.PanelPlanEntry(panel_index=1, y_start=0, y_end=1000,
                              narration="A wizard arrives.",
                              dialogue="", panel_type="single",
                              confidence=0.9),
            sa.PanelPlanEntry(panel_index=2, y_start=1000, y_end=2000,
                              narration="He draws his sword",
                              dialogue="", panel_type="single",
                              confidence=0.8),
            sa.PanelPlanEntry(panel_index=3, y_start=2000, y_end=3000,
                              narration="", dialogue="",
                              panel_type="transition_gutter",
                              confidence=0.7),
            sa.PanelPlanEntry(panel_index=4, y_start=3000, y_end=4000,
                              narration="The end.",
                              dialogue="", panel_type="single",
                              confidence=0.9),
        ])
    recap = make_script(plan, style="recap")
    # Recap joins the 3 non-empty narrations into one paragraph, with the
    # empty panel skipped.
    assert "wizard arrives" in recap
    assert "draws his sword" in recap
    assert "The end." in recap
    literal = make_script(plan, style="literal")
    assert "wizard arrives" in literal
    assert "draws his sword" in literal
    assert "The end." in literal
    # Literal keeps panels as separate blocks; recap joins them with spaces.


def test_report_html_embedded(tmp_path: Path) -> None:
    """The report module writes a self-contained HTML page with base64 panel
    images and narration text."""
    from guided_cutter import CutArtifact, CutPanel
    from report import render_report
    out_dir = tmp_path / "cut_out"
    out_dir.mkdir()
    # Create tiny placeholder PNGs so the report can embed them.
    for name in ("panel_001.png", "panel_002.png"):
        import struct
        import zlib
        def _1x1_png() -> bytes:
            sig = b"\x89PNG\r\n\x1a\n"
            ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            idat = zlib.compress(b"\x00\xff\x00\x00")
            def chunk(typ, data):
                c = typ + data
                return struct.pack(">I", len(data)) + c + struct.pack(
                    ">I", zlib.crc32(c) & 0xffffffff)
            return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
        (out_dir / name).write_bytes(_1x1_png())
    art = CutArtifact(
        source="strip.png", width=800, height=4000, plan_hash="x",
        config={"tolerance": 80},
        panels=[CutPanel(id="001", panel_index=1, y_start=0, y_end=1000,
                         narration="Hello world", dialogue="Hi!",
                         panel_type="single", confidence=0.9,
                         image_file="panel_001.png"),
                CutPanel(id="002", panel_index=2, y_start=1000, y_end=2000,
                         narration="", dialogue="", panel_type="gutter",
                         confidence=0.0, image_file="panel_002.png")])
    # Report sits alongside the cut output dir, so images resolve correctly.
    report_path = out_dir / "report.html"
    render_report(art, report_path)
    html = report_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert "Hello world" in html
    assert "data:image/png;base64," in html
    assert "panel_001.png" in html

def test_end_to_end_dry_run_and_cut(tmp_path: Path) -> None:
    strip = tmp_path / "strip.png"
    make_strip(1200, panels=[(40, 300), (340, 600), (640, 1100)],
               gutters=[(300, 340), (600, 640)]).save(strip)
    plan, artifact, used = gp.run_guided(strip, tmp_path / "unused",
                                         backend_name="none", dry_run=True,
                                         fallback=True)
    assert used is True and artifact is None and plan is not None
    out = tmp_path / "cut"
    _, artifact, _ = gp.run_guided(strip, out, backend_name="none")
    assert artifact is not None
    for p in artifact.panels:
        assert (out / p.image_file).is_file()
        assert 0 <= p.y_start < p.y_end <= 1200
    sidecar = gc.CutArtifact.model_validate_json(
        (out / "panels.json").read_text("utf-8"))
    assert len(sidecar.panels) == len(artifact.panels)


# ---------------------------------------------------------------------------
# Regression tests for the bugs found in the code review.
# ---------------------------------------------------------------------------

class _LowConfBackend:
    """Backend stub: every panel returns confidence 0.1 (all below 0.5)."""
    name = "lowconf"

    def __init__(self, height: int, n: int = 4) -> None:
        self.height, self.n = height, n
        self.calls = 0

    def analyze_chunk(self, image: Image.Image,
                      previous_context: str = ""
                      ) -> tuple[list[sa.PanelPlanEntry], list[str]]:
        self.calls += 1
        h = self.height
        step = h // (self.n + 1)
        entries = [sa.PanelPlanEntry(
            panel_index=i + 1, y_start=i * step, y_end=(i + 1) * step,
            narration=f"panel {i + 1}", dialogue="", panel_type="single",
            confidence=0.1)  # ALL below 0.5 -> must trigger fallback
            for i in range(self.n)]
        return entries, []


def test_low_confidence_fallback_actually_falls_back(tmp_path: Path) -> None:
    """Bug A3 regression: low-confidence AI plans must keep their narrations
    even when the geometry is replaced by the gutter detector."""
    strip = tmp_path / "strip.png"
    make_strip(1600, panels=[(40, 380), (420, 760), (800, 1140), (1180, 1540)],
               gutters=[(380, 420), (760, 800), (1140, 1180)]).save(strip)
    backend = _LowConfBackend(height=1600, n=4)
    plan, _artifact, used = gp.run_guided(
        strip, tmp_path / "out", backend=backend, fallback=True)
    assert used is True, "expected fallback flag when all panels < 0.5"
    assert plan.provenance == "fallback", (
        "the returned plan must come from the gutter detector for geometry")
    # The fallback geometry replaces the AI boundaries, but the AI's
    # narration/dialogue are preserved on the best-overlapping fallback panel.
    assert all(e.narration for e in plan.entries), (
        "AI narrations must survive the low-confidence fallback")
    assert len(plan.entries) >= 3


def test_dimension_mismatch_scales_coordinates(tmp_path: Path) -> None:
    """Bug 1.5 regression: when the plan was made from a resized strip, the
    cutter must scale panel coordinates proportionally rather than warn-and-proceed
    (which would produce garbage crops)."""
    # Make a strip at 1600px tall, then build a plan as if it were 800px
    # tall (half resolution). The cutter should scale everything by 2x.
    strip = tmp_path / "strip.png"
    make_strip(1600, panels=[(40, 380), (420, 760), (800, 1140)],
               gutters=[(380, 420), (760, 800)]).save(strip)
    # Plan says 800px tall; coordinates are in plan-space (0-800).
    # Strip is 1600px -> cutter must scale by 2x on both axes.
    entries = [sa.PanelPlanEntry(
        panel_index=i + 1, y_start=s, y_end=e,
        narration=f"panel {i + 1}", dialogue="", panel_type="single",
        confidence=0.9)
        for i, (s, e) in enumerate([(20, 190), (210, 380), (400, 570)])]
    plan = sa.PanelPlan(source="strip.png", width=400, height=800,
                        model="test", config_hash="h", input_hash="i",
                        provenance="ai", entries=entries)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json(), "utf-8")
    out = tmp_path / "cut"
    artifact = gc.guided_cut(strip, plan, out)
    assert artifact.height == 1600  # scaled to actual strip
    for p in artifact.panels:
        assert 0 <= p.y_start < p.y_end <= 1600
    assert (out / artifact.panels[0].image_file).is_file()


def test_dimension_mismatch_fails_on_wrong_aspect_ratio(tmp_path: Path) -> None:
    """Bug 1.5: a plan with a wildly different aspect ratio must raise, not
    silently produce garbage."""
    strip = tmp_path / "strip.png"
    make_strip(1600, panels=[(40, 380), (420, 760)],
               gutters=[(380, 420)]).save(strip)
    entries = [sa.PanelPlanEntry(
        panel_index=i + 1, y_start=s, y_end=e,
        narration=f"panel {i + 1}", dialogue="", panel_type="single",
        confidence=0.9)
        for i, (s, e) in enumerate([(20, 190), (210, 380)])]
    # width 4000 gives aspect 5.0 vs strip aspect 2.0 -> > 5% divergence
    plan = sa.PanelPlan(source="other.png", width=4000, height=800,
                        model="test", config_hash="h", input_hash="i",
                        provenance="ai", entries=entries)
    import pytest
    with pytest.raises(ValueError, match="aspect ratio"):
        gc.guided_cut(strip, plan, tmp_path / "out")





def test_ollama_backend_name_accepted(tmp_path: Path) -> None:
    """Bug 1.7 regression: the CLI advertises ollama, so build_backend must
    accept 'ollama' (and 'local') without raising."""
    # We can't actually call Ollama, but we can verify the factory accepts
    # the name without a network call (it should fail on connection, not
    # on unknown backend).
    try:
        gp.build_backend("ollama", model="llava")
    except ValueError as exc:
        if "unknown backend" in str(exc):
            raise AssertionError(
                "build_backend rejected 'ollama' — backend name drift") from exc
        # Any other error (connection, etc.) is fine — we just want to
        # confirm the name is recognized.
    except (ConnectionError, OSError):
        pass  # connection errors are acceptable here


def test_cloudflare_backend_name_accepted() -> None:
    """Cloudflare Workers AI backend must be accepted by build_backend."""
    try:
        gp.build_backend("cloudflare")
    except ValueError as exc:
        if "unknown backend" in str(exc):
            raise AssertionError(
                "build_backend rejected 'cloudflare' — backend name drift") from exc
    except RuntimeError:
        pass  # missing credentials is fine; we only care the name is recognized


def test_parse_retry_delay_from_quota_error() -> None:
    """_parse_retry_delay extracts seconds from a Gemini 429 JSON body."""
    exc = ValueError(
        '{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", '
        '"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", '
        '"retryDelay": "51s"}]}}'
    )
    delay = sa._parse_retry_delay(exc)
    assert delay == 51.0


def test_parse_retry_delay_missing() -> None:
    """_parse_retry_delay returns None when no retry info is present."""
    exc = RuntimeError("some random error")
    assert sa._parse_retry_delay(exc) is None


def test_call_with_retry_fails_fast_on_quota(monkeypatch: pytest.MonkeyPatch,
                                             tmp_path: Path) -> None:
    """On a 429 RESOURCE_EXHAUSTED, _call_with_retry must fail-fast (no
    wasted retry sleeps) so the gutter-detector fallback can take over
    immediately instead of stalling for minutes.
    """
    import strip_analyzer as sa

    sleeps: list[float] = []

    class QuotaBackend:
        def analyze_chunk(self, image, previous_context=""):
            raise ValueError(
                '{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", '
                '"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", '
                '"retryDelay": "2s"}]}}'
            )

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(sa.time, "sleep", fake_sleep)

    strip = tmp_path / "strip.png"
    make_strip(1200).save(strip)
    with pytest.raises(sa.VisionAnalysisError):
        sa.analyze_strip(strip, QuotaBackend(), chunk_height=2000, overlap=0,
                         attempts=3)

    # No retry sleeps allowed for 429: fail-fast on the first attempt.
    assert sleeps == []


def test_call_with_retry_sleeps_for_retry_delay(monkeypatch: pytest.MonkeyPatch,
                                                 tmp_path: Path) -> None:
    """_call_with_retry should sleep for the server-suggested retry delay
    on a transient (non-quota) error."""
    import strip_analyzer as sa

    sleeps: list[float] = []

    class FlakyBackend:
        def analyze_chunk(self, image, previous_context=""):
            raise ValueError(
                '{"error": {"code": 503, "status": "UNAVAILABLE", '
                '"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", '
                '"retryDelay": "2s"}]}}'
            )

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(sa.time, "sleep", fake_sleep)

    strip = tmp_path / "strip.png"
    make_strip(1200).save(strip)
    with pytest.raises(sa.VisionAnalysisError):
        sa.analyze_strip(strip, FlakyBackend(), chunk_height=2000, overlap=0,
                         attempts=2)

    assert len(sleeps) == 2
    assert sleeps[0] == 2.0
    assert sleeps[1] == 2.0

# ------------------------- silent-failure hardening (plan hash, bubbles, --
# --------------------------------------- snap measurement, smoke metric) ---


def test_plan_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    """A plan produced from a DIFFERENT strip (same aspect ratio) used to be
    cut against this strip verbatim -> garbage crops. The recorded strip
    SHA-256 must match."""
    import hashlib

    strip = tmp_path / "strip.png"
    make_strip(1200, panels=[(40, 300), (340, 600), (640, 1100)],
               gutters=[(300, 340), (600, 640)]).save(strip)
    other = tmp_path / "other.png"
    make_strip(900, panels=[(40, 300), (340, 600)],
               gutters=[(300, 340)]).save(other)

    plan_file = tmp_path / "foreign.json"
    plan = sa.PanelPlan(
        source="strip.png", width=800, height=1200, model="test",
        config_hash="t", input_hash=hashlib.sha256(other.read_bytes()).hexdigest(),
        entries=[sa.PanelPlanEntry(panel_index=i + 1, y_start=y0, y_end=y1,
                                   narration="n", confidence=0.9)
                 for i, (y0, y1) in enumerate([(40, 300), (340, 600),
                                               (640, 1100)])])
    plan_file.write_text(plan.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="DIFFERENT strip"):
        gp.run_guided(strip, tmp_path / "out", backend_name="none",
                      plan_path=plan_file, fallback=True)


def test_plan_hash_match_is_accepted(tmp_path: Path) -> None:
    """A plan that provably belongs to this strip is cut normally."""
    import hashlib

    strip = tmp_path / "strip.png"
    make_strip(1200, panels=[(40, 300), (340, 600), (640, 1100)],
               gutters=[(300, 340), (600, 640)]).save(strip)
    plan_file = tmp_path / "plan.json"
    plan = sa.PanelPlan(
        source="strip.png", width=800, height=1200, model="test",
        config_hash="t",
        input_hash=hashlib.sha256(strip.read_bytes()).hexdigest(),
        entries=[sa.PanelPlanEntry(panel_index=i + 1, y_start=y0, y_end=y1,
                                   narration="n", dialogue="d", confidence=0.9)
                 for i, (y0, y1) in enumerate([(40, 300), (340, 600),
                                               (640, 1100)])])
    plan_file.write_text(plan.model_dump_json(), encoding="utf-8")
    _plan, artifact, used = gp.run_guided(strip, tmp_path / "out",
                                          backend_name="none",
                                          plan_path=plan_file, fallback=True)
    assert used is False
    assert artifact is not None and len(artifact.panels) == 3


def test_find_valley_cuts_never_cuts_a_forbidden_band() -> None:
    """find_valley_cuts must honour the forbidden rows (speech bubbles): a
    gutter run inside a bubble is not a cut."""
    strip_arr = make_strip(1200, panels=[(40, 600), (616, 1160)],
                           gutters=[(600, 616)])
    gray = np.asarray(strip_arr.convert("L"))
    # unguarded: cuts through the gutter
    cuts = gc.find_valley_cuts(gray)
    assert any(600 < c < 616 for c in cuts)
    # the bubble band covers the gutter -> no cut there
    guarded = gc.find_valley_cuts(gray, forbidden=frozenset(range(596, 621)))
    assert all(not (596 <= c <= 620) for c in guarded)


def test_fallback_passes_bubble_rows_to_both_cutters(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bubble protection is not AI-path-only: the gutter fallback has no AI
    bubble boxes, so it must derive the forbidden rows itself and pass them
    to BOTH the valley scan and the oversized-panel splitter."""
    strip = tmp_path / "strip.png"
    make_sample_strip(strip)
    captured: dict[str, frozenset] = {}
    real_valley = gc.find_valley_cuts
    real_split = gc._split_panel

    def _valley(gray: np.ndarray, *, forbidden=frozenset(), **kw) -> list[int]:
        captured["valley"] = forbidden
        return real_valley(gray, forbidden=forbidden, **kw)

    def _split(gray: np.ndarray, panel, forbidden: frozenset,
               config: gc.CutterConfig, **kw):
        captured["split"] = forbidden
        return real_split(gray, panel, forbidden, config, **kw)

    monkeypatch.setattr(gc, "find_valley_cuts", _valley)
    monkeypatch.setattr(gc, "_split_panel", _split)
    plan = gp.fallback_plan_from_gutter_detector(strip, max_panel_height=1000)
    assert plan.entries
    assert isinstance(captured["valley"], frozenset)
    assert captured["split"] == captured["valley"], (
        "the splitter must use the same forbidden rows as the valley scan")


def test_snap_measured_marks_unsnapped_boundaries(tmp_path: Path) -> None:
    """A boundary kept WITHOUT a gutter match carries 0 px that is 'not
    measured', not '0px error' — snap_measured must say so."""
    strip = tmp_path / "strip.png"
    # continuous art across the AI boundary: no gutter run exists to snap to
    make_strip(1200, panels=[(40, 1160)]).save(strip)
    plan = plan_from([(40, 600), (616, 1160)], height=1200)
    gray = np.asarray(Image.open(strip).convert("L"))
    cuts = gc.build_cuts(
        gray, plan, config=gc.CutterConfig(preserve_boundaries=True))
    assert len(cuts) == 2
    for c in cuts:
        assert len(c.snap_measured) == len(c.snap_distances)
    # the only boundary had no gutter -> nothing was measured
    assert all(not m for c in cuts for m in c.snap_measured)


def test_snap_measured_true_for_real_snaps(tmp_path: Path) -> None:
    strip = tmp_path / "strip.png"
    make_strip(1200, panels=[(40, 300), (340, 600), (640, 1100)],
               gutters=[(300, 340), (600, 640)]).save(strip)
    plan = plan_from([(40, 300), (340, 600), (640, 1100)], height=1200)
    gray = np.asarray(Image.open(strip).convert("L"))
    cuts = gc.build_cuts(gray, plan, config=gc.CutterConfig())
    assert any(m for c in cuts for m in c.snap_measured), (
        "boundaries snapped to a real gutter must count as measurements")
    for c in cuts:
        assert len(c.snap_measured) == len(c.snap_distances)


def test_smoke_snap_stats_exclude_unmeasured_boundaries() -> None:
    """The smoke test's accuracy metric must not count 'no gutter found'
    boundaries as 0px — that made fallback runs report a perfect median."""
    from scripts.smoke_test_live import snap_stats

    def _cut(dists: list[int], measured: list[bool]) -> gc.CutPanel:
        return gc.CutPanel(id="p", panel_index=1, y_start=0, y_end=100,
                           narration="", dialogue="", panel_type="single",
                           confidence=0.9, image_file="p.png",
                           snap_distances=dists, snap_measured=measured)

    stats = snap_stats([_cut([3, 5], [True, True]),      # both measured
                        _cut([0, 0], [False, False]),    # never measured
                        _cut([7], [])])                  # legacy: measured
    assert stats == {"count": 3, "unsnapped": 2, "mean_px": 5.0,
                     "median_px": 5.0, "max_px": 7}
    # every boundary unmeasured -> EMPTY stats, not a perfect 0
    assert snap_stats([_cut([0, 0], [False, False])]) == {
        "count": 0, "unsnapped": 2}

