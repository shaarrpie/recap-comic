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


def test_call_with_retry_sleeps_for_retry_delay(monkeypatch: pytest.MonkeyPatch,
                                                 tmp_path: Path) -> None:
    """_call_with_retry should sleep for the server-suggested retry delay."""
    import strip_analyzer as sa

    sleeps: list[float] = []

    class FlakyBackend:
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
        sa.analyze_strip(strip, FlakyBackend(), chunk_height=2000, overlap=0,
                         attempts=2)

    assert len(sleeps) == 2
    assert sleeps[0] == 2.0
    assert sleeps[1] == 2.0
