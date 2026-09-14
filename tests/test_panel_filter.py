# tests/test_panel_filter.py
"""Offline tests for the deterministic panel-content filter (Phase 2.5).

All fixtures are synthetic strips drawn with NumPy/Pillow — no network, no
API keys, no AI. These tests enforce the two-tier contract:

  * BLANK tier   — only the exact string blank_flag='blank' removes a panel
                   from panels.json. 'suspicious'/'normal'/missing field
                   means keep (fail-safe); a legacy artifact without
                   blank_flag falls back to blank_score >= 0.70.
  * CONTEXT tier — a panel is demoted to context_only=True only when ALL
                   gates pass: it carries dialogue text, its crop is
                   low-saturation, white-dominant, and has edge structure.
                   Muted scene panels (art without dialogue) are never
                   demoted; pixels alone can never remove a panel.
  * Guards       — confirmed panels are untouchable, unscorable panels are
                   kept, an all-blank input never yields an empty output
                   (top-K rescue by blank_score), re-filtering is a no-op,
                   and the sidecar records every decision with evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import panel_filter as pf
from guided_cutter import CutArtifact, CutPanel

W = 800  # strip width used by every fixture


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #
def art_block(h: int, seed: int = 7, w: int = W) -> np.ndarray:
    """Noisy colour 'artwork': high saturation, high edge density."""
    rng = np.random.default_rng(seed)
    return rng.integers(30, 210, (h, w, 3), dtype=np.uint8)


def text_block(h: int, w: int = W) -> np.ndarray:
    """Mostly-white page with dark text strokes: low saturation, white
    dominant, structured edges — a text-only panel candidate."""
    block = np.full((h, w, 3), 254, dtype=np.uint8)
    for y in range(40, h - 40, 34):
        block[y:y + 4, 40:w - 40] = 12
    return block


def uniform_block(h: int, rgb: tuple[int, int, int],
                  w: int = W) -> np.ndarray:
    return np.full((h, w, 3), rgb, dtype=np.uint8)


def panel_dict(i: int, y0: int, y1: int, **over: object) -> dict:
    p: dict[str, object] = {
        "id": f"{i:03d}", "panel_index": i, "y_start": y0, "y_end": y1,
        "narration": f"panel {i}", "dialogue": "", "panel_type": "single",
        "confidence": 0.9, "image_file": f"panel_{i:03d}.png",
    }
    p.update(over)
    return p


def write_panels(d: Path, panels: list[dict], *, source: str = "strip.png",
                width: int = W, height: int | None = None) -> None:
    if height is None:
        height = sum(p["y_end"] - p["y_start"] for p in panels)
    (d / "panels.json").write_text(json.dumps({
        "source": source, "width": width, "height": height,
        "plan_hash": "t", "config": {}, "panels": panels,
    }), "utf-8")


def build_session(tmp_path: Path, blocks: list[np.ndarray],
                  panels: list[dict], *, source: str = "strip.png",
                  write_pngs: bool = True) -> Path:
    """Write strip + panels.json (+ placeholder panel PNGs) into a session
    dir; the strip enables the preferred source-crop scoring path."""
    d = tmp_path / "sess"
    d.mkdir()
    Image.fromarray(np.vstack(blocks)).save(d / source)
    write_panels(d, panels, source=source,
                 height=int(np.vstack(blocks).shape[0]))
    if write_pngs:
        for p in panels:
            Image.new("RGB", (12, 12), "white").save(d / str(p["image_file"]))
    return d


CFG = pf.FilterConfig()


# --------------------------------------------------------------------------- #
# _score_array
# --------------------------------------------------------------------------- #
def test_score_array_zero_size_returns_zeros():
    s = pf._score_array(np.zeros((0, 10, 3), dtype=np.uint8), CFG)
    assert s == {"white_of_content": 0.0, "color_ratio": 0.0,
                 "edge_density": 0.0}


def test_score_array_uniform_colours():
    # white page: all white, no colour, no edges
    s = pf._score_array(uniform_block(600, (255, 255, 255)), CFG)
    assert s["white_of_content"] == 1.0
    assert s["color_ratio"] == 0.0
    assert s["edge_density"] == pytest.approx(0.0, abs=1e-4)
    # black page: no white, no saturation, no edges
    s = pf._score_array(uniform_block(600, (0, 0, 0)), CFG)
    assert s["white_of_content"] == 0.0
    assert s["color_ratio"] == 0.0
    assert s["edge_density"] == pytest.approx(0.0, abs=1e-4)
    # saturated pastel: every non-white pixel is coloured
    s = pf._score_array(uniform_block(600, (120, 180, 220)), CFG)
    assert s["white_of_content"] == 0.0
    assert s["color_ratio"] == 1.0


def test_score_array_art_and_text_signatures():
    art = pf._score_array(art_block(800), CFG)
    text = pf._score_array(text_block(800), CFG)
    # art: saturated, few white pixels, strong edges
    assert art["color_ratio"] > 0.5
    assert art["white_of_content"] < 0.1
    assert art["edge_density"] > 0.05
    # text page: white-dominant, unsaturated, structured edges above the
    # fixed floor but far below art
    assert text["white_of_content"] > 0.85
    assert text["color_ratio"] < 0.05
    assert CFG.fixed_text_edge_floor < text["edge_density"] < art["edge_density"]


# --------------------------------------------------------------------------- #
# _trim_black_padding
# --------------------------------------------------------------------------- #
def test_trim_black_padding_removes_symmetric_padding():
    content = art_block(400)
    padded = np.vstack([np.zeros((60, W, 3), dtype=np.uint8), content,
                        np.zeros((60, W, 3), dtype=np.uint8)])
    out = pf._trim_black_padding(padded, dark_thr=25)
    assert out.shape[0] == 400
    assert np.array_equal(out, content)


def test_trim_black_padding_all_dark_left_as_is():
    arr = np.zeros((300, W, 3), dtype=np.uint8)
    out = pf._trim_black_padding(arr, dark_thr=25)
    assert out.shape == arr.shape  # blank_flag decides, not brightness


def test_trim_black_padding_no_dark_rows_untouched():
    arr = art_block(500)
    assert np.array_equal(pf._trim_black_padding(arr, 25), arr)


def test_trim_black_padding_empty_array():
    arr = np.zeros((0, W, 3), dtype=np.uint8)
    assert pf._trim_black_padding(arr, 25).size == 0


# --------------------------------------------------------------------------- #
# _is_blank (Tier 1: string compare on the cutter's verdict)
# --------------------------------------------------------------------------- #
def test_blank_flag_exact_blank_string_removes():
    assert pf._is_blank({"blank_flag": "blank"}, CFG) is True
    assert pf._is_blank({"blank_flag": "BLANK"}, CFG) is True  # case-folded
    assert pf._is_blank({"blank_flag": "Blank"}, CFG) is True


def test_blank_flag_suspicious_and_unknown_kept():
    # suspicious is a human-review verdict, never silently dropped
    assert pf._is_blank({"blank_flag": "suspicious",
                         "blank_score": 0.95}, CFG) is False
    # unknown flag strings are never treated as blank (string compare only)
    assert pf._is_blank({"blank_flag": "unusual"}, CFG) is False
    # 'normal' + a very high blank_score goes through the legacy score
    # fallback (fail-safe low at 0.70) and IS blank
    assert pf._is_blank({"blank_flag": "normal", "blank_score": 0.95},
                        CFG) is True
    # 'normal' cutter output in practice: score below the suspicious band
    assert pf._is_blank({"blank_flag": "normal", "blank_score": 0.5},
                        CFG) is False


def test_blank_flag_missing_falls_back_to_score():
    # legacy artifact with no blank_flag: threshold fallback, fail-safe low
    assert pf._is_blank({"blank_score": 0.70}, CFG) is True
    assert pf._is_blank({"blank_score": 0.95}, CFG) is True
    assert pf._is_blank({"blank_score": 0.69}, CFG) is False
    assert pf._is_blank({}, CFG) is False
    # garbage score values never raise
    assert pf._is_blank({"blank_score": "not-a-number"}, CFG) is False


# --------------------------------------------------------------------------- #
# _is_text_only (Tier 2: all gates must pass)
# --------------------------------------------------------------------------- #
THR_FIXED = {
    "method": "fixed", "n_panels": 3,
    "text_color_threshold": 0.05,
    "white_dominance_cutoff": 0.80,
    "text_edge_floor": 0.003,
    "is_bw_session": False,
}


def test_text_only_all_gates_pass():
    score = pf._score_array(text_block(800), CFG)
    assert pf._is_text_only({"dialogue": "You're too slow!"},
                            score, THR_FIXED) is True


@pytest.mark.parametrize("mutate,reason", [
    ({"dialogue": ""}, "no dialogue: pixels alone never demote"),
    ({"dialogue": "  "}, "whitespace dialogue is no dialogue"),
    ({"dialogue": "", "panel_type": "gutter"}, "no dialogue and non-text type"),
    ({"dialogue": "hi", "context_only": True}, "already demoted: idempotent"),
])
def test_text_only_content_gate_vetoes(mutate, reason):
    score = pf._score_array(text_block(800), CFG)
    panel = {"dialogue": "hi", "panel_type": "single"}
    panel.update(mutate)
    assert pf._is_text_only(panel, score, THR_FIXED) is False, reason


def test_text_only_pixel_gates_veto():
    # saturated art crop with dialogue: white gate fails (and color gate)
    art = pf._score_array(art_block(800), CFG)
    assert pf._is_text_only({"dialogue": "hi"}, art, THR_FIXED) is False
    # pure white with dialogue: zero edges -> no strokes -> fail
    pure = pf._score_array(uniform_block(800, (255, 255, 255)), CFG)
    assert pf._is_text_only({"dialogue": "hi"}, pure, THR_FIXED) is False
    # saturated-but-white page: color gate fails
    pink = pf._score_array(uniform_block(800, (250, 120, 180)), CFG)
    assert pf._is_text_only({"dialogue": "hi"}, pink, THR_FIXED) is False


def test_text_only_bw_session_skips_saturation_gate():
    # white page with COLOURED strokes: white-dominant with edges, but the
    # saturated text fails the color gate on a colour session; a greyscale
    # session cannot use saturation and skips that gate
    block = np.full((800, W, 3), 254, dtype=np.uint8)
    for y in range(40, 760, 34):
        block[y:y + 4, 40:W - 40] = (255, 40, 40)  # red strokes
    tinted = pf._score_array(block, CFG)
    assert tinted["white_of_content"] > 0.80
    assert tinted["edge_density"] > CFG.fixed_text_edge_floor
    assert tinted["color_ratio"] > 0.9  # nearly every stroke pixel is coloured
    thr_bw = {**THR_FIXED, "is_bw_session": True}
    assert pf._is_text_only({"dialogue": "hi"}, tinted, thr_bw) is True
    assert pf._is_text_only({"dialogue": "hi"}, tinted, THR_FIXED) is False


def test_text_only_missing_score_keeps():
    assert pf._is_text_only({"dialogue": "hi"}, {}, THR_FIXED) is False


# --------------------------------------------------------------------------- #
# _calibrate
# --------------------------------------------------------------------------- #
def test_calibrate_fixed_mode_below_min_panels():
    thr = pf._calibrate([pf._score_array(art_block(200), CFG)] * 3, CFG)
    assert thr["method"] == "fixed"
    assert thr["n_panels"] == 3
    assert thr["text_color_threshold"] == CFG.fixed_text_color_ratio
    assert thr["white_dominance_cutoff"] == CFG.fixed_white_dominance
    assert thr["text_edge_floor"] == CFG.fixed_text_edge_floor
    assert thr["is_bw_session"] is False


def test_calibrate_adaptive_colour_session():
    scores = [pf._score_array(b, CFG) for b in
              [art_block(600), art_block(600), art_block(600),
               text_block(600), art_block(600), text_block(600)]]
    thr = pf._calibrate(scores, CFG)
    assert thr["method"] == "adaptive"
    assert thr["is_bw_session"] is False
    # IQR fence on a mostly-art session stays at/below the cap
    assert 0.0 <= thr["text_color_threshold"] <= CFG.color_ratio_cap
    # edge floor is the ABSOLUTE constant in both modes (a percentile floor
    # would adapt upward and reject real text panels)
    assert thr["text_edge_floor"] == CFG.fixed_text_edge_floor


def test_calibrate_bw_session_detection():
    """A greyscale session (median color_ratio below the B&W threshold)
    skips the saturation gate: it cannot separate text on B&W paper."""

    def gray_block(h: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        gray = rng.integers(30, 210, (h, W), dtype=np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)

    scores = [pf._score_array(gray_block(600, s), CFG) for s in range(6)]
    thr = pf._calibrate(scores, CFG)
    assert thr["method"] == "adaptive"
    assert thr["is_bw_session"] is True
    # B&W sessions fall back to the fixed color threshold (gate skipped)
    assert thr["text_color_threshold"] == CFG.fixed_text_color_ratio


# --------------------------------------------------------------------------- #
# filter_panels — end-to-end two-tier filtering
# --------------------------------------------------------------------------- #
def test_filter_removes_blank_and_demotes_text_only(tmp_path):
    blocks = [art_block(900), uniform_block(700, (255, 255, 255)),
              art_block(900), text_block(1200)]
    panels = [
        panel_dict(1, 0, 900, dialogue="hero enters"),
        panel_dict(2, 900, 1600, blank_flag="blank", blank_score=0.95),
        panel_dict(3, 1600, 2500, narration="silent art"),
        panel_dict(4, 2500, 3700, dialogue="narration box text"),
    ]
    d = build_session(tmp_path, blocks, panels)
    res = pf.filter_panels(d)
    out = json.loads((d / "panels_filtered.json").read_text("utf-8"))

    assert res["total"] == 4
    assert res["removed_blank"] == 1
    assert res["context_only"] == 1
    assert res["kept"] == 2
    ids = [p["id"] for p in out["panels"]]
    assert "002" not in ids                       # blank panel removed
    by_id = {p["id"]: p for p in out["panels"]}
    assert by_id["004"]["context_only"] is True   # text-only demoted
    assert by_id["004"]["dialogue"] == "narration box text"  # dialogue kept
    assert not by_id["001"].get("context_only", False)  # art kept as scene
    assert not by_id["003"].get("context_only", False)
    # original file untouched (default writes panels_filtered.json)
    orig = json.loads((d / "panels.json").read_text("utf-8"))
    assert len(orig["panels"]) == 4
    # output stays schema-valid through the runtime model
    CutArtifact.model_validate_json(
        (d / "panels_filtered.json").read_text("utf-8"))


def test_filter_confirmed_panels_never_touched(tmp_path):
    blocks = [art_block(600), uniform_block(600, (0, 0, 0))]
    panels = [
        panel_dict(1, 0, 600, confirmed=True),
        panel_dict(2, 600, 1200, confirmed=True,
                   blank_flag="blank", blank_score=0.95),
    ]
    d = build_session(tmp_path, blocks, panels)
    res = pf.filter_panels(d)
    assert res["removed_blank"] == 0
    assert res["kept"] == 2
    decisions = {e["id"]: e["_decision"] for e in res["panels"]}
    assert decisions == {"001": pf.DECISION_KEEP,
                         "002": pf.DECISION_KEEP}


def test_filter_unscorable_panels_kept_fail_safe(tmp_path):
    """Panels whose crop cannot be scored (no strip, no PNG) are kept."""
    d = tmp_path / "sess"
    d.mkdir()
    # no strip, no PNGs; only the bare artifact
    panels = [panel_dict(1, 0, 400, dialogue="hi"),
              panel_dict(2, 400, 800, blank_flag="blank", blank_score=0.9)]
    write_panels(d, panels)
    res = pf.filter_panels(d)
    out = json.loads((d / "panels_filtered.json").read_text("utf-8"))
    # blank tier still applies (it needs no pixels); panel 1 kept (no score)
    assert [p["id"] for p in out["panels"]] == ["001"]
    methods = {e["id"]: e["_score_method"] for e in res["panels"]}
    assert methods["001"] == "missing"


def test_filter_suspicious_blank_flag_kept(tmp_path):
    """A suspicious (human-review) verdict must never silently drop a panel."""
    blocks = [art_block(600), uniform_block(600, (240, 240, 240))]
    panels = [panel_dict(1, 0, 600, dialogue="one"),
              panel_dict(2, 600, 1200, blank_flag="suspicious",
                         blank_score=0.72, dialogue="two")]
    d = build_session(tmp_path, blocks, panels)
    res = pf.filter_panels(d)
    assert res["removed_blank"] == 0
    assert res["kept"] + res["context_only"] == 2


def test_filter_empty_output_guard_rescues_top_k(tmp_path):
    """Every panel blank -> rescue the least-blank top-K, never zero output."""
    blocks = [uniform_block(500, (255, 255, 255)) for _ in range(5)]
    panels = [panel_dict(i + 1, i * 500, (i + 1) * 500,
                         blank_flag="blank", blank_score=score)
              for i, score in enumerate([0.99, 0.91, 0.95, 0.98, 0.93])]
    d = build_session(tmp_path, blocks, panels)
    res = pf.filter_panels(d)
    out = json.loads((d / "panels_filtered.json").read_text("utf-8"))
    assert res["rescued"] == min(CFG.min_kept_top_k, 5) == 3
    assert len(out["panels"]) == 3
    # rescued = the three LOWEST blank_scores (0.91=002, 0.93=005, 0.95=003)
    assert sorted(p["id"] for p in out["panels"]) == \
        ["002", "003", "005"]
    # no duplicates (regression: v3 double-appended rescued panels)
    assert len({p["id"] for p in out["panels"]}) == 3
    rescued_ids = {e["id"] for e in res["panels"]
                   if e["_decision"] == pf.DECISION_KEEP + "_rescued"}
    assert rescued_ids == {"002", "003", "005"}


def test_filter_sidecar_records_evidence(tmp_path):
    blocks = [art_block(600), text_block(600)]
    panels = [panel_dict(1, 0, 600, dialogue="hi"),
              panel_dict(2, 600, 1200, dialogue="box text")]
    d = build_session(tmp_path, blocks, panels)
    pf.filter_panels(d)
    side = json.loads((d / "filter_summary.json").read_text("utf-8"))
    assert side["filter_version"] == pf.FILTER_VERSION
    assert side["input_file"] == "panels.json"
    assert side["out_file"] == "panels_filtered.json"
    assert side["strip_found"] is True
    assert len(side["decisions"]) == 2
    assert {dd["decision"] for dd in side["decisions"]} >= {"keep"}
    assert all("scores" in dd or dd["method"] == "missing"
               for dd in side["decisions"])
    assert len(side["filter_hash"]) == 12


def test_filter_dry_run_writes_nothing(tmp_path):
    blocks = [art_block(600), uniform_block(600, (0, 0, 0))]
    panels = [panel_dict(1, 0, 600, dialogue="hi"),
              panel_dict(2, 600, 1200, blank_flag="blank", blank_score=0.9)]
    d = build_session(tmp_path, blocks, panels)
    res = pf.filter_panels(d, dry_run=True)
    assert res["removed_blank"] == 1
    assert (d / "panels_filtered.json").exists() is False
    assert (d / "filter_summary.json").exists() is False


def test_filter_idempotent_on_own_output(tmp_path):
    """Re-filtering an already-filtered artifact is a no-op: blanks are
    gone and context_only panels are never re-classified."""
    blocks = [art_block(600), uniform_block(600, (255, 255, 255)),
              text_block(700)]
    panels = [panel_dict(1, 0, 600, dialogue="scene one"),
              panel_dict(2, 600, 1200, blank_flag="blank", blank_score=0.95),
              panel_dict(3, 1200, 1900, dialogue="text box")]
    d = build_session(tmp_path, blocks, panels)
    first = pf.filter_panels(d)
    first_out = json.loads((d / "panels_filtered.json").read_text("utf-8"))
    snapshot = [(p["id"], p.get("context_only", False))
                for p in first_out["panels"]]

    # second pass reads the FIRST PASS OUTPUT as its panels.json
    d2 = tmp_path / "pass2"
    d2.mkdir()
    (d2 / "panels.json").write_text(json.dumps(first_out), "utf-8")
    Image.fromarray(np.vstack(blocks)).save(d2 / "strip.png")
    second = pf.filter_panels(d2)

    second_out = json.loads((d2 / "panels_filtered.json").read_text("utf-8"))
    assert [(p["id"], p.get("context_only", False))
            for p in second_out["panels"]] == snapshot
    assert second["removed_blank"] == 0
    assert second["context_only"] == first["context_only"]


def test_filter_renumbering_and_merged_with(tmp_path):
    """Indices are re-keyed 1..n over the kept set; merged_with patched."""
    blocks = [art_block(500), uniform_block(500, (255, 255, 255)),
              art_block(500), art_block(500)]
    panels = [panel_dict(1, 0, 500, dialogue="a"),
              panel_dict(2, 500, 1000, blank_flag="blank", blank_score=0.95),
              panel_dict(3, 1000, 1500, dialogue="b"),
              panel_dict(4, 1500, 2000, dialogue="c", merged_with=[3, 4])]
    d = build_session(tmp_path, blocks, panels)
    res = pf.filter_panels(d)
    assert res["removed_blank"] == 1
    out = json.loads((d / "panels_filtered.json").read_text("utf-8"))
    by_id = {p["id"]: p for p in out["panels"]}
    assert [p["panel_index"] for p in out["panels"]] == [1, 2, 3]
    # merged_with [3, 4]: 3 -> new 2 survives, 4 == self -> dropped
    assert by_id["004"]["merged_with"] == [2]


def test_filter_quarantine_moves_blank_pngs(tmp_path):
    blocks = [art_block(600), uniform_block(600, (0, 0, 0))]
    panels = [panel_dict(1, 0, 600, dialogue="hi"),
              panel_dict(2, 600, 1200, blank_flag="blank", blank_score=0.9)]
    d = build_session(tmp_path, blocks, panels)
    res = pf.filter_panels(d, out_file="panels.json",
                           quarantine_pngs=True)
    assert res["quarantined"] == 1
    assert not (d / "panel_002.png").exists()
    q = d / "_filtered_panels" / "panel_002.png"
    assert q.is_file()
    # kept panel PNG untouched, still visible to panel_*.png globs
    assert (d / "panel_001.png").is_file()


def test_filter_strips_underscore_keys_from_output(tmp_path):
    """panels.json stays schema-clean: _scores/_score_method/_decision
    live only in the sidecar/annotated result, never in the artifact."""
    blocks = [art_block(600)]
    panels = [panel_dict(1, 0, 600, dialogue="hi")]
    d = build_session(tmp_path, blocks, panels)
    pf.filter_panels(d)
    out = json.loads((d / "panels_filtered.json").read_text("utf-8"))
    assert all(not k.startswith("_") for p in out["panels"] for k in p)
    CutArtifact.model_validate_json(
        (d / "panels_filtered.json").read_text("utf-8"))


# --------------------------------------------------------------------------- #
# filter_panels_inplace — backup-once semantics
# --------------------------------------------------------------------------- #
def test_inplace_backup_created_once(tmp_path):
    blocks = [art_block(600), uniform_block(600, (255, 255, 255))]
    panels = [panel_dict(1, 0, 600, dialogue="hi"),
              panel_dict(2, 600, 1200, blank_flag="blank", blank_score=0.95)]
    d = build_session(tmp_path, blocks, panels)
    original_text = (d / "panels.json").read_text("utf-8")

    pf.filter_panels_inplace(d)
    backup = d / "panels_original.json"
    assert backup.is_file()
    assert backup.read_text("utf-8") == original_text
    filtered = json.loads((d / "panels.json").read_text("utf-8"))
    assert len(filtered["panels"]) == 1

    # second run: the FIRST backup wins; panels.json (already filtered)
    # is not backed up over the original
    (d / "panels.json").write_text(json.dumps(filtered), "utf-8")
    pf.filter_panels_inplace(d)
    assert backup.read_text("utf-8") == original_text


def test_inplace_missing_panels_file_raises(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        pf.filter_panels_inplace(d)


# --------------------------------------------------------------------------- #
# Strip location + crop scoring priority
# --------------------------------------------------------------------------- #
def test_locate_strip_candidate_paths(tmp_path):
    d = tmp_path / "sess"
    d.mkdir()
    data = {"source": "strip.png"}
    assert pf._locate_strip(d, data) is None
    (tmp_path / "strip.png").write_bytes(b"")  # parent dir candidate
    found = pf._locate_strip(d, data)
    assert found == tmp_path / "strip.png"
    (d / "strip.png").write_bytes(b"")        # session dir wins
    assert pf._locate_strip(d, data) == d / "strip.png"
    # manual/blank sources never resolve
    assert pf._locate_strip(d, {"source": "manual"}) is None
    assert pf._locate_strip(d, {"source": ""}) is None


def test_strip_crop_scoring_preferred_over_png(tmp_path):
    """When the strip is present, scores come from the full-res source
    crop (method='strip_crop'), not the normalized letterboxed PNG."""
    blocks = [art_block(600), text_block(600)]
    panels = [panel_dict(1, 0, 600, dialogue="hi"),
              panel_dict(2, 600, 1200, dialogue="box")]
    d = build_session(tmp_path, blocks, panels)
    res = pf.filter_panels(d)
    methods = {e["id"]: e["_score_method"] for e in res["panels"]}
    assert set(methods.values()) == {"strip_crop"}


def test_png_scoring_falls_back_when_strip_absent(tmp_path):
    """Without the strip, the letterboxed PNG is scored after trimming its
    black padding (method='padding_stripped')."""
    blocks = [art_block(600), text_block(760)]
    panels = [panel_dict(1, 0, 600, dialogue="hi"),
              panel_dict(2, 600, 1360, dialogue="box")]
    d = build_session(tmp_path, blocks, panels, source="gone.png")
    (d / "gone.png").unlink()  # the strip really is gone
    # PNGs written like guided_cutter would: 390 wide, padded to >=760
    for p, block in zip(panels, blocks):
        img = Image.fromarray(block).resize(
            (390, max(1, round(block.shape[0] * 390 / W))),
            Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (390, 760), (0, 0, 0))
        canvas.paste(img, (0, (760 - img.height) // 2))
        canvas.save(d / str(p["image_file"]))
    res = pf.filter_panels(d)
    methods = {e["id"]: e["_score_method"] for e in res["panels"]}
    assert set(methods.values()) == {"padding_stripped"}


# --------------------------------------------------------------------------- #
# Config overrides
# --------------------------------------------------------------------------- #
def test_filter_config_with_overrides():
    cfg = pf.FilterConfig().with_overrides(iqr_k=1.0, min_panels_for_adaptive=2)
    assert cfg.iqr_k == 1.0
    assert cfg.min_panels_for_adaptive == 2
    # unknown keys are rejected, not silently ignored
    with pytest.raises(ValueError, match="unknown filter overrides"):
        pf.FilterConfig().with_overrides(bogus_knob=1)


# --------------------------------------------------------------------------- #
# Downstream contract: context_only panels are skipped by narration and
# the video timeline (the reason the field exists)
# --------------------------------------------------------------------------- #
def test_context_only_skipped_by_downstream_stages(tmp_path):
    import narrator
    import recap_video as rv

    panels = [
        CutPanel(id="001", panel_index=1, y_start=0, y_end=800,
                 narration="Scene one.", dialogue="", panel_type="single",
                 confidence=0.9, image_file="panel_001.png"),
        CutPanel(id="002", panel_index=2, y_start=800, y_end=1600,
                 narration="Text box.", dialogue="inner monologue",
                 panel_type="single", confidence=0.9,
                 image_file="panel_002.png", context_only=True),
        CutPanel(id="003", panel_index=3, y_start=1600, y_end=2400,
                 narration="Scene two.", dialogue="", panel_type="single",
                 confidence=0.9, image_file="panel_003.png"),
    ]
    for p in panels:
        Image.new("RGB", (800, p.y_end - p.y_start), "white").save(
            tmp_path / p.image_file)
    art = CutArtifact(source="strip.png", width=800, height=2400,
                      plan_hash="x", config={}, panels=panels)

    # narrator: no spoken line for the context-only panel
    script = narrator.make_script_from_cut(art)
    assert "Scene one." in script and "Scene two." in script
    assert "Text box." not in script

    # video: no narration entry, no timeline frame
    cfg = rv.VideoConfig(tts="none")
    nar = rv.build_narration(art, cfg, panels_hash="h")
    assert [e.panel_id for e in nar.entries] == ["001", "003"]
    aud = rv.synthesize_audio(nar, tmp_path / "audio", cfg)
    tl = rv.build_timeline(art, tmp_path, nar, aud, tmp_path / "audio",
                           cfg, panels_hash="h")
    assert [e.panel_id for e in tl.entries] == ["001", "003"]


def test_cut_panel_context_only_survives_json_roundtrip():
    p = CutPanel(id="001", panel_index=1, y_start=0, y_end=100,
                 narration="", dialogue="", panel_type="single",
                 confidence=0.9, image_file="panel_001.png",
                 context_only=True)
    p2 = CutPanel.model_validate_json(p.model_dump_json())
    assert p2.context_only is True
    # default is False and the key is optional in the wire format
    p3 = CutPanel.model_validate_json(
        '{"id":"002","panel_index":1,"y_start":0,"y_end":100,"narration":"",'
        '"dialogue":"","panel_type":"single","confidence":0.9,'
        '"image_file":"panel_002.png"}')
    assert p3.context_only is False


# --------------------------------------------------------------------------- #
# CLI (_install_cli / standalone panel_filter.py)
# --------------------------------------------------------------------------- #
def _run_cli(d: Path, *flags: str) -> "subprocess.CompletedProcess":
    import subprocess
    import sys
    return subprocess.run(
        [sys.executable, str(Path(pf.__file__).resolve()), str(d), *flags],
        capture_output=True, text=True, timeout=120,
        cwd=str(Path(pf.__file__).resolve().parent))


def test_cli_default_writes_filtered_json(tmp_path):
    blocks = [art_block(600), uniform_block(600, (255, 255, 255))]
    panels = [panel_dict(1, 0, 600, dialogue="hi"),
              panel_dict(2, 600, 1200, blank_flag="blank", blank_score=0.95)]
    d = build_session(tmp_path, blocks, panels)
    r = _run_cli(d)
    assert r.returncode == 0, r.stderr
    assert "Removed" in r.stdout and "1 blank" in r.stdout
    assert (d / "panels_filtered.json").is_file()
    assert json.loads((d / "panels.json").read_text("utf-8"))["panels"]


def test_cli_dry_run_mutually_exclusive_with_apply(tmp_path):
    d = tmp_path / "sess"
    d.mkdir()
    write_panels(d, [panel_dict(1, 0, 400)])
    r = _run_cli(d, "--apply", "--dry-run")
    assert r.returncode == 2
    assert "mutually exclusive" in (r.stderr + r.stdout)


def test_cli_strict_loose_mutually_exclusive(tmp_path):
    d = tmp_path / "sess"
    d.mkdir()
    write_panels(d, [panel_dict(1, 0, 400)])
    r = _run_cli(d, "--strict", "--loose")
    assert r.returncode == 2
    assert "mutually exclusive" in (r.stderr + r.stdout)


def test_cli_apply_overwrites_with_backup_and_quarantine(tmp_path):
    blocks = [art_block(600), uniform_block(600, (0, 0, 0))]
    panels = [panel_dict(1, 0, 600, dialogue="hi"),
              panel_dict(2, 600, 1200, blank_flag="blank", blank_score=0.9)]
    d = build_session(tmp_path, blocks, panels)
    original = (d / "panels.json").read_text("utf-8")
    r = _run_cli(d, "--apply", "--quarantine")
    assert r.returncode == 0, r.stderr
    out = json.loads((d / "panels.json").read_text("utf-8"))
    assert [p["id"] for p in out["panels"]] == ["001"]
    assert (d / "panels_original.json").read_text("utf-8") == original
    assert (d / "_filtered_panels" / "panel_002.png").is_file()


def test_cli_dry_run_writes_nothing(tmp_path):
    blocks = [art_block(600)]
    panels = [panel_dict(1, 0, 600, dialogue="hi")]
    d = build_session(tmp_path, blocks, panels)
    r = _run_cli(d, "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "dry-run" in r.stdout
    assert not (d / "panels_filtered.json").exists()
    assert not (d / "panels_original.json").exists()
    assert not (d / "filter_summary.json").exists()


def test_cli_fixed_and_min_panels_flags(tmp_path):
    """--fixed forces fixed calibration regardless of session size."""
    blocks = [art_block(500), art_block(500), text_block(500),
              art_block(500), art_block(500), art_block(500),
              art_block(500)]
    panels = [panel_dict(1, 0, 500, dialogue="a"),
              panel_dict(2, 500, 1000, dialogue="b"),
              panel_dict(3, 1000, 1500, dialogue="text box"),
              panel_dict(4, 1500, 2000, dialogue="c"),
              panel_dict(5, 2000, 2500, dialogue="d"),
              panel_dict(6, 2500, 3000, dialogue="e"),
              panel_dict(7, 3000, 3500, dialogue="f")]
    d = build_session(tmp_path, blocks, panels)
    r = _run_cli(d, "--fixed", "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "fixed" in r.stdout
