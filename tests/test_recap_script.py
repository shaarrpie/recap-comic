# tests/test_recap_script.py
"""Offline tests for recap_script.py (Phase 2.5 whole-chapter script pass).

All model calls are fakes — no network, no keys. These enforce:
  * built path: prompt carries ALL panels + memory, response lines are
    validated against real panel indices, non-decreasing order enforced,
    script.json written with provenance
  * cached path: same input_hash -> cache hit, no second model call
  * fallback paths: no key / no text / bad response / model failure ->
    joined captions, never a crash
  * is_non_lexical: "..." / "—" / "" never become script lines
  * panel eligibility: blank + context_only panels are never mapped
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

import recap_script as rs
from guided_cutter import CutArtifact, CutPanel


def _panel(i: int, y0: int, y1: int, narration: str = "",
           dialogue: str = "", **kw) -> CutPanel:
    return CutPanel(id=f"panel_{i:03d}", panel_index=i, y_start=y0,
                    y_end=y1, narration=narration, dialogue=dialogue,
                    panel_type="panel", confidence=0.9,
                    image_file=f"panel_{i:03d}.png", **kw)


@pytest.fixture()
def session(tmp_path: Path) -> tuple[Path, CutArtifact]:
    panels = [
        _panel(1, 0, 800, "A gray background with no characters.",
               '"Where am I?"'),
        _panel(2, 800, 1600, "A man in a suit faces a door.", '"Follow me."'),
        _panel(3, 1600, 2400, "..."),          # non-lexical caption
        _panel(4, 2400, 3200, "An explosion rocks the corridor.",
               "Get down!"),
        _panel(5, 3200, 4000, "A hand reaches from the rubble.",
               '"Who is there?!"'),
    ]
    art = CutArtifact(source="strip.png", width=800, height=4000,
                      plan_hash="x", config={}, panels=panels)
    (tmp_path / "panels.json").write_text(art.model_dump_json(), "utf-8")
    return tmp_path, art


class FakeModel:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("no scripted response left")
        return self.responses.pop(0)


GOOD_LINES = json.dumps({"lines": [
    {"panel_index": 1, "text": "Jin was an ordinary guy — until this door.",
     "part": "hook", "quote": "Where am I?"},
    {"panel_index": 4, "text": "Then the corridor explodes around him.",
     "part": "escalation", "quote": None},
    {"panel_index": 5, "text": "A hand claws out of the rubble.",
     "part": "cliffhanger", "quote": None},
]})


# ------------------------------------------------------------ is_non_lexical --
def test_is_non_lexical():
    for s in ("", "...", "…", "—", "-", "*", "  .  ", "?!"):
        assert rs.is_non_lexical(s), s
    for s in ("Jin runs.", "a person stands", "hello world"):
        assert not rs.is_non_lexical(s), s


# -------------------------------------------------------------- built path --
def test_build_writes_script_json_with_validated_lines(session):
    d, _ = session
    model = FakeModel([GOOD_LINES])
    res = rs.build_chapter_script(d, model_call=model)

    assert res["status"] == "built"
    assert res["used_fallback"] is False
    assert len(res["lines"]) == 3
    # validated mapping: panel_index -> panel_id from panels.json,
    # and lines are sorted into panel order (narrator speaks in order)
    assert [ln["panel_id"] for ln in res["lines"]] == \
        ["panel_001", "panel_004", "panel_005"]
    assert [ln["part"] for ln in res["lines"]] == \
        ["hook", "escalation", "cliffhanger"]
    assert res["lines"][0]["quote"] == "Where am I?"
    assert res["lines"][1]["quote"] is None
    # text = joined line texts (TTS-ready, punctuation enforced)
    assert res["text"].startswith("Jin was an ordinary guy")
    # script.json on disk matches
    on_disk = json.loads((d / "script.json").read_text("utf-8"))
    assert on_disk["version"] == rs.SCRIPT_VERSION
    assert on_disk["input_hash"] == res["input_hash"]
    assert len(on_disk["lines"]) == 3


def test_prompt_contains_all_panels_and_dedups_nonlexical(session):
    d, _ = session
    model = FakeModel([GOOD_LINES])
    rs.build_chapter_script(d, model_call=model)
    prompt = model.prompts[0]
    # every usable panel's caption appears...
    assert "gray background" in prompt
    assert "man in a suit" in prompt
    assert "explosion" in prompt
    # ...but non-lexical panel 3 is marked unusable, not spoken
    assert "(no usable description)" in prompt
    # dialogue rides along for the scriptwriter
    assert '"Follow me."' in prompt
    # structure rules are in the prompt
    assert "cliffhanger" in prompt
    assert "hook" in prompt


def test_panel_indices_out_of_range_are_dropped(session):
    d, _ = session
    bad = json.dumps({"lines": [
        {"panel_index": 99, "text": "phantom panel", "part": "hook"},
        {"panel_index": 1, "text": "Real line.", "part": "setup"},
        {"panel_index": 2, "text": "Second.", "part": "escalation"},
    ]})
    res = rs.build_chapter_script(d, model_call=FakeModel([bad]))
    assert res["status"] == "built"
    assert [ln["panel_index"] for ln in res["lines"]] == [1, 2]


def test_out_of_order_lines_are_sorted_not_rejected(session):
    d, _ = session
    unordered = json.dumps({"lines": [
        {"panel_index": 4, "text": "later", "part": "hook"},
        {"panel_index": 1, "text": "earlier", "part": "setup"},
        {"panel_index": 2, "text": "middle one", "part": "escalation"},
        {"panel_index": 2, "text": "middle two", "part": "cliffhanger"},
    ]})
    res = rs.build_chapter_script(d, model_call=FakeModel([unordered]))
    assert res["status"] == "built"
    # the narrator speaks in panel order: sorted, nothing dropped
    assert [ln["panel_index"] for ln in res["lines"]] == [1, 2, 2, 4]
    assert [ln["text"] for ln in res["lines"]] == \
        ["earlier.", "middle one.", "middle two.", "later."]


def test_one_line_response_is_rejected_as_fallback(session):
    d, _ = session
    one = json.dumps({"lines": [
        {"panel_index": 1, "text": "Only line.", "part": "hook"}]})
    res = rs.build_chapter_script(d, model_call=FakeModel([one]))
    # a 1-line "script" cannot recap a chapter -> fallback join
    assert res["status"] == "fallback"
    assert res["used_fallback"] is True
    assert res["lines"] == []
    # fallback text is the joined captions (non-lexical "..." filtered)
    assert "gray background" in res["text"]
    assert "..." not in res["text"].split()
    assert not (d / "script.json").exists()


def test_garbage_response_falls_back(session):
    d, _ = session
    res = rs.build_chapter_script(
        d, model_call=FakeModel(["not json at all"]))
    assert res["status"] == "fallback"
    assert res["text"].strip()


def test_model_failure_falls_back_never_raises(session):
    d, _ = session
    res = rs.build_chapter_script(
        d, model_call=FakeModel([]))          # raises on call
    assert res["status"] == "fallback"
    assert res["used_fallback"] is True


# ---------------------------------------------------------------- cached path --
def test_cache_hit_skips_second_model_call(session):
    d, _ = session
    model = FakeModel([GOOD_LINES])
    first = rs.build_chapter_script(d, model_call=model)
    assert first["status"] == "built"

    second_model = FakeModel([])              # would raise if called
    second = rs.build_chapter_script(d, model_call=second_model)
    assert second["status"] == "cached"
    assert second_model.prompts == []
    assert second["lines"] == first["lines"]


def test_changed_panels_invalidate_cache(tmp_path):
    panels = [_panel(1, 0, 800, "First caption.")]
    art = CutArtifact(source="s.png", width=800, height=800,
                      plan_hash="x", config={}, panels=panels)
    (tmp_path / "panels.json").write_text(art.model_dump_json(), "utf-8")
    line = json.dumps({"lines": [
        {"panel_index": 1, "text": "Line one.", "part": "setup"}]})
    # need 2+ lines to be valid: use two panels
    panels = [_panel(1, 0, 400, "First caption."),
              _panel(2, 400, 800, "Second caption.")]
    art = CutArtifact(source="s.png", width=800, height=800,
                      plan_hash="x", config={}, panels=panels)
    (tmp_path / "panels.json").write_text(art.model_dump_json(), "utf-8")
    good = json.dumps({"lines": [
        {"panel_index": 1, "text": "Line one.", "part": "setup"},
        {"panel_index": 2, "text": "Line two.", "part": "cliffhanger"}]})

    model = FakeModel([good])
    rs.build_chapter_script(tmp_path, model_call=model)
    # edit a caption -> different input_hash -> model called again
    art.panels[1].narration = "Second caption, EDITED."
    (tmp_path / "panels.json").write_text(art.model_dump_json(), "utf-8")
    model2 = FakeModel([good])
    res = rs.build_chapter_script(tmp_path, model_call=model2)
    assert res["status"] == "built"       # cache did not hit
    assert len(model2.prompts) == 1


def test_force_rebuilds(session):
    d, _ = session
    model = FakeModel([GOOD_LINES])
    rs.build_chapter_script(d, model_call=model)
    model2 = FakeModel([GOOD_LINES])
    res = rs.build_chapter_script(d, model_call=model2, force=True)
    assert res["status"] == "built"
    assert len(model2.prompts) == 1


# -------------------------------------------------------------- eligibility --
def test_blank_and_context_only_never_mapped(tmp_path):
    panels = [
        _panel(1, 0, 400, "Visible.", '"Hi."'),
        _panel(2, 400, 800, "Blank.", blank_flag="blank"),
        _panel(3, 800, 1200, "Text only.", context_only=True),
        _panel(4, 1200, 1600, "Last.", '"Bye."'),
    ]
    art = CutArtifact(source="s.png", width=800, height=1600,
                      plan_hash="x", config={}, panels=panels)
    (tmp_path / "panels.json").write_text(art.model_dump_json(), "utf-8")

    model = FakeModel([GOOD_LINES])
    rs.build_chapter_script(tmp_path, model_call=model)
    prompt = model.prompts[0]
    # the eligibility filter feeds the model a list without 2/3
    assert "Text only." not in prompt
    assert "Blank." not in prompt
    # and lines can never map onto them (indices 2, 3 unknown)
    bad = json.dumps({"lines": [
        {"panel_index": 2, "text": "onto blank", "part": "hook"},
        {"panel_index": 3, "text": "onto context", "part": "setup"},
        {"panel_index": 1, "text": "real", "part": "escalation"},
        {"panel_index": 4, "text": "also real", "part": "cliffhanger"},
    ]})
    res = rs.build_chapter_script(tmp_path, model_call=FakeModel([bad]),
                                  force=True)
    assert [ln["panel_index"] for ln in res["lines"]] == [1, 4]


# ---------------------------------------------------------------- no-session --
def test_missing_panels_json_returns_none(tmp_path):
    assert rs.build_chapter_script(tmp_path, model_call=FakeModel([])) is None


# ------------------------------------------------------------------ load_script --
def test_load_script_roundtrip_and_rejects_bad(tmp_path):
    assert rs.load_script(tmp_path) is None            # absent
    (tmp_path / "script.json").write_text("not json", "utf-8")
    assert rs.load_script(tmp_path) is None            # unparseable
    good = {"version": rs.SCRIPT_VERSION, "lines": [
        {"panel_id": "p1", "panel_index": 1, "text": "x", "part": "hook",
         "quote": None}]}
    (tmp_path / "script.json").write_text(json.dumps(good), "utf-8")
    assert rs.load_script(tmp_path) == good
