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
def test_panel_list_block_scrubs_non_english_text():
    """Mojibake and untranslated bubbles cached in an OLD panels.json must not
    reach the model (and from there edge-tts). Cleaning happens at prompt-build
    time, so already-narrated chapters are fixed without re-spending a single
    vision call."""
    hangul = "\uc544\uc774\uc2a1"
    mojibake = "\u03c6\u00f2\u00ff\u221e"
    p = CutPanel(id="001", panel_index=1, y_start=0, y_end=800,
                 narration=f"A man shouts {mojibake} in anger",
                 dialogue=f"{hangul} / WHERE THE HELL DO YOU THINK YOU ARE",
                 panel_type="single", confidence=0.9,
                 image_file="panel_001.png")
    art = CutArtifact(source="s.png", width=800, height=800, plan_hash="x",
                      config={}, panels=[p])
    rows = rs._panel_list_block(art)
    assert len(rows) == 1
    assert rows[0]["visual"] == "A man shouts in anger"
    assert rows[0]["dialogue"] == "WHERE THE HELL DO YOU THINK YOU ARE"
    # the rendered prompt row is pure ASCII: nothing for TTS to misread
    line = rs._panel_row(rows[0])
    assert all(ord(c) < 128 for c in line)


def test_build_writes_script_json_with_validated_lines(session):
    d, _ = session
    model = FakeModel([GOOD_LINES])
    res = rs.build_chapter_script(d, model_call=model)

    assert res["status"] == "built"
    assert res["used_fallback"] is False
    # panels 1/4/5 came from the model; 2 is caption-filled because the
    # gap-fill round had no scripted response left; 3 has nothing speakable
    # ("..." caption, no dialogue) so it stays unspoken by design.
    assert len(res["lines"]) == 4
    # validated mapping: panel_index -> panel_id from panels.json,
    # and lines are sorted into panel order (narrator speaks in order)
    assert [ln["panel_id"] for ln in res["lines"]] == \
        ["panel_001", "panel_002", "panel_004", "panel_005"]
    assert [ln["part"] for ln in res["lines"]] == \
        ["hook", "escalation", "escalation", "cliffhanger"]
    assert [ln["source"] for ln in res["lines"]] == \
        ["script", "caption_fill", "script", "script"]
    assert res["lines"][0]["quote"] == "Where am I?"
    assert res["lines"][2]["quote"] is None
    # text = joined line texts (TTS-ready, punctuation enforced)
    assert res["text"].startswith("Jin was an ordinary guy")
    # coverage audit
    cov = res["coverage"]
    assert (cov["panels"], cov["model_lines"], cov["caption_fill_lines"],
            cov["unspoken_panels"]) == (5, 3, 1, 1)
    # script.json on disk matches
    on_disk = json.loads((d / "script.json").read_text("utf-8"))
    assert on_disk["version"] == rs.SCRIPT_VERSION
    assert on_disk["input_hash"] == res["input_hash"]
    assert len(on_disk["lines"]) == 4


def test_gap_fill_covers_panels_the_main_call_missed(session):
    """A short main response triggers a bounded gap-fill pass, and the model's
    own words win over the caption fallback."""
    d, _ = session
    gap = json.dumps({"lines": [
        {"panel_index": 2, "text": "His guide tells him to follow.",
         "part": "setup", "quote": "Follow me."},
        {"panel_index": 3, "text": "Silence answers from the dark.",
         "part": "setup", "quote": None},
    ]})
    model = FakeModel([GOOD_LINES, gap])
    res = rs.build_chapter_script(d, model_call=model)

    by_id = {ln["panel_id"]: ln for ln in res["lines"]}
    assert by_id["panel_002"]["source"] == "gap_fill"
    assert by_id["panel_002"]["text"] == "His guide tells him to follow."
    assert by_id["panel_003"]["source"] == "gap_fill"
    assert res["coverage"]["caption_fill_lines"] == 0
    assert res["coverage"]["gap_fill_lines"] == 2
    # every panel of the chapter is now spoken
    assert res["coverage"]["unspoken_panels"] == 0
    assert len(res["lines"]) == 5
    # and the gap prompt carried only the missing panels + the running context
    gap_prompt = model.prompts[1]
    assert "still have NO" in gap_prompt
    assert "2. A man in a suit faces a door." in gap_prompt
    assert "1. A gray background" not in gap_prompt   # already covered
    assert "Jin was an ordinary guy" in gap_prompt    # context window


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


def test_prompt_truncation_keeps_json_contract_and_head_tail(
        session, monkeypatch):
    """A long chapter must shrink the PANEL LIST, never the prompt tail.

    The JSON-schema instructions the parser depends on sit AFTER
    {panel_list} in USER_PROMPT_TEMPLATE, so the old prompt[:MAX_PROMPT_CHARS]
    head-truncation deleted them exactly when they were most needed.
    """
    d, _ = session
    big = CutArtifact.model_validate_json((d / "panels.json").read_text("utf-8"))
    many = []
    for n in range(25):
        many.append(_panel(n + 1, n * 10, n * 10 + 10,
                           f"Panel number {n} does something dramatic and "
                           f"describes it at some length here."))
    big = big.model_copy(update={"panels": many})
    (d / "panels.json").write_text(big.model_dump_json(), "utf-8")

    # Force the truncation path deterministically (the default 60k budget is
    # generous enough that a 25-panel chapter fits untouched). The budget also
    # has to clear the fixed template that carries the title + English-only +
    # credit/atmosphere contracts (the template is ~4.9k chars now), so 6000 --
    # still far under 60k, so the many-panel list is genuinely truncated.
    monkeypatch.setattr(rs, "MAX_PROMPT_CHARS", 6000)

    model = FakeModel([GOOD_LINES])
    rs.build_chapter_script(d, model_call=model, force=True)
    prompt = model.prompts[0]

    assert len(prompt) <= 6000
    # The output contract survived: this is what used to be cut off.
    assert "Return STRICT JSON" in prompt, (
        "the JSON-schema instructions were truncated out of the prompt")
    # The panel list was elided in the MIDDLE, keeping head + tail (the hook
    # and the cliffhanger matter most), not head-only.
    assert "Panel number 0 " in prompt          # first panel kept
    assert "elided" in prompt                    # middle marker present
    assert "Panel number 24 " in prompt          # last panel kept


def test_panel_indices_out_of_range_are_dropped(session):
    d, _ = session
    bad = json.dumps({"lines": [
        {"panel_index": 99, "text": "phantom panel", "part": "hook"},
        {"panel_index": 1, "text": "Real line.", "part": "setup"},
        {"panel_index": 2, "text": "Second.", "part": "escalation"},
    ]})
    res = rs.build_chapter_script(d, model_call=FakeModel([bad]))
    assert res["status"] == "built"
    idx = [ln["panel_index"] for ln in res["lines"]]
    assert 99 not in idx                 # invented index never reaches the TTS
    # coverage still fills every speakable panel (3 has a "..." caption and no
    # dialogue, so nothing can be said over it)
    assert idx == [1, 2, 4, 5]
    assert res["coverage"]["model_lines"] == 2
    assert res["coverage"]["caption_fill_lines"] == 2


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
    # the narrator speaks in panel order: sorted; and the one-line-per-panel
    # contract keeps the FIRST line for a duplicated panel_index
    model_lines = [ln for ln in res["lines"] if ln["source"] == "script"]
    assert [ln["panel_index"] for ln in model_lines] == [1, 2, 4]
    assert [ln["text"] for ln in model_lines] == \
        ["earlier.", "middle one.", "later."]
    assert len(model_lines) == 3


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
    # main call + one gap-fill round (GOOD_LINES leaves panels 2/3 unspoken)
    assert len(model2.prompts) == 2


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


# ---------------------------------------------------------------- YouTube title
def test_build_captures_and_persists_title(session):
    d, _ = session
    resp = json.dumps({
        "title": "He Was The Weakest — Until He Logged In",
        "lines": [
            {"panel_index": 1, "text": "Jin was ordinary.",
             "part": "hook", "quote": None},
            {"panel_index": 4, "text": "Then it explodes.",
             "part": "escalation", "quote": None},
            {"panel_index": 5, "text": "A hand claws out.",
             "part": "cliffhanger", "quote": None}]})
    res = rs.build_chapter_script(d, model_call=FakeModel([resp]))
    assert res["status"] == "built"
    assert res["title"] == "He Was The Weakest — Until He Logged In"
    on_disk = json.loads((d / "script.json").read_text("utf-8"))
    assert on_disk["title"] == res["title"]


def test_build_without_title_is_gracefully_none(session):
    d, _ = session
    res = rs.build_chapter_script(d, model_call=FakeModel([GOOD_LINES]))
    assert res["status"] == "built"
    assert res["title"] is None


def test_clean_title_caps_at_word_boundary():
    out = rs._clean_title("word " * 40)
    assert len(out) <= rs._TITLE_MAX_CHARS
    assert not out.endswith(" ") and "wor" in out


def test_generate_recap_title_parses_primary_and_alts():
    resp = json.dumps({"title": "SSS-Rank Hunter Rises",
                       "alternatives": ["Alt one", "Alt two"]})
    res = rs.generate_recap_title("Some recap narration.",
                                  model_call=FakeModel([resp]))
    assert res["title"] == "SSS-Rank Hunter Rises"
    assert res["alternatives"] == ["Alt one", "Alt two"]


def test_generate_recap_title_empty_and_failure_are_none():
    assert rs.generate_recap_title("  ")["title"] is None
    # model_call raising is swallowed -> no crash, no title (bonus feature)
    boom = rs.generate_recap_title("text", model_call=FakeModel([]))
    assert boom == {"title": None, "alternatives": []}
