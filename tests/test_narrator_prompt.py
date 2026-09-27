# tests/test_narrator_prompt.py
"""Offline tests for the narrator persona wiring (narrator_prompt.py).

The "MANWA RECAP STORYTELLER" style guide must reach the two places that
write SPOKEN text (the whole-chapter script pass and the webapp single-panel
regeneration) WITHOUT breaking the JSON pipeline contract, and must NOT
leak into the per-panel vision prompt (which stays literal alt-text).
"""
from __future__ import annotations

import json

import narrator_prompt as np_
import recap_script as rs

PERSONA_MARKERS = [
    "MANWA RECAP STORYTELLER",
    "NEVER BE BORING",
    "HYPE MODE",
    "our boy",
    "I CALLED IT",
    "FINAL CHECKLIST",
]


def test_persona_prompt_embedded_verbatim():
    for marker in PERSONA_MARKERS:
        assert marker in np_.NARRATOR_STYLE_PROMPT
    # sanity: the big example scene survived the embed
    assert "Like Team Rocket" in np_.NARRATOR_STYLE_PROMPT


def test_script_system_prompt_is_persona_plus_contract():
    sys_p = rs.SYSTEM_PROMPT
    assert sys_p.startswith(np_.NARRATOR_STYLE_PROMPT)
    # the old one-line scriptwriter contract still holds, as a tail guard
    assert "PIPELINE CONTRACT" in sys_p
    assert "ONE JSON object ONLY" in sys_p
    # TTS safety: censor placeholders must never be emitted
    assert "[__]" in sys_p  # mentioned only to forbid it
    assert "never emit censor placeholders" in sys_p
    # SWEAR BUDGET: severe profanity is pinned to the opening line
    assert "SWEAR BUDGET" in sys_p
    assert "ONLY in the very first spoken line" in sys_p
    assert "mild curse" in sys_p
    # contract wins over the style guide's length-free examples: one spoken
    # line per panel means one short sentence per line
    assert "ONE short sentence" in sys_p


def test_user_template_points_at_the_persona():
    tpl = rs.USER_PROMPT_TEMPLATE
    assert "hype manwa-recap storyteller" in tpl
    # full-coverage contract: no panel may be left unspoken
    assert "ONE LINE PER PANEL" in tpl
    assert "EXACTLY one line for every panel" in tpl
    assert "skip panels that add nothing" not in tpl
    # English-only contract: nothing non-Latin or promotional may be spoken
    assert "English ONLY" in tpl
    assert "narrate a site name, URL, app promo or scanlation credit" in tpl
    # credit/logo carve-out must name its precedence over one-line-per-panel
    assert 'OUTRANKS "one line per panel."' in tpl
    assert "CREDIT / LOGO / BANNER PANELS" in tpl
    # atmosphere panels must be redirected from visual inventory to carried
    # tension, not banned outright (coverage still needs a line per panel)
    assert "ATMOSPHERE PANELS" in tpl
    # the gap-fill pass carries the same rule, or coverage reintroduces it
    assert "English only" in rs.GAP_FILL_PROMPT_TEMPLATE
    # the old anti-style rule that fought the persona is gone
    assert "No flowery prose" not in tpl
    # swearing budget repeated where the lines are actually written
    assert "Swearing budget" in tpl
    assert "ONLY in the FIRST line" in tpl
    # structural contract untouched
    for part in rs.STRUCTURE:
        assert part in tpl


def test_vision_prompt_stays_literal():
    """Per-panel vision alt-text must NOT inherit the persona."""
    from adapters.ai_narration import PANEL_NARRATION_PROMPT
    assert "MANWA RECAP STORYTELLER" not in PANEL_NARRATION_PROMPT
    assert "ONE short simple sentence" in PANEL_NARRATION_PROMPT


def test_version_bump_invalidates_pre_persona_cache(tmp_path):
    """Old v1-v5 script.json must not replay. v4 and earlier were written under
    the "3-12 lines, skip panels" contract (the dead-air cause); v5 predates the
    English-only rule, so its lines can contain a URL or mojibake the narrator
    was forced to speak."""
    assert rs.SCRIPT_VERSION == 7
    for stale_version in (1, 2, 3, 4, 5, 6):
        stale = {"version": stale_version, "style": "recap",
                 "input_hash": "x" * 32,
                 "lines": [{"panel_id": "p1", "panel_index": 1,
                            "text": "old", "part": "hook",
                            "quote": None}],
                 "text": "old", "structure": rs.STRUCTURE}
        (tmp_path / "script.json").write_text(json.dumps(stale), "utf-8")
        assert rs.load_script(tmp_path) is None
        tmp_path.joinpath("script.json").unlink()


def test_webapp_regen_system_prompt_carries_persona():
    from webapp.narration_api import _regen_system
    sys_p = _regen_system()
    assert sys_p.startswith(np_.NARRATOR_STYLE_PROMPT)
    assert "Output ONLY JSON" in sys_p
    assert "never emit censor placeholders" in sys_p
    assert "ONE panel" in sys_p


def test_webapp_regen_swear_budget_tracks_opening_panel():
    from webapp.narration_api import _regen_system
    opening = _regen_system(first_panel=True)
    assert "this IS the video's opening line" in opening
    mid = _regen_system(first_panel=False)
    assert "allowed ONLY in the video's opening line" in mid
    assert "mild curse" in mid
    assert opening != mid
