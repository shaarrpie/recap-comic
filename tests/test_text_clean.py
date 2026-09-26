# tests/test_text_clean.py
"""Offline tests for the English-only + scanlation-promo text rules.

Every case here comes from real pipeline output: a mangled Korean sound effect
that edge-tts read aloud, and a "Read at ASURASCANS.COM" splash the
one-line-per-panel contract forced the narrator to hype. Foreign characters are
written as escapes so this file stays ASCII -- a test for encoding hygiene
cannot itself depend on the editor's encoding.
"""
from __future__ import annotations

import pytest

from text_clean import (
    clean_text,
    english_bubbles,
    is_promo_text,
    latin_ratio,
    promo_reason,
    strip_non_latin,
)

# "\uc544\uc774\uc2a1" = Korean aisik; "\u03c6\u00f2\u00ff\u221e" = the
# phi/o-grave/y-diaeresis/infinity soup a mangled sound effect looks like.
HANGUL = "\uc544\uc774\uc2a1"
MOJIBAKE = "\u03c6\u00f2\u00ff\u221e\u00f2\u00e4"


# ---------------------------------------------------------------- language --
def test_latin_ratio_separates_english_from_foreign():
    assert latin_ratio("WHERE THE HELL DO YOU THINK YOU'RE GOING?") == 1.0
    assert latin_ratio(HANGUL) == 0.0
    assert latin_ratio("") == 1.0          # no letters is not "foreign"
    assert latin_ratio(None) == 1.0
    assert latin_ratio("......?") == 1.0   # non-lexical, handled elsewhere
    # 3 Hangul chars + "OK": mostly foreign, and the bubble rule must see it
    # that way (the threshold sits at 0.6).
    assert 0.0 < latin_ratio(f"{HANGUL} OK") < 0.6


def test_strip_non_latin_keeps_english_and_typography():
    assert strip_non_latin(f"He shouts {HANGUL} and leaves") \
        == "He shouts and leaves"
    assert strip_non_latin(MOJIBAKE) == ""
    # Curly quotes / ellipsis / em dash are English punctuation, not foreign.
    pretty = "Don\u2019t go \u2014 it\u2019s over\u2026"
    assert strip_non_latin(pretty) == pretty
    assert strip_non_latin("") == ""
    assert strip_non_latin(None) == ""


def test_english_bubbles_drops_only_the_foreign_segments():
    dialogue = f"{HANGUL} / WHO MADE YOU DO THIS? / {MOJIBAKE}"
    kept, dropped = english_bubbles(dialogue)
    assert kept == "WHO MADE YOU DO THIS?"
    assert dropped == 2


def test_english_bubbles_keeps_a_bubble_with_one_stray_symbol():
    # Mostly-English with a single untranslated word must survive: dropping it
    # would silently remove story dialogue.
    kept, dropped = english_bubbles(f"Doctor, watch out! {HANGUL}")
    assert dropped == 0
    assert "watch out" in kept


def test_english_bubbles_all_foreign_becomes_empty_not_garbage():
    kept, dropped = english_bubbles(f"{HANGUL} / {HANGUL}")
    assert kept == ""
    assert dropped == 2
    # No dialogue at all is a no-op (never an error, never invented text).
    assert english_bubbles("") == ("", 0)
    assert english_bubbles(None) == ("", 0)


def test_clean_text_is_idempotent():
    once_n, once_d, _ = clean_text(f"Panel {MOJIBAKE} art", f"HI / {HANGUL}")
    twice_n, twice_d, dropped = clean_text(once_n, once_d)
    assert (twice_n, twice_d) == (once_n, once_d)
    assert dropped == 0
    assert once_n == "Panel art" and once_d == "HI"


# ------------------------------------------------------------------- promo --
@pytest.mark.parametrize("text", [
    "Read at ASURASCANS.COM for the fastest releases",
    "CONTINUE READING THE NOVEL AT ASURASCANS.COM/NOVELS",
    # The vision model truncates the URL off the end of the bubble, so the
    # phrase alone has to carry it (real miss in gallery chapter 4).
    'ENDS ON NOVEL CHAPTER 5 - "CRISIS (2)" / CONTINUE READING THE NOVEL AT',
    "A purple and white promotional banner for AsuraScans.com",
    "Support the creator on patreon",
    f"{HANGUL} / asurascans.com",
    "TL: RandomGuy QC: Someone",
])
def test_promo_text_detects_scanlation_cards(text):
    assert is_promo_text(text) is True
    assert promo_reason(text)


@pytest.mark.parametrize("text", [
    "FOG SWORD: RUSHING FOG STORM",          # the bug this rule first shipped with
    "FOG SWORD : EIGHT FLUTTERING PETALS",
    "IS HE A HIGHER UP?",
    "Unshakable Roots of a Great Tree",
    "I read at the table every night.",
    "She kept reading the letter in silence.",      # "reading" alone is story
    "You will reread this duel in your dreams.",   # "reread" alone is story
    "The editor of the realm has spoken.",   # "editor" alone is a story word
    "(no usable description)",
    "",
    None,
])
def test_promo_text_never_eats_story_text(text):
    assert is_promo_text(text) is False
    assert promo_reason(text) is None


def test_is_promo_text_checks_every_field_and_fails_open():
    # The URL may live in the alt-text rather than the bubble (or vice versa).
    assert is_promo_text("", "banner reading ASURASCANS.COM/NOVELS") is True
    assert is_promo_text("ASURASCANS.COM", "") is True
    assert is_promo_text(None, None, "") is False
    assert is_promo_text("HE DREW HIS SWORD") is False
