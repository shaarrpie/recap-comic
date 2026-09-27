# text_clean.py
"""Deterministic text hygiene for narrated recap text. English-only + promo.

Two jobs, both offline (no model, no network, no image needed):

1. **English-only.** The vision pass sometimes transcribes untranslated
   Korean/Chinese/Japanese bubbles verbatim, or emits mojibake for them (a
   Korean sound effect coming back as Greek-plus-accent soup). That text is
   spoken by edge-tts *literally*, so the narrator reads garbage on screen.
   The helpers here strip non-Latin runs from alt-text and drop whole
   speech-bubble segments that are not English, so only speakable English
   reaches the script pass and the TTS.

2. **Promo detection.** Scanlation groups also ship in-story ad splashes
   ("Read at GROUP.COM for the fastest releases"). They survive the page-level
   landscape drop because they are portrait crops, and the one-line-per-panel
   script contract then *forces* the narrator to invent a line for an ad, so
   every chapter ended by hyping a scan site. ``is_promo_text`` marks them so
   the filter can demote them to ``context_only`` (no frame, no line).

Deliberately conservative: every helper fails open (returns the input / says
"not promo") on anything ambiguous, because a false positive here silently
deletes story art from the recap.
"""
from __future__ import annotations

import re

__all__ = ["latin_ratio", "strip_non_latin", "english_bubbles", "clean_text",
           "is_promo_text", "promo_reason", "strip_promo"]


# ---------------------------------------------------------------- language --
# A letter is "English enough" when its lowercase form is ASCII a-z. That
# single test covers Hangul/Kana/CJK/Greek/Cyrillic *and* the Latin-1 soup
# produced by mojibake (ò, ÿ, ä all fail it), without needing a script table.
def _is_latin_letter(ch: str) -> bool:
    return "a" <= ch.lower() <= "z"


def latin_ratio(text: str | None) -> float:
    """Share of a string's letters that are ASCII Latin (1.0 when no letters).

    A string with no letters at all ("...", "1 2 3") is not foreign, it is
    non-lexical: callers already drop those elsewhere.
    """
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 1.0
    return sum(1 for c in letters if _is_latin_letter(c)) / len(letters)


# Anything outside ASCII word characters, common punctuation and the typographic
# marks English text actually uses. Korean/CJK/Greek/emoji all land here.
_ALLOWED = re.compile(
    "[^"
    "A-Za-z0-9"
    r"\s!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~"
    "\u2018\u2019\u201a\u201c\u201d\u201e\u2013\u2014\u2026\u00b7\u2022"
    "\u00a0\u00b0\u00b1\u00d7\u00f7\u2122\u2190-\u21ff"
    "]"
)


def _tidy(text: str) -> str:
    """Collapse the whitespace holes left by removing a foreign run."""
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"([,.!?]){2,}", r"\1", text)
    return text.strip(" ,;")


def strip_non_latin(text: str | None) -> str:
    """Remove non-Latin characters from alt-text, keeping the English parts.

    ``"He shouts <hangul> and leaves"`` becomes ``"He shouts and leaves"``.
    """
    if not text:
        return text or ""
    return _tidy(_ALLOWED.sub(" ", text))


# Segments below this Latin share are untranslated foreign text, not English
# with a stray symbol. 0.6 keeps "Doctor Min-jun, ìì´ì¤!" (mostly
# English) while dropping "ìì´ì¡... " outright.
_MIN_LATIN_RATIO = 0.6
_MIN_LATIN_LETTERS = 2


def english_bubbles(dialogue: str | None) -> tuple[str, int]:
    """Keep only the English speech-bubble segments of a dialogue string.

    The pipeline joins several bubbles with " / ", so filtering is per segment:
    an untranslated bubble is dropped instead of being spoken as garbage, and
    its English neighbours survive untouched.

    Returns (kept_text, dropped_segment_count).
    """
    if not dialogue:
        return dialogue or "", 0
    kept: list[str] = []
    dropped = 0
    for seg in dialogue.split("/"):
        piece = seg.strip()
        if not piece:
            continue
        letters = [c for c in piece if c.isalpha()]
        if (len([c for c in letters if _is_latin_letter(c)]) < _MIN_LATIN_LETTERS
                or latin_ratio(piece) < _MIN_LATIN_RATIO):
            dropped += 1
            continue
        kept.append(_tidy(_ALLOWED.sub(" ", piece)))
    return " / ".join(k for k in kept if k), dropped


def clean_text(narration: str | None,
               dialogue: str | None) -> tuple[str, str, int]:
    """English-only (alt-text, dialogue) + how many segments were dropped."""
    clean_narr = strip_non_latin(narration)
    clean_dial, dropped = english_bubbles(dialogue)
    return clean_narr, clean_dial, dropped


# ------------------------------------------------------------------- promo --
# A domain name in a panel's own text is the single most reliable scanlation
# marker: story dialogue never carries "asurascans.com".
_DOMAIN_RE = re.compile(
    r"\b[a-z0-9][a-z0-9-]{1,60}\."
    r"(?:com|net|org|io|gg|me|tv|cc|to|sh|app|site|online|club|xyz|top|one|"
    r"pro|life|world|plus)\b",
    re.IGNORECASE,
)
# Explicit group/promo vocabulary. Kept narrow on purpose: "read at" and
# "editor" are ordinary story words, so they are NOT here. Every alternative is
# \b-anchored: without it the credit-label group matched the "rd:" inside
# "FOG SWORD: RUSHING FOG STORM" -- a story technique card, and a false
# positive here silently deletes story art from the recap.
_PROMO_RE = re.compile(
    r"(?:\bfastest releases\b|\b(?:re-read|reread)\s+(?:it\s+)?(?:at|on|for)\b"
    r"|\bcontinue reading\b|\bends on novel chapter\b|\bpatreon\b|"
    r"\bko-?fi\b|\bpaypal\b|\bnovelupdates\b|\bmanga(?:dex|fox|anelo|zobo)\b|"
    r"\breelise\b|\bneox\b|\basurascans\b|\bscanlation\b|\bscanlator\b|"
    r"\braw (?:and|&) ?translation\b|\btypesetter\b|\bproofreader\b|"
    r"\bredraw(?:er)?\b|\bclean(?:er)? ?by\b|\b(?:tl|tc|ts|pr|cl|rd|qc|ed)\s*[:|]|"
    r"\bdiscord(?: server)?\b|\btelegram (?:channel|group)\b|"
    r"\bsupport (?:us|this group)\b|\bad-?free\b|\bno ads\b|"
    r"\bpremium (?:release|access)\b|\bearly access\b|\bhosted (?:at|by)\b|"
    r"\bchannel (?:name|logo)\b|\bwatermark\b)",
    re.IGNORECASE,
)


def promo_reason(text: str | None) -> str | None:
    """The matching token (for logs/tests), or None when the text is story."""
    blob = (text or "").strip()
    if not blob:
        return None
    hit = _DOMAIN_RE.search(blob) or _PROMO_RE.search(blob)
    return hit.group(0).strip() if hit else None


def is_promo_text(*texts: str | None) -> bool:
    """True when a panel's own text identifies it as a scanlation ad/credit.

    Text-based (not pixel-based) because promo splashes are drawn art: a purple
    paint-splash banner is colourful, so the filter's uniform-fill and
    text-on-white gates legitimately refuse to demote it.
    """
    blob = " ".join(t.strip() for t in texts if t and t.strip())
    if not blob:
        return False
    return bool(_DOMAIN_RE.search(blob) or _PROMO_RE.search(blob))


# Extra promo phrasing the safety-net scrub strips from narrator lines but
# that is deliberately NOT in ``_PROMO_RE`` (which also drives panel demotion,
# where a false positive silently deletes story art). Kept to unambiguous
# call-to-action wording so it never eats a story sentence.
_BRAND_PHRASE_RE = re.compile(
    r"(?:\basura\s*scans?\b|\breaper\s*scans?\b|\bread\s+more\b"
    r"|\bread\s+(?:more|first|it|them|ahead)\s+(?:at|on|to)\b"
    r"|\bsmash\s+(?:the\s+)?(?:like\s+and\s+)?subscribe\b"
    r"|\bsubscrib(?:e|ing)\b|\bhit\s+(?:the\s+)?(?:bell|like)\b"
    r"|\bfollow\s+(?:me|us|him|her)\b|\b(?:check|head)\s+(?:out|over)\s+to\b)",
    re.IGNORECASE,
)


def strip_promo(text: str | None) -> str:
    """Remove scanlation domain/promo tokens from a line, keeping story text.

    Reuses the exact vetted patterns behind ``is_promo_text`` (so the safety net
    can never drift from the detector) plus ``_BRAND_PHRASE_RE`` (call-to-action
    wording scrubbed here only, never used to demote a panel). Fails open on
    anything ambiguous and preserves ordinary sentence punctuation, because a
    false positive here would mangle the narrator's real lines.
    """
    if not text:
        return text or ""
    # domain FIRST: removing the bare group name (``asurascans``) before the
    # full ``asurascans.com`` would orphan the ".com" and leave "at.com".
    cleaned = _DOMAIN_RE.sub(" ", text)
    cleaned = _PROMO_RE.sub(" ", cleaned)
    cleaned = _BRAND_PHRASE_RE.sub(" ", cleaned)
    return _tidy(cleaned)
