# recap_script.py
"""Phase 2.5 — whole-chapter recap script pass.

WHY THIS EXISTS
---------------
Per-panel vision narration is (by design) alt-text: "a person in a white
shirt stands in front of a gray background". A recap is a GLOBAL
structure — hook -> setup -> escalation -> cliffhanger — and no single
panel call can see it. This module makes ONE text-model call that receives
the ordered list of {panel_index, visual_description, dialogue} for the
WHOLE chapter (plus the story memory seeded by story_context.py) and
writes narration that names characters, follows causality, and ends on a
cliffhanger. Sentences are then mapped back onto panel ranges so the
timeline, TTS and captions stay per-panel.

CONTRACT
--------
Input : a session dir containing panels.json (CutArtifact) whose panels
        carry the per-panel visual captions + dialogue.
Output: script.json next to panels.json:
        {
          "version": 1,
          "style": "recap",
          "generator": "recap_script.1",
          "model_used": "...",
          "fallback_used": false,
          "input_hash": "<sha of the visible inputs>",
          "lines": [
            {"panel_id": "panel_001", "panel_index": 1,
             "text": "Jinwoo wakes up in a hospital bed — again.",
             "part": "hook",          # hook|setup|escalation|cliffhanger
             "quote": null,           # verbatim dialogue, voiced separately
             "source": "script"}      # script|gap_fill|caption_fill
          ],
          "coverage":                # how the one-line-per-panel contract was
                                     # actually satisfied (audit trail)
            {"panels": 68, "model_lines": 61, "gap_fill_lines": 6,
             "caption_fill_lines": 1, "gap_fill_rounds": 1,
             "unspoken_panels": 0},
          "title": "He Was The Weakest Hunter — Until He Logged In",  # CTR
          "text": "<full script, one paragraph>",
          "structure": ["hook", "setup", "escalation", "cliffhanger"]
        }

COVERAGE
--------
Every usable panel gets exactly one spoken line. The whole-chapter call is
asked for that directly; whatever it misses is re-asked in bounded gap-fill
batches, and anything still missing is filled from the panel's own vision
caption. Only a panel with nothing speakable at all (non-lexical caption and
no dialogue) stays unspoken, and the timeline then keeps it as an auditable
short silent beat rather than dropping the frame.

FALLBACK
--------
No key / no model / model failure -> returns the joined-caption script
with used_fallback=True and NO lines (callers keep per-panel captions).
Never raises for a failed call; only raises for programmer errors
(missing panels.json, unparseable artifact).

CACHING
-------
script.json is skipped when its input_hash matches (panel narration +
dialogue + order + style + memory version). `force=True` rebuilds.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from guided_cutter import CutArtifact
from narrator_prompt import NARRATOR_STYLE_PROMPT
from text_clean import clean_text

log = logging.getLogger(__name__)

SCRIPT_FILE = "script.json"
# v2: narrator persona (narrator_prompt.NARRATOR_STYLE_PROMPT) joined the
# system prompt; v3: SWEAR BUDGET — severe profanity only in the opening
# line (~first seconds of audio), mild curses occasionally after that;
# v4: the pass also emits a CTR-optimised YouTube "title" alongside the
# lines (title is metadata, never spoken), so pre-title cached scripts
# regenerate once; v5: FULL COVERAGE — every panel must be spoken. The old
# "3-12 lines, skip panels that add nothing" contract left 57 of 68 panels
# with no line, which the timeline then rendered as 1s silent filler beats
# (57s of voiceless video, rushed pans). v5 demands one line per panel and
# backstops the model with a batched gap-fill pass plus a deterministic
# caption fill, so a scene panel can never be silent or skipped;
# v6: ENGLISH-ONLY — the panel list is scrubbed of non-Latin text (and
# untranslated bubbles are dropped) before the call, and the prompt forbids
# speaking URLs / scanlation promo text. Live data had the narrator reading
# "φòÿ∞òä..." aloud and closing every chapter on a "Read at ASURASCANS.COM
# for the fastest releases" splash, so pre-v6 scripts must regenerate.
SCRIPT_VERSION = 6
GENERATOR = "recap_script.6"

# Structure template for the model + for validating its mapping.
STRUCTURE = ["hook", "setup", "escalation", "cliffhanger"]

# Non-lexical "narration" strings the vision model emits for blank/silent
# panels. They must never reach the script or TTS.
_NON_LEXICAL = re.compile(r"^[\s.·•—–\-_*~…!?]*$")

# The "MANWA RECAP STORYTELLER" persona decides HOW the narrator talks
# (hype, jokes, roasts, hooks). The contract tail decides WHAT it emits:
# it must win over anything in the style guide that conflicts with the
# JSON pipeline contract or with TTS safety.
_STYLE_CONTRACT = (
    "\n\nPIPELINE CONTRACT (overrides any conflicting instruction above): "
    "You are the scriptwriter for a manhwa/webtoon recap video. You write "
    "the narrator's voice-over for ONE chapter, given an ordered list of "
    "panels with their visual descriptions and dialogue. Line count, panel "
    "order, and output shape come from the user message. Respond with ONE "
    "JSON object ONLY. No prose, no markdown, no code fences. Your lines go "
    "straight into TTS: never emit censor placeholders like [__]. "
    "SWEAR BUDGET (overrides the SWEARING guide above): severe profanity "
    "(fuck, shit, bitch, asshole, motherfucker and friends) is allowed "
    "ONLY in the very first spoken line — the hook, heard over the "
    "video's first seconds; every later line may use at most an "
    "occasional mild curse (damn, hell, crap, ass, bastard) and never a "
    "severe one. Style energy must never stretch a line past ONE short "
    "sentence (~14 words): every panel is spoken, so a long line is a "
    "truncated line."
)

SYSTEM_PROMPT = NARRATOR_STYLE_PROMPT + _STYLE_CONTRACT

USER_PROMPT_TEMPLATE = """\
Write the recap narration for this chapter.

You get, in reading order, each panel's visual description (written as
literal alt-text by a vision model) and its dialogue. The descriptions are
clipped and repetitive — your job is to see THROUGH them to the story.

STRUCTURE (mandatory):
1. hook — assigned to the FIRST panel: open with the dramatic question or
   cold-open stakes over the chapter's opening image ("Jinwoo was an
   ordinary office worker... until THIS happened."). Do not start by
   describing panel 1 literally.
2. setup — the next 1-3 lines: establish who/where/why.
3. escalation — the bulk: causality ("because X, Y"), stakes, names,
   reactions. Connect panels; never describe backgrounds.
4. cliffhanger — assigned to the LAST panel you use: end on the chapter's
   final beat, unresolved.

RULES:
- English ONLY. Every line must be speakable English: never copy non-Latin
  text (untranslated Korean/Chinese/Japanese bubbles, mojibake) and never
  narrate a site name, URL, app promo or scanlation credit — that is not the
  story. Foreign text is stripped from the panel list before you see it, so a
  non-Latin character in your output means you invented one.
- Use character NAMES from the dialogue/memory, not "a person" or "a man".
- Narrate STORY: what happens, why it matters, what changes. Never
  inventory what is visible ("a gray background with no characters").
- ONE LINE PER PANEL — mandatory, no exceptions. Every panel_index in the
  list gets exactly one line of its own; the narrator is never silent over
  a panel. Never merge panels, never skip panels, never write one line that
  covers several. Each line is ONE short sentence (max ~14 words) so it
  fits inside a ~4-second speech window — a long line gets truncated by the
  pipeline and the panel is left hanging.
- Voice: the hype manwa-recap storyteller from your instructions — casual,
  funny, reactive, opinionated. Present tense. Punchy.
  NEVER boring: every line moves plot, lands a joke, hypes a moment, or
  builds suspense.
- Swearing budget: severe words (fuck/shit/...) ONLY in the FIRST line
  (the hook); later lines get at most an occasional mild curse
  (damn/hell/crap), never a severe one.
- Do not invent events that contradict the panel list; inference of
  causality between shown events is allowed and expected.
- If the memory block names characters/threads, use those names and keep
  them consistent.
- TITLE: also output a single "title" field — a click-worthy YouTube recap
  title for the WHOLE chapter. <=70 chars, front-load the hook, open a
  curiosity gap, name the key character/rank, at most ONE ALL-CAPS power
  word, a number if natural. No clickbait lies, no emoji, no profanity, no
  censor placeholders. The title is metadata (never narrated), so it is
  exempt from the 1-2 short sentence line rule.

{memory_block}
Panel list (in order):
{panel_list}

Return STRICT JSON, exactly:
{{"title": "<one click-worthy YouTube title for the recap, <=70 chars>",
 "lines": [{{"panel_index": <int from the list>, "text": "<narrator line>",
              "part": "hook|setup|escalation|cliffhanger",
              "quote": "<one verbatim dialogue line to voice as a "
                       "character, or null>"}}]}}
Every panel_index MUST come from the list. Lines MUST be listed in
non-decreasing panel_index order (the narrator speaks in panel order;
hook/setup/escalation/cliffhanger are TONES, not a reordering of the
video). Emit EXACTLY one line for every panel in the list — the number of
lines must equal the number of panels, in the same order, with no panel
missing and none repeated.
"""

# Gap-fill: the whole-chapter call can run out of output budget on a long
# chapter (one line per 146-panel chapter is a lot of JSON), so any panel it
# left unspoken gets a second, small, bounded pass. Same persona, same rules,
# only the missing panels in the list.
GAP_FILL_PROMPT_TEMPLATE = """\
The recap for this chapter is almost finished, but these panels still have NO
narration line. The narrator must speak over every single panel, so write one
line for EACH panel below.

CONTEXT (the lines already recorded just before these panels, so your new
lines continue the story instead of repeating it):
{context}

{memory_block}
Panels still needing a line (in order):
{panel_list}

Return STRICT JSON, exactly:
{{"lines": [{{"panel_index": <int from the list>, "text": "<narrator line>",
              "part": "hook|setup|escalation|cliffhanger",
              "quote": "<one verbatim dialogue line to voice, or null>"}}]}}
One line per listed panel_index — no panel may be left out, none repeated,
listed in the same order. Each text is ONE short sentence (max ~14 words).
Narrate the story beat, never the artwork; keep established names consistent.
English only: never emit a non-Latin character, a URL or promo text.
"""

GAP_FILL_BATCH = 30          # panels per gap-fill call (bounded output size)
GAP_FILL_MAX_ROUNDS = 8      # calls per chapter before the caption fill
SCRIPT_MAX_TOKENS = 8192     # one line per panel needs far more than 4096

MAX_PANEL_CHARS = 4_000     # per-panel description cap in the prompt
MAX_PROMPT_CHARS = 60_000   # whole-prompt cap


def is_non_lexical(text: str | None) -> bool:
    """True when text carries no speakable words ("...", "—", "", "*")."""
    return bool(_NON_LEXICAL.match(text or ""))


def _usable(panel) -> bool:
    """Panels that can carry a script line."""
    if getattr(panel, "blank_flag", "normal") == "blank":
        return False
    # context_only panels: dialogue is story context (visible to the model
    # as context), but they get no frame — exclude from line mapping.
    return not getattr(panel, "context_only", False)


def _panel_list_block(artifact: CutArtifact) -> list[dict]:
    """Ordered {panel_index, visual, dialogue} blocks for the prompt."""
    out = []
    for p in sorted(artifact.panels,
                    key=lambda q: (q.panel_index, q.y_start)):
        if not _usable(p):
            continue
        # English-only at prompt time (not just at caption time): chapters cut
        # before the cleaner existed have mojibake and untranslated bubbles
        # cached in panels.json, and re-narrating them would cost a vision
        # call per panel. Cleaning here costs nothing and fixes old data too.
        visual, dialogue, _dropped = clean_text(p.narration, p.dialogue)
        visual = visual.strip()
        if is_non_lexical(visual):
            visual = "(no usable description)"
        visual = visual[:MAX_PANEL_CHARS]
        dialogue = dialogue.strip()
        out.append({
            "panel_index": p.panel_index,
            "panel_id": p.id,
            "visual": visual,
            "dialogue": dialogue,
        })
    return out


def _memory_block(ctx: dict | None, panels_max: int) -> str:
    """[STORY MEMORY] digest for the script prompt (compact form)."""
    if not ctx:
        return "(no prior story memory)"
    chars = ctx.get("characters") or {}
    locs = ctx.get("locations") or {}
    threads = [t for t in (ctx.get("story_threads") or [])
               if t.get("status") == "active"]
    events = ctx.get("key_events") or []
    parts: list[str] = []
    if chars:
        parts.append("Cast: " + "; ".join(
            f"{name} ({(c.get('role') or '?')})" + (
                f" — {(c.get('description') or '')[:80]}" if c.get("description") else "")
            for name, c in list(chars.items())[:12]))
    if locs:
        parts.append("Locations: " + ", ".join(list(locs)[:8]))
    if threads:
        parts.append("Open threads: " + " | ".join(
            (t.get("summary") or "")[:120] for t in threads[:5]))
    if events:
        parts.append("Recent events (oldest first): " + " | ".join(
            (e.get("summary") or "")[:100]
            for e in list(events)[-panels_max:][:10]))
    if not parts:
        return "(no prior story memory)"
    return "[STORY MEMORY]\n" + "\n".join(parts) + "\n[/STORY MEMORY]"


def _input_hash(artifact: CutArtifact, style: str, ctx: dict | None) -> str:
    import hashlib
    parts = [f"{p.id}|{p.panel_index}|{(p.narration or '').strip()}|"
             f"{(p.dialogue or '').strip()}|{getattr(p, 'context_only', False)}"
             for p in sorted(artifact.panels,
                             key=lambda q: (q.panel_index, q.y_start))]
    h = hashlib.sha256()
    h.update(("\n".join(parts) + f"\nstyle={style}").encode("utf-8"))
    if ctx:
        h.update(f"\nmem={len(ctx.get('characters') or {})}:"
                 f"{len(ctx.get('story_threads') or {})}".encode())
    return h.hexdigest()[:32]


def _extract_lines(raw: str, panels: list[dict],
                   *, source: str = "script") -> list[dict]:
    """Pull valid {"lines": [...]} entries out of a model response.

    Keeps the FIRST line per panel_index (the one-line-per-panel contract has
    no room for a duplicate), drops unknown indices, non-lexical text and
    junk, and returns them in panel order. Never raises: a malformed response
    simply yields fewer (or no) lines, and the caller decides what that means.
    """
    first, last = raw.find("{"), raw.rfind("}")
    if first == -1 or last <= first:
        return []
    try:
        obj = json.loads(raw[first:last + 1])
    except ValueError:
        return []
    lines = obj.get("lines")
    if not isinstance(lines, list):
        return []
    idx_by_index = {p["panel_index"]: p for p in panels}
    seen: set[int] = set()
    parsed: list[dict] = []
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        idx = ln.get("panel_index")
        text = ln.get("text")
        if not isinstance(idx, int) or idx not in idx_by_index:
            continue
        if idx in seen:
            continue
        if not isinstance(text, str) or is_non_lexical(text):
            continue
        seen.add(idx)
        text = " ".join(text.split())
        if text[-1:] not in ".!?…\"'”’":
            text += "."                    # TTS-ready lines in script.json
        part = ln.get("part")
        if part not in STRUCTURE:
            part = "escalation"
        quote = ln.get("quote")
        if not isinstance(quote, str) or is_non_lexical(quote):
            quote = None
        parsed.append({
            "panel_id": idx_by_index[idx]["panel_id"],
            "panel_index": idx,
            "text": " ".join(text.split()),
            "part": part,
            "quote": quote,
            "source": source,
        })
    # The narrator speaks in panel order (one TTS clip per panel). A model
    # that lists lines out of order is corrected by sorting, not rejected.
    parsed.sort(key=lambda ln: ln["panel_index"])
    return parsed


def _parse_response(raw: str, panels: list[dict]) -> dict | None:
    """Parse + validate the model's {"lines": [...]} against the panel list."""
    parsed = _extract_lines(raw, panels)
    if len(parsed) < 2:
        return None
    title = None
    first, last = raw.find("{"), raw.rfind("}")
    if first != -1 and last > first:
        try:
            obj = json.loads(raw[first:last + 1])
            raw_title = obj.get("title") if isinstance(obj, dict) else None
            if isinstance(raw_title, str):
                title = _clean_title(raw_title) or None
        except ValueError:
            pass
    return {"lines": parsed, "title": title}


def _caption_fill_line(p: dict) -> str | None:
    """Deterministic last-resort line for a panel no model pass covered.

    Falls back to the panel's own vision caption, then to its dialogue. Raw
    alt-text is not the persona voice, but silence over a visible panel is
    worse: the contract is that every panel is spoken. Returns None only when
    the panel has nothing speakable at all (non-lexical caption AND no
    dialogue) — the timeline then keeps it as an auditable silent beat.
    """
    visual = (p.get("visual") or "").strip()
    if visual and visual != "(no usable description)" \
            and not is_non_lexical(visual):
        text = visual
    else:
        # "/"-separated bubbles read as one continuous line; strip quotes so
        # TTS does not announce them as narration.
        text = re.sub(r"[\"“”]+", " ", (p.get("dialogue") or ""))
        text = re.sub(r"\s*/\s*", " ", text)
    text = " ".join(text.split())
    if not text or is_non_lexical(text):
        return None
    if text[-1:] not in ".!?…\"'”’":
        text += "."
    return text


def _panel_row(p: dict) -> str:
    """One numbered "N. visual [dialogue: ...]" line for a panel list."""
    return (f"{p['panel_index']}. {p['visual']}"
            + (f'  [dialogue: {p["dialogue"]}]' if p["dialogue"] else ""))


def _panel_rows(batch: list[dict]) -> str:
    """The numbered panel block both prompts send to the model."""
    return "\n".join(_panel_row(p) for p in batch)


def _ensure_coverage(lines: list[dict], panels: list[dict], ctx: dict | None,
                     call: Callable[[str], str] | None,
                     ) -> tuple[list[dict], dict[str, int]]:
    """Guarantee one line per usable panel; never drop a panel for silence.

    The whole-chapter call is asked for full coverage, but it can still come
    back short (output-token limit, refusal, truncated JSON, a 429 that eats
    one of the two models). Rather than let those panels go silent, this
    re-asks in bounded batches (``GAP_FILL_MAX_ROUNDS`` x ``GAP_FILL_BATCH``)
    and finally fills from the panel's own caption. Every line records which
    stage produced it, so a recap can be audited for how much of it was
    model-written vs caption-filled.
    """
    by_index = {ln["panel_index"]: ln for ln in lines}
    missing = [p for p in panels if p["panel_index"] not in by_index]
    filled_by_gap, filled_by_caption = 0, 0
    rounds = 0
    while missing and call is not None and rounds < GAP_FILL_MAX_ROUNDS:
        batch = missing[:GAP_FILL_BATCH]
        rounds += 1
        prior = sorted(by_index.values(), key=lambda e: e["panel_index"])
        context = " | ".join(ln["text"] for ln in prior[-6:]) \
            or "(none yet — this is the opening of the chapter)"
        prompt = GAP_FILL_PROMPT_TEMPLATE.format(
            context=context, memory_block=_memory_block(ctx, len(batch)),
            panel_list=_panel_rows(batch))
        try:
            raw = call(prompt)
        except Exception as exc:  # noqa: BLE001 - coverage degrades, never dies
            log.warning("[script] gap-fill round %d failed (%s); falling back "
                        "to captions", rounds, exc)
            break
        got = _extract_lines(raw, batch, source="gap_fill")
        for ln in got:
            if ln["panel_index"] in by_index:
                continue
            by_index[ln["panel_index"]] = ln
            filled_by_gap += 1
        missing = [p for p in panels if p["panel_index"] not in by_index]
        log.info("[script] gap-fill round %d: +%d lines (%d panels still "
                 "unspoken)", rounds, len(got), len(missing))

    for p in panels:
        if p["panel_index"] in by_index:
            continue
        text = _caption_fill_line(p)
        if not text:
            continue
        by_index[p["panel_index"]] = {
            "panel_id": p["panel_id"], "panel_index": p["panel_index"],
            "text": text, "part": "escalation", "quote": None,
            "source": "caption_fill"}
        filled_by_caption += 1
    out = [by_index[i] for i in sorted(by_index)]
    stats = {"panels": len(panels), "model_lines": len(lines),
             "gap_fill_lines": filled_by_gap,
             "caption_fill_lines": filled_by_caption,
             "gap_fill_rounds": rounds,
             "unspoken_panels": len(panels) - len(out)}
    log.info("[script] coverage %d/%d panels spoken (gap_fill=%d "
             "caption_fill=%d silent=%d)", len(out), len(panels),
             filled_by_gap, filled_by_caption, stats["unspoken_panels"])
    return out, stats


def _fallback_join(artifact: CutArtifact, style: str) -> str:
    """Old behaviour: joined per-panel captions (dedup + non-lexical filter)."""
    from narrator import make_script_from_cut
    return make_script_from_cut(artifact, style=style)


def _load_cached(session: Path, input_hash: str) -> dict | None:
    path = session / SCRIPT_FILE
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if (cached.get("input_hash") == input_hash
            and cached.get("version") == SCRIPT_VERSION
            and isinstance(cached.get("lines"), list)
            and cached["lines"]):
        return cached
    return None


# --------------------------------------------------------------- YouTube title
# A click-worthy title is metadata (never narrated/TTS'd), so it lives beside
# the recap lines. The whole-chapter pass emits it in its single JSON call;
# generate_recap_title() also derives one on demand from an existing recap text
# without rewriting the lines (used by `guided title`).
_TITLE_MAX_CHARS = 80
_TITLE_SYSTEM = (
    "You write high-CTR YouTube titles for anime/manhwa recap videos. Given "
    "the recap narration for one chapter, output ONLY a JSON object (no "
    "markdown, no fences) shaped: "
    '{"title": "...", "alternatives": ["...", "..."]}. '
    "Rules: <=70 characters, front-load the hook, open a curiosity gap, name "
    "the key character/rank, at most ONE ALL-CAPS power word, a number where "
    "natural, NEVER clickbait lies, emoji, profanity, or censor placeholders "
    "like [__]. One primary title + two alternatives."
)
_TITLE_USER = (
    "Series: {series}\n\nRecap narration for this chapter:\n{text}\n\n"
    "Return the JSON object now."
)


def _clean_title(raw: str) -> str:
    """Normalise a model title: collapse whitespace, strip wrapping quotes,
    cap length at the last word boundary under _TITLE_MAX_CHARS."""
    t = re.sub(r"\s+", " ", (raw or "").strip()).strip(" \"'\u201c\u201d")
    if len(t) <= _TITLE_MAX_CHARS:
        return t
    cut = t[:_TITLE_MAX_CHARS]
    trimmed = cut.rsplit(" ", 1)[0].rstrip(" ,;:.-")  # never clip a word
    return trimmed or cut.rstrip()


def generate_recap_title(
    chapter_text: str,
    *,
    series_title: str = "",
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "",
    request_fn: Any | None = None,
    model_call: Any | None = None,
) -> dict[str, Any]:
    """Return {"title": str|None, "alternatives": [str, ...]} for a recap.

    One text-model call over the finished narration. Best-effort: any failure
    (no key, model error, unparseable) degrades to {"title": None, ...} and
    NEVER raises, mirroring the script pass's no-crash contract.
    """
    empty: dict[str, Any] = {"title": None, "alternatives": []}
    if not chapter_text.strip():
        return empty
    prompt = _TITLE_USER.format(series=series_title or "(unknown)",
                                text=chapter_text[:4000])
    try:
        if model_call is not None:
            raw = model_call(prompt)
        else:
            from adapters import ai_models as _ai
            outcome = _ai.generate_text_with_fallback(
                _TITLE_SYSTEM, prompt, operation="recap-title",
                api_key=api_key, base_url=base_url,
                primary_model=model or _ai.PRIMARY_MODEL,
                request_fn=request_fn)
            raw = outcome.result
        first, last = raw.find("{"), raw.rfind("}")
        if first == -1 or last <= first:
            raise ValueError("no JSON object in title response")
        obj = json.loads(raw[first:last + 1])
        raw_title = obj.get("title")
        title = _clean_title(raw_title) if isinstance(raw_title, str) else ""
        alts = [_clean_title(a) for a in
                (obj.get("alternatives") or []) if isinstance(a, str)]
        alts = [a for a in alts if a][:3]
        log.info("[script] YouTube title: %s", title or "(none)")
        return {"title": title or None, "alternatives": alts}
    except Exception as exc:  # noqa: BLE001 - title is a bonus, never fatal
        log.warning("[script] title generation failed (%s)", exc)
        return empty


def build_chapter_script(
    session_dir: str | Path,
    *,
    style: str = "recap",
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "",
    request_fn: Callable[..., str] | None = None,
    model_call: Callable[[str], str] | None = None,
    force: bool = False,
    artifact: CutArtifact | None = None,
) -> dict[str, Any] | None:
    """Build (or reuse) the whole-chapter script for a session.

    Returns a summary dict:
        {"status": "built"|"cached"|"fallback",
         "used_fallback": bool, "lines": [...], "text": "...",
         "model_used": str, ...}
    or None when panels.json is missing/unusable (offline mode).
    """
    session = Path(session_dir)
    if artifact is None:
        panels_path = session / "panels.json"
        if not panels_path.is_file():
            return None
        artifact = CutArtifact.model_validate_json(
            panels_path.read_text("utf-8"))

    ctx: dict | None = None
    try:
        from story_context import load_context
        ctx = load_context(session)
    except Exception:  # noqa: BLE001
        ctx = None

    input_hash = _input_hash(artifact, style, ctx)
    if not force:
        cached = _load_cached(session, input_hash)
        if cached is not None:
            log.info("[script] cache hit (%d lines)", len(cached["lines"]))
            return {"status": "cached", "used_fallback": False,
                    **cached}

    panels = _panel_list_block(artifact)
    has_text = any(p["visual"] != "(no usable description)"
                   or p["dialogue"] for p in panels)
    # A caller passing api_key=None means "use the configured pool", not
    # "run offline" — that is exactly how every CLI/webapp path calls this.
    # Checking only `api_key is not None` used to send a bare
    # build_chapter_script(dir) straight to the joined-caption fallback even
    # with AGNES_API_KEY set, and never even attempted the model. The pool is
    # consulted (never a single resolved key) so rotation stays intact.
    from adapters import ai_models as _ai
    can_call = bool(model_call is not None or request_fn is not None
                    or _ai.api_key_pool(api_key))
    if not has_text or not can_call:
        # offline / fallback path: keep the old joined captions
        text = _fallback_join(artifact, style)
        log.info("[script] fallback join (%d chars, has_text=%s, can_call=%s)",
                 len(text), has_text, can_call)
        return {"status": "fallback", "used_fallback": True,
                "lines": [], "text": text, "structure": [],
                "title": None,
                "model_used": "", "input_hash": input_hash}

    panel_lines = [_panel_row(p) for p in panels]

    def _assemble(block: str) -> str:
        return USER_PROMPT_TEMPLATE.format(
            memory_block=_memory_block(ctx, len(panels)), panel_list=block)

    prompt = _assemble("\n".join(panel_lines))
    if len(prompt) > MAX_PROMPT_CHARS:
        # Truncate the PANEL LIST, never the assembled prompt: the
        # "Return STRICT JSON, exactly: …" contract the parser depends on
        # sits AFTER {panel_list} in the template, so the old
        # prompt[:MAX_PROMPT_CHARS] head-truncation deleted exactly those
        # instructions on long chapters and the model started emitting
        # prose. Keep the first and last panels (hook + cliffhanger) and
        # elide the redundant middle.
        budget = MAX_PROMPT_CHARS - len(_assemble(""))
        elided = "…(middle panels elided to fit the prompt budget)…"
        block = ""
        for keep in range(len(panel_lines), 0, -1):
            half = keep // 2
            cand = (panel_lines[:half]
                    + ([elided] if keep < len(panel_lines) else [])
                    + panel_lines[len(panel_lines) - (keep - half):])
            block = "\n".join(cand)
            if len(block) <= budget:
                break
        else:
            block = panel_lines[0][:max(0, budget)]
        prompt = _assemble(block)
        log.info("[script] panel list trimmed %d -> fit %d chars",
                 len(panel_lines), len(block))

    model_used, fb = "", False

    def _call(gap_prompt: str) -> str:
        """One model request (used by the gap-fill rounds)."""
        if model_call is not None:
            return model_call(gap_prompt)
        from adapters import ai_models as _ai
        outcome = _ai.generate_text_with_fallback(
            SYSTEM_PROMPT, gap_prompt,
            operation="chapter-script-gap-fill",
            api_key=api_key, base_url=base_url,
            primary_model=model or _ai.PRIMARY_MODEL,
            max_tokens=SCRIPT_MAX_TOKENS,
            request_fn=request_fn)
        return outcome.result

    try:
        if model_call is not None:
            raw = model_call(prompt)
        else:
            from adapters import ai_models as _ai
            outcome = _ai.generate_text_with_fallback(
                SYSTEM_PROMPT, prompt,
                operation="chapter-script",
                api_key=api_key, base_url=base_url,
                primary_model=model or _ai.PRIMARY_MODEL,
                # one line per panel: a 146-panel chapter needs far more
                # output than the 4096 default (and these are reasoning
                # models, which spend part of the budget on hidden tokens)
                max_tokens=SCRIPT_MAX_TOKENS,
                request_fn=request_fn)
            raw = outcome.result
            model_used, fb = outcome.model_used, outcome.fallback_used
    except Exception as exc:  # noqa: BLE001 - fall back, never kill the run
        log.warning("[script] model call failed (%s); using joined captions",
                    exc)
        text = _fallback_join(artifact, style)
        return {"status": "fallback", "used_fallback": True,
                "lines": [], "text": text, "structure": [],
                "title": None,
                "model_used": "", "input_hash": input_hash}

    parsed = _parse_response(raw, panels)
    if parsed is None:
        log.warning("[script] unparseable/invalid model response; using "
                    "joined captions")
        text = _fallback_join(artifact, style)
        return {"status": "fallback", "used_fallback": True,
                "lines": [], "text": text, "structure": [],
                "title": None,
                "model_used": "", "input_hash": input_hash}

    # Full-coverage enforcement: every usable panel must carry a line, so the
    # narrator never pauses over visible art (gap-fill batches, then captions).
    lines, coverage = _ensure_coverage(parsed["lines"], panels, ctx, _call)

    script_text = " ".join(ln["text"] for ln in lines)
    result = {
        "version": SCRIPT_VERSION,
        "style": style,
        "generator": GENERATOR,
        "model_used": model_used or "custom",
        "fallback_used": bool(fb),
        "input_hash": input_hash,
        "lines": lines,
        "coverage": coverage,
        "title": parsed.get("title"),
        "text": script_text,
        "structure": STRUCTURE,
    }
    path = session / SCRIPT_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                   "utf-8")
    tmp.replace(path)
    log.info("[script] built %d lines over %d panels, %d chars, parts=%s",
             len(lines), coverage["panels"], len(script_text),
             [ln["part"] for ln in lines])
    return {"status": "built", "used_fallback": False, **result}


def load_script(session_dir: str | Path) -> dict | None:
    """Load a previously built script.json (None when absent/broken)."""
    path = Path(session_dir) / SCRIPT_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if (isinstance(data.get("lines"), list) and data["lines"]
            and data.get("version") == SCRIPT_VERSION):
        return data
    return None
