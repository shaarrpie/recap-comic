# story_context.py v3 -- Persistent narrative memory for the recap-comic pipeline.
"""Persistent narrative memory for the recap pipeline.

Project-root module. Designed to integrate with adapters/ai_narration.py;
see the INTEGRATION section at the bottom of this file for the exact,
signature-checked patch.

WHAT IT DOES
-----------
Maintains a session-level story_context.json that grows as the vision
model processes panels:

  * Pass 0 (build_seed_context) -- runs ONCE before narration. Sends all
    panel dialogue/narration text (INCLUDING context_only panels -- that is
    exactly why panel_filter keeps them) to the model for an initial
    entity roster.
  * Pass 1 (per panel) -- the narration prompt is augmented with a
    [STORY MEMORY] block; the model's response may carry an "entities"
    key inside the SAME strict-JSON object it already returns; those
    entities are merged into the context after each panel.

CHANGES vs v2 (v2 review issues all addressed)
---------------------------------------------
R1. inject_into_prompt() rebuilt: size ladder (10/5/5/3 -> 6/3/4/2 ->
    4/2/3/1 -> 2/1/2/0 -> 1/0/1/0 characters/locations/threads/events)
    instead of whole-section dropping. Threads are NEVER dropped before
    characters (cheapest continuity signal); the closing delimiter and
    continuity instruction are never cut off; the block is never empty
    while the context is non-empty; default cap raised to 1200 chars;
    descriptions truncated to 100 chars inside prompts (stored untouched).
R2. make_text_model_call() adapter: wraps
    adapters.ai_models.generate_text_with_fallback(system, user) correctly
    and returns outcome.result (a str) -- the v2 patch lambda passed one
    argument to a two-argument function and ignored the result object.
R3. extract_entities_from_response() accepts the parsed dict OR the raw
    response string (outermost-JSON extraction), so _parse_narration()
    in ai_narration.py needs NO signature change.
R4. last_seen == 0 is a "seeded, not yet seen on-panel" sentinel: ranked
    fresh for injection (the v2 code demoted every seeded character to
    stale, defeating the seed pass) and rendered consistently.
R5. key_events now actually injected (brief "Recent events" digest,
    newest first) -- the v2 changelog claimed this but never built it.
R6. Thread resolve/dedupe uses word-overlap matching (>= 0.6 of the
    hint's non-stopword tokens, minimum 2 words) instead of raw
    substring, so a one-word hint ("rachel") cannot close a thread.
R7. A future-version story_context.json is backed up to
    story_context.json.bak before being reset (v2 silently destroyed it).
R8. carry_forward() copies a previous chapter's context into a new
    session and RESETS per-chapter counters (first_seen/last_seen/key_events)
    so carried panel numbers cannot poison the new chapter's staleness
    math; carried threads stay (they span chapters) and the new chapter
    re-seeds (seed_built=False).
R9. ENTITIES_SCHEMA_FRAGMENT is pure JSON (no // comments that a
    literal-minded model would copy into invalid JSON). scrub_fences()
    is public (the v2 patch imported a private name).
R10. update_context() isolates failures PER FIELD: one corrupt field
     (e.g. a non-numeric _meta.last_panel) no longer discards the whole
     panel's entity update.
Plus: role defaults survive sanitisation; new_locations single lookup;
seed text-block cap checked BEFORE appending the overflowing line.

SESSION CONTINUITY
------------------
story_context.json is per session_dir by design. To chain chapters, call
carry_forward(prev_session_dir, new_session_dir, chapter=N) BEFORE
build_seed_context: the roster and open threads carry over, per-chapter
counters reset, and the new chapter's text re-seeds the cast.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CONTEXT_FILE = "story_context.json"
_VERSION = 2
_MAX_DESC_CHARS = 200        # max chars stored for any description / event
_MAX_PROMPT_DESC_CHARS = 100  # max description chars inside a prompt block
_MAX_TEXT_BLOCK_CHARS = 12_000  # seed text cap
_MAX_KEY_EVENTS = 50          # cap key_events list (newest kept)
_STALENESS_PANELS = 60        # panels since last_seen before "stale"
_THREAD_MATCH_RATIO = 0.6    # word-overlap ratio to resolve/dedupe threads

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "and",
    "or", "for", "his", "her", "their", "its", "with", "at", "by", "on",
    "as", "be", "been", "this", "that", "it",
})


# ---------------------------------------------------------------------------
# Entity schema fragment (pure JSON -- paste into PANEL_NARRATION_PROMPT)
# ---------------------------------------------------------------------------
ENTITIES_SCHEMA_FRAGMENT = """\
  "entities": {
    "characters_seen": ["<known character name visible/mentioned>"],
    "new_characters": [{"name": "...", "role": "protagonist|antagonist|ally|minor", "description": "...", "aliases": ["..."]}],
    "locations_seen": ["<known location name>"],
    "new_locations": [{"name": "...", "description": "..."}],
    "threads_resolved": ["<short description of an ongoing thread that is now closed>"],
    "threads_new": ["<one-line new plot thread>"],
    "new_events": ["<key plot event one-liner>"]
  }"""

ENTITIES_RULE = (
    'Include the optional "entities" key ONLY when this panel adds or '
    "updates story facts; omit the key entirely otherwise. Never wrap "
    "your response in markdown fences."
)

# Injection size ladder: (max_characters, max_locations, max_threads,
# max_events). Tried top-down until the block fits max_chars.
_LADDER = [
    (10, 5, 5, 3),
    (6, 3, 4, 2),
    (4, 2, 3, 1),
    (2, 1, 2, 0),
    (1, 0, 1, 0),
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def prompt_sha(full_prompt: str) -> str:
    """SHA-256 (hex, first 16 chars) of the FINAL assembled prompt.

    Cache keys in adapters/ai_narration.py must hash this -- not the static
    template -- so cached narrations are never reused across different
    memory states (see the integration patch below).
    """
    return hashlib.sha256(
        full_prompt.encode("utf-8", errors="replace")).hexdigest()[:16]


def _sanitize(text: Any, max_chars: int = _MAX_DESC_CHARS) -> str:
    """Truncate and strip control chars from model-sourced text."""
    if not isinstance(text, str):
        return ""
    return text.strip().replace("\n", " ")[:max_chars]


def _clean_choice(val: Any, default: str) -> str:
    """Sanitise an enum-ish field, preserving the default when empty."""
    s = _sanitize(val)
    return s if s else default


def _as_list(val: Any, item_type: type | None = None) -> list:
    """Coerce to list; wrap scalar; optionally filter by type."""
    if val is None:
        return []
    if not isinstance(val, list):
        val = [val]
    if item_type is not None:
        val = [v for v in val if isinstance(v, item_type)]
    return val


def _as_list_of_dicts(val: Any) -> list[dict]:
    """Coerce to list[dict]; wrap a bare dict."""
    if isinstance(val, dict):
        return [val]
    if isinstance(val, list):
        return [v for v in val if isinstance(v, dict)]
    return []


def scrub_fences(text: str) -> str:
    """Strip stray code-fence lines from narration before it reaches TTS."""
    if not isinstance(text, str):
        return ""
    return "\n".join(
        line for line in text.splitlines()
        if not line.strip().startswith("```")
    ).strip()


def _words(text: str) -> set[str]:
    """Lowercased non-stopword tokens for fuzzy thread matching."""
    return {w for w in re.findall(r"[a-z0-9']+", str(text).lower())
            if w not in _STOPWORDS}


def _thread_match(hint: str, summary: str) -> bool:
    """True when `hint` and `summary` share >= 60% of the hint's tokens.

    Requires at least 2 hint words: a one-word hint ("rachel") must never
    close a thread on its own (the v2 substring match did).
    """
    hw, sw = _words(hint), _words(summary)
    if len(hw) < 2 or not sw:
        return False
    return len(hw & sw) / len(hw) >= _THREAD_MATCH_RATIO


def find_entity_key(name: Any, store: dict) -> str | None:
    """Canonical dict key for `name`, matched case-insensitively against
    keys AND all stored aliases. None when unknown."""
    if not isinstance(name, str) or not name.strip():
        return None
    target = name.strip().lower()
    for key, entity in store.items():
        if key.strip().lower() == target:
            return key
        for alias in _as_list((entity or {}).get("aliases"), str):
            if alias.strip().lower() == target:
                return key
    return None


# Back-compat alias for early drafts that imported the private name.
_find_entity_key = find_entity_key


# ---------------------------------------------------------------------------
# Scaffold / load / save
# ---------------------------------------------------------------------------
def _empty_context() -> dict:
    return {
        "series_title": "",
        "chapter": None,
        "characters": {},     # name -> {role, description, first_seen,
        #                      #           last_seen, aliases, status,
        #                      #           associated_locations, relationships}
        "locations": {},      # name -> {description, first_seen, last_seen}
        "story_threads": [],  # [{id, summary, status: active|resolved}]
        "key_events": [],     # [{panel, summary}] capped at _MAX_KEY_EVENTS
        "_meta": {"last_panel": 0, "seed_built": False, "version": _VERSION},
    }


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base recursively. Never mutates either argument."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_context(session_dir: str | Path) -> dict:
    """Load story_context.json, deep-merged over the scaffold.

    Handles partial files, hand-edited files, and older versions without
    crashing. A file from a FUTURE version is backed up to
    story_context.json.bak (first backup wins) and reset -- never
    silently destroyed (R7).
    """
    path = Path(session_dir) / CONTEXT_FILE
    if path.is_file():
        try:
            loaded = json.loads(path.read_text("utf-8"))
            loaded_version = (loaded.get("_meta") or {}).get("version", 0)
            if isinstance(loaded_version, bool) or \
                    not isinstance(loaded_version, int):
                loaded_version = 0
            if loaded_version > _VERSION:
                bak = path.with_suffix(".json.bak")
                if not bak.exists():
                    bak.write_text(path.read_text("utf-8"), "utf-8")
                log.warning("[story_context] file version %d > code version "
                            "%d -- backed up to %s and reset",
                            loaded_version, _VERSION, bak.name)
                return _empty_context()
            ctx = _deep_merge(_empty_context(), loaded)
            ctx["_meta"]["version"] = _VERSION
            return ctx
        except Exception as exc:  # noqa: BLE001 - corrupted file: start fresh
            log.warning("[story_context] load error (%s) -- starting fresh", exc)
    return _empty_context()


def save_context(ctx: dict, session_dir: str | Path) -> None:
    """Atomically write story_context.json."""
    path = Path(session_dir) / CONTEXT_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ctx, indent=2, ensure_ascii=False), "utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# R8 -- Chapter chaining
# ---------------------------------------------------------------------------
def carry_forward(prev_session_dir: str | Path,
                  new_session_dir: str | Path,
                  chapter: int | None = None) -> dict:
    """Copy the previous chapter's context into a new session dir, with the
    per-chapter state reset.

    Kept: character/location roster (+descriptions, aliases), active
    threads (they span chapters).
    Reset: first_seen/last_seen (panel numbers are chapter-local and would
    poison staleness math), key_events (chapter-specific), seed_built
    (the new chapter's text re-seeds the roster), last_panel.
    """
    ctx = load_context(prev_session_dir)
    for store in (ctx["characters"], ctx["locations"]):
        for entry in store.values():
            entry["first_seen"] = 0
            entry["last_seen"] = 0
    ctx["key_events"] = []
    ctx["_meta"]["last_panel"] = 0
    ctx["_meta"]["seed_built"] = False
    ctx["chapter"] = chapter
    Path(new_session_dir).mkdir(parents=True, exist_ok=True)
    save_context(ctx, new_session_dir)
    log.info("[story_context] carried forward: %d characters, %d locations, "
             "%d active threads",
             len(ctx["characters"]), len(ctx["locations"]),
             sum(1 for t in ctx["story_threads"]
                 if t.get("status") == "active"))
    return ctx


# ---------------------------------------------------------------------------
# R2 -- Correct adapter for adapters.ai_models.generate_text_with_fallback
# ---------------------------------------------------------------------------
_SEED_SYSTEM = (
    "You are pre-reading a manhwa/webtoon chapter to build a cast and "
    "location list before any narration is written. This is a one-time "
    "setup call. Return ONLY a JSON object (no markdown, no fences, no "
    'prose) shaped exactly: {"series_title": "", "characters": [{"name": '
    '"", "role": "protagonist|antagonist|ally|minor|unknown", '
    '"description": "", "aliases": []}], "locations": [{"name": "", '
    '"description": ""}], "story_threads": [{"summary": ""}]}'
)


def make_text_model_call(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "",
    request_fn: Callable[..., str] | None = None,
) -> Callable[[str], str]:
    """Build a single-argument ``prompt -> response text`` callable on top
    of adapters.ai_models.generate_text_with_fallback.

    The v2 patch passed one argument to a two-argument function and
    returned the AIFallbackResult object itself; this adapter is the
    corrected call shape (system + user, .result).
    """
    from adapters import ai_models as _ai

    primary = model or _ai.PRIMARY_MODEL

    def call(prompt: str) -> str:
        outcome = _ai.generate_text_with_fallback(
            _SEED_SYSTEM, prompt,
            operation="story-context-seed",
            api_key=api_key, base_url=base_url,
            primary_model=primary, request_fn=request_fn)
        return outcome.result

    return call


# ---------------------------------------------------------------------------
# R3 -- Entity extraction (dict OR raw response string)
# ---------------------------------------------------------------------------
def extract_entities_from_response(response: dict | str) -> dict | None:
    """Extract the optional "entities" dict from a narration response.

    Accepts either the parsed response dict or the RAW model string (the
    outermost JSON object is extracted, tolerating fences/prose), so
    _parse_narration() in ai_narration.py needs no signature change.
    Returns None when absent or malformed -- the narration is still valid.
    """
    parsed: dict | None = None
    if isinstance(response, dict):
        parsed = response
    elif isinstance(response, str):
        first = response.find("{")
        last = response.rfind("}")
        if first != -1 and last > first:
            try:
                obj = json.loads(response[first:last + 1])
                parsed = obj if isinstance(obj, dict) else None
            except ValueError:
                parsed = None
    if not isinstance(parsed, dict):
        return None
    raw = parsed.get("entities")
    return raw if isinstance(raw, dict) else None


# ---------------------------------------------------------------------------
# Thread lifecycle (R6: token-overlap matching, not substring)
# ---------------------------------------------------------------------------
def _resolve_threads(threads: list[dict], hints: Any) -> None:
    """Mark matching active threads resolved, in place."""
    for hint in _as_list(hints, str):
        for t in threads:
            if (t.get("status") == "active"
                    and _thread_match(hint, t.get("summary", ""))):
                t["status"] = "resolved"
                log.info("[story_context] thread resolved: %s", t["summary"])
                break  # one hint closes at most one thread


def _add_thread(threads: list[dict], summary: Any) -> None:
    """Append a new active thread unless it duplicates an active one."""
    summary = _sanitize(summary, 300)
    if not summary:
        return
    for t in threads:
        if t.get("status") == "active" and _thread_match(
                summary, t.get("summary", "")):
            return  # near-duplicate of an active thread
    threads.append({"id": f"t{len(threads) + 1}", "summary": summary,
                    "status": "active"})


# ---------------------------------------------------------------------------
# Context update (R10: per-field failure isolation)
# ---------------------------------------------------------------------------
def _safe(label: str, panel_index: int, fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - one bad field never kills the rest
        log.warning("[story_context] %s error (panel %d): %s",
                    label, panel_index, exc)


def update_context(ctx: dict, entities: dict, panel_index: int) -> None:
    """Merge an entity-update dict into ctx IN PLACE (save externally).

    Every field access is guarded and every failure is isolated per field:
    a corrupt _meta or one malformed entry discards only that field, not
    the whole panel's update (R10).
    """
    if not isinstance(entities, dict):
        return

    chars = ctx.setdefault("characters", {})
    locs = ctx.setdefault("locations", {})
    events = ctx.setdefault("key_events", [])
    threads = ctx.setdefault("story_threads", [])
    meta = ctx.setdefault("_meta", {})

    def _known_seen() -> None:
        for raw_name in _as_list(entities.get("characters_seen"), str):
            key = find_entity_key(raw_name, chars)
            if key:
                chars[key]["last_seen"] = panel_index

    def _new_characters() -> None:
        for nc in _as_list_of_dicts(entities.get("new_characters")):
            name = _sanitize(nc.get("name", ""))
            if not name:
                continue
            key = find_entity_key(name, chars)
            if key is None:
                chars[name] = {
                    "role": _clean_choice(nc.get("role", "unknown"),
                                          "unknown"),
                    "description": _sanitize(nc.get("description", "")),
                    "first_seen": panel_index,
                    "last_seen": panel_index,
                    "aliases": [_sanitize(a) for a in
                                _as_list(nc.get("aliases"), str) if a],
                    "status": "active",
                    "associated_locations": [],
                    "relationships": {},
                }
                log.info("[story_context] new character: %s (%s)",
                         name, nc.get("role", ""))
            else:
                c = chars[key]
                c["last_seen"] = panel_index
                if nc.get("description") and not c.get("description"):
                    c["description"] = _sanitize(nc["description"])
                stored = c.setdefault("aliases", [])
                for alias in _as_list(nc.get("aliases"), str):
                    alias = _sanitize(alias)
                    if alias and alias not in stored:
                        stored.append(alias)

    def _locs_seen() -> None:
        for raw_name in _as_list(entities.get("locations_seen"), str):
            key = find_entity_key(raw_name, locs)
            if key:
                locs[key]["last_seen"] = panel_index

    def _new_locations() -> None:
        for nl in _as_list_of_dicts(entities.get("new_locations")):
            name = _sanitize(nl.get("name", ""))
            if not name:
                continue
            key = find_entity_key(name, locs)
            if key is None:
                locs[name] = {
                    "description": _sanitize(nl.get("description", "")),
                    "first_seen": panel_index,
                    "last_seen": panel_index,
                }
                log.info("[story_context] new location: %s", name)
            else:
                locs[key]["last_seen"] = panel_index

    def _threads() -> None:
        _resolve_threads(threads, entities.get("threads_resolved"))
        for summary in _as_list(entities.get("threads_new"), str):
            _add_thread(threads, summary)

    def _events() -> None:
        nonlocal events  # rebind below; without this the append raises
        for ev in _as_list(entities.get("new_events"), str):
            ev = _sanitize(ev)
            if ev:
                events.append({"panel": panel_index, "summary": ev})
        if len(events) > _MAX_KEY_EVENTS:
            ctx["key_events"] = events[-_MAX_KEY_EVENTS:]
            events = ctx["key_events"]

    def _meta_update() -> None:
        meta["last_panel"] = max(int(meta.get("last_panel", 0) or 0),
                                 panel_index)

    _safe("characters_seen", panel_index, _known_seen)
    _safe("new_characters", panel_index, _new_characters)
    _safe("locations_seen", panel_index, _locs_seen)
    _safe("new_locations", panel_index, _new_locations)
    _safe("threads", panel_index, _threads)
    _safe("key_events", panel_index, _events)
    _safe("meta", panel_index, _meta_update)


# ---------------------------------------------------------------------------
# R1/R4/R5 -- Prompt injection
# ---------------------------------------------------------------------------
def _char_priority(panel_index: int):
    """Sort key: fresh(0) < seeded-unseen(1) < stale(2); newest seen first.

    last_seen == 0 means "seeded by the pre-read, not yet seen on-panel"
    -- that is FRESH information, not stale (the v2 code demoted every
    seeded character to stale, defeating the seed pass).
    """
    def priority(kv):
        ls = kv[1].get("last_seen", 0) or 0
        if ls == 0:
            cls = 1          # seeded by pre-read, not yet seen on-panel
        elif (panel_index - ls) > _STALENESS_PANELS:
            cls = 2          # stale
        else:
            cls = 0          # fresh
        return (cls, -ls)
    return priority


def _char_tag(ls: Any, panel_index: int) -> str:
    if not isinstance(ls, int) or ls <= 0:
        return "(pre-read cast, not yet seen)"
    if (panel_index - ls) > _STALENESS_PANELS:
        return f"[last seen panel {ls}]"
    return f"(panel {ls})"


def inject_into_prompt(ctx: dict, panel_index: int,
                       max_chars: int = 1200) -> str:
    """Return the [STORY MEMORY] block for a panel's narration prompt.

    R1: fits the budget with a size ladder (shrinking item counts) instead
    of dropping whole sections; threads outlast characters; the footer and
    closing delimiter are never cut; the result is never empty while the
    context holds anything. R5: the last few key events are injected as a
    brief digest.
    """
    chars = ctx.get("characters") or {}
    locs = ctx.get("locations") or {}
    threads = [t for t in (ctx.get("story_threads") or [])
               if t.get("status") == "active"]
    events = ctx.get("key_events") or []

    if not chars and not locs and not threads:
        return ""

    header = f"[STORY MEMORY -- panel {panel_index}]"
    footer = (
        "If you recognise any of these characters or locations, refer to "
        "them by their established name and maintain continuity (do not "
        "introduce them as new).\n[/STORY MEMORY]"
    )

    ordered_chars = sorted(chars.items(), key=_char_priority(panel_index))
    ordered_locs = sorted(
        locs.items(), key=lambda kv: kv[1].get("last_seen", 0) or 0,
        reverse=True)

    def thread_id_num(t: dict) -> int:
        try:
            return int(str(t.get("id", "t0"))[1:])
        except ValueError:
            return 0

    ordered_threads = sorted(threads, key=thread_id_num, reverse=True)
    recent_events = list(reversed(events))  # newest first

    def char_lines(n: int) -> list[str]:
        out = []
        for name, c in ordered_chars[:n]:
            aliases = c.get("aliases") or []
            alias_s = f" (also: {', '.join(aliases[:3])})" if aliases else ""
            desc = (c.get("description") or "")[:_MAX_PROMPT_DESC_CHARS]
            tag = _char_tag(c.get("last_seen", 0), panel_index)
            rels = "; ".join(f"{k}: {v}" for k, v in
                             list((c.get("relationships") or {}).items())[:2])
            line = f"  - {name}{alias_s} [{c.get('role', '?')}]{tag}: {desc}."
            if rels:
                line += f" ({rels})"
            out.append(line)
        return out

    def loc_lines(n: int) -> list[str]:
        return [f"  - {name} (panel {loc.get('last_seen', 0) or 0}): "
                f"{(loc.get('description') or '')[:_MAX_PROMPT_DESC_CHARS]}"
                for name, loc in ordered_locs[:n]]

    def thread_lines(n: int) -> list[str]:
        return [f"  - {t.get('summary', '')}" for t in ordered_threads[:n]]

    def event_lines(n: int) -> list[str]:
        return [f"  - panel {ev.get('panel', '?')}: "
                f"{(ev.get('summary') or '')[:_MAX_PROMPT_DESC_CHARS]}"
                for ev in recent_events[:n]]

    def build(nc: int, nl: int, nt: int, ne: int) -> str:
        sections = []
        cl = char_lines(nc)
        if cl:
            sections.append("Characters:\n" + "\n".join(cl))
        ll = loc_lines(nl)
        if ll:
            sections.append("Locations:\n" + "\n".join(ll))
        tl = thread_lines(nt)
        if tl:
            sections.append("Ongoing threads:\n" + "\n".join(tl))
        el = event_lines(ne)
        if el:
            sections.append("Recent events:\n" + "\n".join(el))
        return f"{header}\n" + "\n".join(sections) + f"\n{footer}"

    # Size ladder: shrink item counts until the block fits.
    for nc, nl, nt, ne in _LADDER:
        block = build(nc, nl, nt, ne)
        if len(block) <= max_chars:
            return block

    # Ladder exhausted: keep the cheapest continuity signal (threads) and
    # at most one character/location line, whatever fits. Never drop the
    # footer, never return "" while the context is non-empty.
    for nc, nl, nt, ne in ((1, 0, 5, 0), (1, 0, 2, 0), (1, 0, 1, 0),
                           (0, 0, 1, 0)):
        block = build(nc, nl, nt, ne)
        if len(block) <= max_chars:
            return block
    return build(0, 0, 1, 0)  # one thread line + footer; may exceed a tiny cap


# ---------------------------------------------------------------------------
# Seed pass
# ---------------------------------------------------------------------------
def build_seed_context(
    panels: list[dict],
    session_dir: str | Path,
    model_call: Callable[[str], str],
    series_title: str = "",
    chapter: int | None = None,
    force_rebuild: bool = False,
) -> dict:
    """Run ONCE before narration to seed the context from panel text.

    model_call: a callable taking one prompt string and returning the
    model's response text -- use make_text_model_call() for the project's
    adapters, or any (prompt -> str) callable in tests.

    seed_built=True is set ONLY on success: a transient model failure
    leaves seed_built=False so the next pipeline run retries (H1).
    context_only panels are included -- their dialogue is exactly the
    story context panel_filter kept them for.
    """
    ctx = load_context(session_dir)
    if ctx["_meta"].get("seed_built") and not force_rebuild:
        log.info("[story_context] seed already built -- skipping")
        return ctx

    ctx["series_title"] = ctx.get("series_title") or series_title
    ctx["chapter"] = chapter

    text_lines: list[str] = []
    char_total = 0
    for p in panels:
        idx = p.get("panel_index", "?")
        parts = [p.get("dialogue", ""), p.get("narration", ""),
                 p.get("text", "")]
        combined = " | ".join(t for t in parts
                              if isinstance(t, str) and t.strip())
        if not combined:
            continue
        line = f"Panel {idx}: {combined}"
        if char_total + len(line) > _MAX_TEXT_BLOCK_CHARS:
            log.warning("[story_context] text_block cap (%d chars) reached "
                        "before panel %s; remaining panels skipped for seed",
                        _MAX_TEXT_BLOCK_CHARS, idx)
            break
        text_lines.append(line)
        char_total += len(line)

    if not text_lines:
        log.info("[story_context] no text in panels -- seed empty context")
        ctx["_meta"]["seed_built"] = True  # nothing to retry
        save_context(ctx, session_dir)
        return ctx

    prompt = ("Panel text in order:\n\n" + "\n".join(text_lines)
              + "\n\nReturn the JSON object now.")
    log.info("[story_context] running seed extraction over %d text panels "
             "(%d chars)...", len(text_lines), char_total)
    try:
        raw = model_call(prompt)
        first = raw.find("{")
        last = raw.rfind("}")
        if first == -1 or last <= first:
            raise ValueError("no JSON object found in model response")
        seed = json.loads(raw[first:last + 1])
    except Exception as exc:  # noqa: BLE001 - H1: leave seed_built=False
        log.warning("[story_context] seed extraction failed (%s) -- will "
                    "retry next run", exc)
        save_context(ctx, session_dir)
        return ctx

    ctx["series_title"] = (ctx["series_title"]
                           or _sanitize(seed.get("series_title", ""), 100))

    def _seed_characters() -> None:
        for c in _as_list_of_dicts(seed.get("characters")):
            name = _sanitize(c.get("name", ""))
            if name and find_entity_key(name, ctx["characters"]) is None:
                ctx["characters"][name] = {
                    "role": _clean_choice(c.get("role", "unknown"), "unknown"),
                    "description": _sanitize(c.get("description", "")),
                    "first_seen": 0,
                    "last_seen": 0,
                    "aliases": [_sanitize(a) for a in
                                _as_list(c.get("aliases"), str) if a],
                    "status": "active",
                    "associated_locations": [],
                    "relationships": {},
                }

    def _seed_locations() -> None:
        for loc in _as_list_of_dicts(seed.get("locations")):
            name = _sanitize(loc.get("name", ""))
            if name and find_entity_key(name, ctx["locations"]) is None:
                ctx["locations"][name] = {
                    "description": _sanitize(loc.get("description", "")),
                    "first_seen": 0,
                    "last_seen": 0,
                }

    def _seed_threads() -> None:
        for t in _as_list_of_dicts(seed.get("story_threads")):
            _add_thread(ctx["story_threads"], t.get("summary", ""))

    _safe("seed characters", 0, _seed_characters)
    _safe("seed locations", 0, _seed_locations)
    _safe("seed threads", 0, _seed_threads)

    ctx["_meta"]["seed_built"] = True  # only on success
    save_context(ctx, session_dir)
    log.info("[story_context] seed complete: %d characters, %d locations, "
             "%d active threads",
             len(ctx["characters"]), len(ctx["locations"]),
             sum(1 for t in ctx["story_threads"]
                 if t.get("status") == "active"))
    return ctx


# ---------------------------------------------------------------------------
# INTEGRATION patch for adapters/ai_narration.py (signature-checked)
# ---------------------------------------------------------------------------
# 1. Imports at the top of adapters/ai_narration.py:
#
#       from story_context import (
#           build_seed_context, inject_into_prompt, update_context,
#           extract_entities_from_response, scrub_fences, prompt_sha,
#           make_text_model_call, ENTITIES_SCHEMA_FRAGMENT, ENTITIES_RULE,
#       )
#
# 2. Extend the JSON schema line of PANEL_NARRATION_PROMPT so the model
#    returns entities in the SAME strict-JSON object (no fences, no
#    second block):
#
#       old: {"narration": "...", "dialogue": "..."}
#       new: {"narration": "...", "dialogue": "...", <ENTITIES_SCHEMA_FRAGMENT>}
#       ...and append ENTITIES_RULE to the Rules list.
#
# 3. Before the panel loop (seed; note the correct two-arg call shape --
#    the v2 patch's one-arg lambda was wrong):
#
#       panels_for_seed = [p.model_dump() for p in artifact.panels]
#       ctx = build_seed_context(
#           panels_for_seed, session,
#           model_call=make_text_model_call(api_key=key, base_url=base_url,
#                                           model=model),
#       )
#
# 4. Per panel, BEFORE the cache lookup -- assemble the FINAL prompt with
#    memory and hash THAT for the cache key (replaces the static
#    prompt_sha computed once at module scope):
#
#       memory = inject_into_prompt(ctx, panel.panel_index)
#       final_prompt = PANEL_NARRATION_PROMPT
#       if memory:
#           final_prompt = f"{PANEL_NARRATION_PROMPT}\n\n{memory}"
#       cache_key = (f"{_image_sha(data)[:16]}_{prompt_sha(final_prompt)}_"
#                    f"{effective_primary}_{_ai.FALLBACK_MODEL}")
#
#    then pass final_prompt to generate_vision_with_fallback in place of
#    PANEL_NARRATION_PROMPT (it becomes the system message; the memory
#    block therefore rides along with the image).
#
# 5. After the response:
#
#       narration, dialogue = _parse_narration(outcome.result)
#       narration = scrub_fences(narration)          # TTS safety net
#       entities = extract_entities_from_response(outcome.result)  # or the dict
#       if entities:
#           update_context(ctx, entities, panel.panel_index)
#           save_context(ctx, session)
#
#    context_only panels: they get NO model call (panel_filter contract),
#    so they contribute only via the seed pass in step 3. Do not add any
#    extra skip logic -- recap_video.py / cinematic_effects.py already
#    skip them downstream.
#
# 6. Chapter chaining (webapp continuation): before the first build of a
#    new chapter's session, copy the previous chapter's memory forward:
#
#       from story_context import carry_forward
#       carry_forward(prev_session_dir, new_session_dir, chapter=job.chapter)
#


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import tempfile
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with tempfile.TemporaryDirectory() as td:
        ctx = _empty_context()
        ctx["characters"]["Bam"] = {
            "role": "protagonist", "description": "black hair, determined",
            "first_seen": 1, "last_seen": 3,
            "aliases": ["Twenty-Fifth Bam"], "status": "active",
            "associated_locations": [], "relationships": {"Rachel": "motivation"},
        }
        ctx["locations"]["Tower Entrance"] = {
            "description": "massive stone gate", "first_seen": 1,
            "last_seen": 2,
        }
        _add_thread(ctx["story_threads"],
                    "Bam is climbing the Tower to find Rachel")

        print("--- inject (panel 5) ---")
        print(inject_into_prompt(ctx, panel_index=5))

        assert find_entity_key("twenty-fifth bam", ctx["characters"]) == "Bam"
        assert find_entity_key("BAM", ctx["characters"]) == "Bam"
        print("Alias match: PASS")

        update_context(ctx, {
            "characters_seen": ["Twenty-Fifth Bam"],
            "new_characters": [{"name": "Headon", "role": "antagonist",
                                "description": "white fluffy Tower guardian",
                                "aliases": []}],
            "locations_seen": ["Tower Entrance"],
            "new_events": ["Headon sets Bam's first test"],
        }, panel_index=5)
        save_context(ctx, td)
        reloaded = load_context(td)
        assert "Headon" in reloaded["characters"]
        assert reloaded["characters"]["Bam"]["last_seen"] == 5
        assert len(reloaded["key_events"]) == 1

        update_context(ctx, {"threads_resolved": ["find rachel"]},
                       panel_index=6)
        assert [t for t in ctx["story_threads"]
                if t["status"] == "resolved"], "thread should resolve"
        update_context(ctx, {"characters_seen": "Bam"}, panel_index=7)
        print("All smoke tests PASSED")
