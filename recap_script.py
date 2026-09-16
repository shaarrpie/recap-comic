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
             "quote": null}           # verbatim dialogue, voiced separately
          ],
          "text": "<full script, one paragraph>",
          "structure": ["hook", "setup", "escalation", "cliffhanger"]
        }

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

log = logging.getLogger(__name__)

SCRIPT_FILE = "script.json"
SCRIPT_VERSION = 1
GENERATOR = "recap_script.1"

# Structure template for the model + for validating its mapping.
STRUCTURE = ["hook", "setup", "escalation", "cliffhanger"]

# Non-lexical "narration" strings the vision model emits for blank/silent
# panels. They must never reach the script or TTS.
_NON_LEXICAL = re.compile(r"^[\s.·•—–\-_*~…!?]*$")

SYSTEM_PROMPT = (
    "You are the scriptwriter for a manhwa/webtoon recap video. You write "
    "the narrator's voice-over for ONE chapter, given an ordered list of "
    "panels with their visual descriptions and dialogue. Respond with ONE "
    "JSON object ONLY. No prose, no markdown, no code fences."
)

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
- Use character NAMES from the dialogue/memory, not "a person" or "a man".
- Narrate STORY: what happens, why it matters, what changes. Never
  inventory what is visible ("a gray background with no characters").
- 1-2 short sentences per panel; skip panels that add nothing (merge them
  into neighbours). Fewer, better lines beat one line per panel.
- Plain, punchy YouTube-recap English. Present tense. No flowery prose.
- Do not invent events that contradict the panel list; inference of
  causality between shown events is allowed and expected.
- If the memory block names characters/threads, use those names and keep
  them consistent.

{memory_block}
Panel list (in order):
{panel_list}

Return STRICT JSON, exactly:
{{"lines": [{{"panel_index": <int from the list>, "text": "<narrator line>",
              "part": "hook|setup|escalation|cliffhanger",
              "quote": "<one verbatim dialogue line to voice as a "
                       "character, or null>"}}]}}
Every panel_index MUST come from the list. Lines MUST be listed in
non-decreasing panel_index order (the narrator speaks in panel order;
hook/setup/escalation/cliffhanger are TONES, not a reordering of the
video). 3-12 lines total, never one per panel.
"""

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
        visual = (p.narration or "").strip()
        if is_non_lexical(visual):
            visual = "(no usable description)"
        visual = visual[:MAX_PANEL_CHARS]
        dialogue = (p.dialogue or "").strip()
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


def _parse_response(raw: str, panels: list[dict]) -> dict | None:
    """Parse + validate the model's {"lines": [...]} against the panel list."""
    first, last = raw.find("{"), raw.rfind("}")
    if first == -1 or last <= first:
        return None
    try:
        obj = json.loads(raw[first:last + 1])
    except ValueError:
        return None
    lines = obj.get("lines")
    if not isinstance(lines, list) or not lines:
        return None
    idx_by_index = {p["panel_index"]: p for p in panels}
    parsed: list[dict] = []
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        idx = ln.get("panel_index")
        text = ln.get("text")
        if not isinstance(idx, int) or idx not in idx_by_index:
            continue
        if not isinstance(text, str) or is_non_lexical(text):
            continue
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
        })
    # The narrator speaks in panel order (one TTS clip per panel). A model
    # that lists lines out of order is corrected by sorting, not rejected.
    parsed.sort(key=lambda ln: ln["panel_index"])
    return {"lines": parsed} if len(parsed) >= 2 else None


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
    can_call = model_call is not None or (
        api_key is not None or request_fn is not None)
    if not has_text or not can_call:
        # offline / fallback path: keep the old joined captions
        text = _fallback_join(artifact, style)
        log.info("[script] fallback join (%d chars, has_text=%s, can_call=%s)",
                 len(text), has_text, can_call)
        return {"status": "fallback", "used_fallback": True,
                "lines": [], "text": text, "structure": [],
                "model_used": "", "input_hash": input_hash}

    panel_lines = [
        f"{p['panel_index']}. {p['visual']}"
        + (f'  [dialogue: {p["dialogue"]}]' if p["dialogue"] else "")
        for p in panels]

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
                request_fn=request_fn)
            raw = outcome.result
            model_used, fb = outcome.model_used, outcome.fallback_used
    except Exception as exc:  # noqa: BLE001 - fall back, never kill the run
        log.warning("[script] model call failed (%s); using joined captions",
                    exc)
        text = _fallback_join(artifact, style)
        return {"status": "fallback", "used_fallback": True,
                "lines": [], "text": text, "structure": [],
                "model_used": "", "input_hash": input_hash}

    parsed = _parse_response(raw, panels)
    if parsed is None:
        log.warning("[script] unparseable/invalid model response; using "
                    "joined captions")
        text = _fallback_join(artifact, style)
        return {"status": "fallback", "used_fallback": True,
                "lines": [], "text": text, "structure": [],
                "model_used": "", "input_hash": input_hash}

    script_text = " ".join(ln["text"] for ln in parsed["lines"])
    result = {
        "version": SCRIPT_VERSION,
        "style": style,
        "generator": GENERATOR,
        "model_used": model_used or "custom",
        "fallback_used": bool(fb),
        "input_hash": input_hash,
        "lines": parsed["lines"],
        "text": script_text,
        "structure": STRUCTURE,
    }
    path = session / SCRIPT_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                   "utf-8")
    tmp.replace(path)
    log.info("[script] built %d lines, %d chars, parts=%s",
             len(parsed["lines"]), len(script_text),
             [ln["part"] for ln in parsed["lines"]])
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
