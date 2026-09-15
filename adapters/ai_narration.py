# adapters/ai_narration.py
"""AI narration for ALREADY-CROPPED panels (second step, user-triggered).

Workflow this serves:
    1. Deterministic CV cut first (backend "deterministic"/"none": uniform
       blank-color rows -> gutters -> panel PNGs). NO AI involved.
    2. The user presses "Generate narration (AI)" -> THIS module sends each
       cropped panel PNG to Qwen3.5-397B-A17B (Mistral Medium 3.5 fallback)
       for narration + dialogue extraction ONLY.

Hard rule: geometry is NEVER touched here. The prompt asks for no
coordinates, the response carries none, and every panel's y_start / y_end /
image_file is snapshotted before and asserted after. Cropping stays 100%
deterministic; AI only writes words.

Narrative memory (story_context.py, wired per its INTEGRATION section):
    * ONE seed call before the panel loop builds the character/location
      roster and open threads from all panel text (dialogue + narration,
      including context_only panels).
    * Each panel's prompt is augmented with a [STORY MEMORY] block, so
      panels 3 and 5 know they show the SAME character.
    * The cache key hashes the FINAL prompt (template + memory), so cached
      captions are never reused across different memory states.
    * Optional "entities" key in each response updates the memory after
      every panel.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from story_context import (
    ENTITIES_RULE,
    ENTITIES_SCHEMA_FRAGMENT,
    build_seed_context,
    extract_entities_from_response,
    inject_into_prompt,
    make_text_model_call,
    prompt_sha,
    scrub_fences,
    update_context,
)

log = logging.getLogger(__name__)

PANEL_NARRATION_PROMPT = """\
You are narrating ONE cropped comic/manhwa panel for a recap video.
Respond with ONE JSON object ONLY. No prose, no markdown, no code fences.
Start with { and end with }.

Return STRICT JSON matching this exact shape:
{"narration": "...", "dialogue": "...", """ + ENTITIES_SCHEMA_FRAGMENT + """}

Rules:
- narration: ONE short simple sentence describing ONLY what is visible in
  THIS panel image (characters, action, setting, mood). Never invent events
  that are not shown. Plain language, no flowery description. Empty string
  only if the panel is blank/indecipherable.
- dialogue: speech-bubble and SFX text in the panel, verbatim, in reading
  order, separated by " / ". Empty string if there is none.
- Do NOT output coordinates, panel ids, or anything else.
- """ + ENTITIES_RULE + """
"""


def _extract_json_obj(text: str) -> dict:
    """Pull one JSON object out of a response that may add fences/prose."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[A-Za-z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    obj = json.loads(t[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model output is not a JSON object")
    return obj


def _parse_narration(text: str) -> tuple[str, str]:
    obj = _extract_json_obj(text)
    narration = obj.get("narration", "")
    dialogue = obj.get("dialogue", "")
    if not isinstance(narration, str) or not isinstance(dialogue, str):
        raise ValueError("narration/dialogue must be strings")
    return narration.strip(), dialogue.strip()


def _image_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def narrate_cropped_panels(
    session_dir: str | Path,
    *,
    panels_file: str = "panels.json",
    api_key: str | None = None,
    model: str = "",
    base_url: str | None = None,
    force: bool = False,
    gap_s: float = 1.0,
    cache_dir: str | Path | None = None,
    request_fn: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Fill narration/dialogue for cropped panels via Qwen -> Mistral.

    Reads panels.json + each panel_*.png in `session_dir`, writes words
    back (geometry byte-identical). Per-panel results are cached keyed by
    image SHA + prompt + model pair, so re-runs skip finished panels.
    Panels whose AI call fails on BOTH models keep their old text and are
    reported in summary["failed"]; if every panel fails, raises
    RuntimeError (never fabricated text).
    """
    from adapters import ai_models as _ai
    from guided_cutter import CutArtifact

    session = Path(session_dir)
    panels_path = session / panels_file
    if not panels_path.is_file():
        raise FileNotFoundError(f"panels file not found: {panels_path}")
    artifact = CutArtifact.model_validate_json(panels_path.read_text("utf-8"))
    if not artifact.panels:
        raise ValueError("panels.json contains no panels")

    # Geometry lock: snapshot everything the AI must not change.
    geometry = {p.id: (p.y_start, p.y_end, p.image_file) for p in artifact.panels}

    cache_root = Path(cache_dir) if cache_dir is not None else (
        Path(__file__).resolve().parent.parent / ".cache" / "ai-narration")
    cache_root.mkdir(parents=True, exist_ok=True)
    key = api_key or _ai.api_key_from_env()
    if not key and request_fn is None:
        raise RuntimeError(
            "XKIRO_API_KEY is not set; set it in .env or pass api_key")

    # --- Story memory: ONE seed pass over all panel text before the loop.
    # Includes context_only panels (their dialogue is exactly the story
    # context panel_filter kept them for). Failure is non-fatal: memory
    # stays empty and narration proceeds without continuity hints.
    ctx: dict = {}
    try:
        panels_for_seed = [p.model_dump() for p in artifact.panels]
        ctx = build_seed_context(
            panels_for_seed, session,
            model_call=make_text_model_call(
                api_key=key, base_url=base_url, model=model,
                request_fn=request_fn),
            series_title=artifact.source,
            force_rebuild=False)
    except Exception as exc:  # noqa: BLE001 - memory is an enhancement
        log.warning("[AI] story-context seed skipped (%s); narrating "
                    "without memory", exc)

    summary: dict[str, Any] = {
        "panels": len(artifact.panels), "narrated": 0, "cached": 0,
        "failed": [], "models_used": {}, "story_memory": bool(ctx),
    }
    sidecar: dict[str, Any] = {}
    failures: list[str] = []

    for i, panel in enumerate(artifact.panels):
        img_path = session / panel.image_file
        if not img_path.is_file():
            log.warning("[AI] panel %s image missing (%s); keeping old text",
                        panel.id, panel.image_file)
            failures.append(f"{panel.id}: image file missing")
            continue
        data = img_path.read_bytes()
        # Cache key MUST include the model pair actually in use, or a
        # custom model silently reuses cached text from the default pair.
        # It hashes the FINAL prompt (template + [STORY MEMORY] block), so
        # a cached caption is never reused across different memory states.
        effective_primary = model or _ai.PRIMARY_MODEL
        memory_block = (inject_into_prompt(ctx, panel.panel_index)
                        if ctx else "")
        final_prompt = PANEL_NARRATION_PROMPT
        if memory_block:
            final_prompt = f"{PANEL_NARRATION_PROMPT}\n\n{memory_block}"
        cache_key = (f"{_image_sha(data)[:16]}_{prompt_sha(final_prompt)}_"
                     f"{effective_primary}_{_ai.FALLBACK_MODEL}")
        # sanitize for Windows filenames
        cache_key = re.sub(r"[^A-Za-z0-9_.-]", "_", cache_key)
        cache_path = cache_root / f"{panel.id}_{cache_key}.json"

        cached: dict | None = None
        if not force and cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text("utf-8"))
            except (OSError, ValueError):
                cached = None
        if cached is not None:
            narration, dialogue = cached.get("narration", ""), cached.get("dialogue", "")
            summary["cached"] += 1
            used, fb = cached.get("model_used", "?"), bool(cached.get("fallback_used"))
            log.info("[AI] panel %s narration from cache (model=%s)",
                     panel.id, used)
        else:
            b64 = base64.b64encode(data).decode("ascii")
            try:
                outcome = _ai.generate_vision_with_fallback(
                    final_prompt, b64,
                    operation=f"ai-narrate:{panel.id}",
                    api_key=key, base_url=base_url,
                    primary_model=model or _ai.PRIMARY_MODEL,
                    request_fn=request_fn)
                narration, dialogue = _parse_narration(outcome.result)
                narration = scrub_fences(narration)      # TTS safety net
                used, fb = outcome.model_used, outcome.fallback_used
                # Merge the optional entities update into memory (best
                # effort; a malformed block never kills the narration).
                if ctx:
                    entities = extract_entities_from_response(outcome.result)
                    if entities:
                        try:
                            update_context(ctx, entities, panel.panel_index)
                            from story_context import save_context
                            save_context(ctx, session)
                        except Exception as exc:  # noqa: BLE001
                            log.warning("[AI] story-context update failed "
                                        "(panel %s): %s", panel.id, exc)
                cache_path.write_text(json.dumps({
                    "narration": narration, "dialogue": dialogue,
                    "model_used": used, "fallback_used": fb,
                }, indent=2), encoding="utf-8")
                summary["narrated"] += 1
            except Exception as exc:  # noqa: BLE001 - per-panel; keep old text
                log.warning("[AI] panel %s narration failed (%s); "
                            "keeping old text", panel.id, exc)
                failures.append(f"{panel.id}: {exc}")
                continue
            if gap_s > 0 and i < len(artifact.panels) - 1:
                time.sleep(gap_s)
        # Only overwrite with non-empty AI text; never blank out existing words.
        if narration:
            panel.narration = narration
        if dialogue:
            panel.dialogue = dialogue
        sidecar[panel.id] = {"model_used": used, "fallback_used": fb,
                             "cached": cached is not None}
        summary["models_used"][used] = summary["models_used"].get(used, 0) + 1

    # Geometry lock: prove nothing moved.
    for p in artifact.panels:
        if (p.y_start, p.y_end, p.image_file) != geometry[p.id]:
            raise RuntimeError(
                f"internal: panel {p.id} geometry changed during AI "
                "narration (this must never happen)")

    if summary["narrated"] + summary["cached"] == 0:
        raise RuntimeError(
            "AI narration failed for every panel: " + "; ".join(failures))
    summary["failed"] = failures

    tmp = panels_path.with_suffix(".json.tmp")
    tmp.write_text(artifact.model_dump_json(indent=2) + "\n", "utf-8")
    tmp.replace(panels_path)
    (session / "ai_narration.json").write_text(
        json.dumps(sidecar, indent=2) + "\n", "utf-8")
    # Phase 2.5: whole-chapter script pass over the fresh captions (falls
    # back to the plain join when no key/model is available). Per-panel
    # captions stay the ground truth in panels.json; script.json holds the
    # narrated recap lines the video stage prefers.
    script_summary: dict[str, Any] = {}
    try:
        from recap_script import build_chapter_script
        script_summary = build_chapter_script(
            session, api_key=key, base_url=base_url, model=model,
            request_fn=request_fn,
            model_call=make_text_model_call(
                api_key=key, base_url=base_url, model=model,
                request_fn=request_fn) if (key or request_fn) else None,
            force=force) or {}
        summary["script_pass"] = script_summary
        script_text_val = script_summary.get("text")
        if isinstance(script_text_val, str) and script_text_val.strip():
            (session / "narration.txt").write_text(
                script_text_val, "utf-8")
        elif ctx or script_summary.get("used_fallback"):
            # memory exists but the script pass chose not to run/failed:
            # keep narration.txt as the joined-caption fallback
            from narrator import make_script_from_cut
            (session / "narration.txt").write_text(
                make_script_from_cut(artifact,
                                     style=_narration_style(session)),
                "utf-8")
    except Exception as exc:  # noqa: BLE001 - script is a convenience copy
        log.warning("[AI] chapter script pass skipped: %s", exc)
        try:
            from narrator import make_script_from_cut
            (session / "narration.txt").write_text(
                make_script_from_cut(artifact,
                                     style=_narration_style(session)),
                "utf-8")
        except Exception:  # noqa: BLE001
            pass
    log.info("[AI] narration complete panels=%d narrated=%d cached=%d "
             "failed=%d memory=%s",
             summary["panels"], summary["narrated"], summary["cached"],
             len(failures), bool(ctx))
    return summary


def _narration_style(session: Path) -> str:
    try:
        edit = json.loads((session / "narration_edit.json").read_text("utf-8"))
        if edit.get("style") in ("recap", "literal"):
            return edit["style"]
    except (OSError, ValueError):
        pass
    return "recap"
