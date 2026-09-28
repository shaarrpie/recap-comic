# adapters/ai_narration.py
"""AI narration for ALREADY-CROPPED panels (second step, user-triggered).

Workflow this serves:
    1. Deterministic CV cut first (backend "deterministic"/"none": uniform
       blank-color rows -> gutters -> panel PNGs). NO AI involved.
    2. The user presses "Generate narration (AI)" -> THIS module sends each
       cropped panel PNG to Agnes (2.5 Flash primary, 2.0 Flash fallback)
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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from story_context import (
    ENTITIES_RULE,
    ENTITIES_SCHEMA_FRAGMENT,
    build_seed_context,
    build_seed_context_from_images,
    inject_into_prompt,
    make_text_model_call,
    make_vision_seed_call,
    prompt_sha,
    scrub_fences,
)
from text_clean import clean_text

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


def _build_contact_sheet(session: Path, panels, *, max_panels: int = 24,
                         cell_w: int = 320, cell_h: int = 460) -> bytes | None:
    """Grid of downscaled panel thumbnails (PNG bytes) for ONE cast-survey
    vision call. Bounded to `max_panels`; returns None when PIL or the images
    are unavailable so the caller can fall back to no memory (never fatal)."""
    try:
        from io import BytesIO

        from PIL import Image
    except ImportError:
        return None
    thumbs = []
    for p in list(panels)[:max_panels]:
        path = session / p.image_file
        if not path.is_file():
            continue
        try:
            im = Image.open(path)
            im.load()
            im.thumbnail((cell_w, cell_h))
            if im.mode != "RGB":
                im = im.convert("RGB")
            thumbs.append(im)
        except Exception as exc:  # noqa: BLE001 - one bad image can't kill it
            log.warning("[AI] contact-sheet skipped %s: %s", p.image_file, exc)
    if not thumbs:
        return None
    cols = min(4, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    gap = 8
    sheet = Image.new(
        "RGB", (cols * cell_w + (cols + 1) * gap,
                rows * cell_h + (rows + 1) * gap), (255, 255, 255))
    for i, im in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = gap + c * (cell_w + gap) + (cell_w - im.width) // 2
        y = gap + r * (cell_h + gap) + (cell_h - im.height) // 2
        sheet.paste(im, (x, y))
    buf = BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()


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
    seed_cast_from_images: bool = True,
    concurrency: int = 16,
) -> dict[str, Any]:
    """Fill narration/dialogue for cropped panels via Agnes (primary ->
    fallback).

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
    # `key` is only an existence check (api_key_from_env returns the first).
    # Every downstream call gets the ORIGINAL api_key (None for the CLI), so
    # ai_models.api_key_pool(None) expands to the full AGNES_API_KEY +
    # AGNES_API_KEYS pool and rotates across keys on 429s. Passing a resolved
    # single key here would collapse the pool to one key.
    key = api_key or _ai.api_key_from_env()
    if not key and request_fn is None:
        raise RuntimeError(
            "AGNES_API_KEY is not set; set it in .env or pass api_key")

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
                api_key=api_key, base_url=base_url, model=model,
                request_fn=request_fn),
            series_title=artifact.source,
            force_rebuild=False)
    except Exception as exc:  # noqa: BLE001 - memory is an enhancement
        log.warning("[AI] story-context seed skipped (%s); narrating "
                    "without memory", exc)

    # Fresh-cut fallback: when the text seed found no cast (empty dialogue on a
    # first pass), run ONE vision cast-survey over a contact-sheet so even the
    # opening panels get named continuity. Skipped when a roster already exists
    # or no vision backend is available; failure is never fatal.
    if seed_cast_from_images and (key or request_fn) and \
            not (ctx.get("characters") or ctx.get("locations")):
        try:
            sheet = _build_contact_sheet(session, artifact.panels)
            if sheet is not None:
                ctx = build_seed_context_from_images(
                    base64.b64encode(sheet).decode("ascii"), session,
                    vision_call=make_vision_seed_call(
                        api_key=api_key, base_url=base_url, model=model,
                        request_fn=request_fn),
                    series_title=artifact.source, force_rebuild=force)
        except Exception as exc:  # noqa: BLE001 - image seed is an enhancement
            log.warning("[AI] image cast seed skipped (%s)", exc)

    summary: dict[str, Any] = {
        "panels": len(artifact.panels), "narrated": 0, "cached": 0,
        "failed": [], "models_used": {}, "story_memory": bool(ctx),
    }
    sidecar: dict[str, Any] = {}
    failures: list[str] = []

    # Per-panel vision is INDEPENDENT (no cross-panel memory mutation), so the
    # whole chapter's ~15s/panel network latency OVERLAPS in a thread pool
    # instead of serialising. Story-memory continuity is restored by the single
    # whole-chapter script pass below (build_chapter_script), not by threading
    # each panel. The seed ctx is frozen for the loop, so the cache key is
    # stable and re-runs hit cache (the old evolving-memory key caused misses).
    def _narrate_one(panel):
        img_path = session / panel.image_file
        if not img_path.is_file():
            return panel, None, "image file missing"
        data = img_path.read_bytes()
        effective_primary = model or _ai.PRIMARY_MODEL
        memory_block = (inject_into_prompt(ctx, panel.panel_index)
                        if ctx else "")
        final_prompt = PANEL_NARRATION_PROMPT
        if memory_block:
            final_prompt = f"{PANEL_NARRATION_PROMPT}\n\n{memory_block}"
        cache_key = (f"{_image_sha(data)[:16]}_{prompt_sha(final_prompt)}_"
                     f"{effective_primary}_{_ai.FALLBACK_MODEL}")
        cache_key = re.sub(r"[^A-Za-z0-9_.-]", "_", cache_key)
        cache_path = cache_root / f"{panel.id}_{cache_key}.json"
        if not force and cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text("utf-8"))
            except (OSError, ValueError):
                cached = None
            if cached is not None:
                cached["_fresh"] = False
                return panel, cached, None
        try:
            b64 = base64.b64encode(data).decode("ascii")
            outcome = _ai.generate_vision_with_fallback(
                final_prompt, b64, operation=f"ai-narrate:{panel.id}",
                api_key=api_key, base_url=base_url,
                primary_model=model or _ai.PRIMARY_MODEL,
                # reasoning models spend part of the budget on hidden
                # reasoning_content; 6000 avoids empty-content "failures".
                max_tokens=6000, request_fn=request_fn)
            narration, dialogue = _parse_narration(outcome.result)
            narration = scrub_fences(narration)          # TTS safety net
            narration, dialogue, dropped = clean_text(narration, dialogue)
            if dropped:
                log.info("[AI] panel %s dropped %d non-English bubble "
                         "segment(s) from TTS text", panel.id, dropped)
            res = {"narration": narration, "dialogue": dialogue,
                   "model_used": outcome.model_used,
                   "fallback_used": bool(outcome.fallback_used),
                   "_fresh": True}
            cache_path.write_text(json.dumps(
                {k: res[k] for k in ("narration", "dialogue",
                                     "model_used", "fallback_used")},
                indent=2), encoding="utf-8")
            return panel, res, None
        except Exception as exc:  # noqa: BLE001 - per-panel; keep old text
            return panel, None, str(exc)

    workers = max(1, int(concurrency))
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="ainar") as pool:
        results = list(pool.map(_narrate_one, artifact.panels))
    for panel, res, err in results:
        if err is not None:
            log.warning("[AI] panel %s narration failed (%s); keeping old text",
                        panel.id, err)
            failures.append(f"{panel.id}: {err}")
            continue
        if res is None:
            continue
        narration, dialogue = res.get("narration", ""), res.get("dialogue", "")
        used = res.get("model_used", "?")
        fb = bool(res.get("fallback_used"))
        fresh = bool(res.get("_fresh"))
        # Only overwrite with non-empty AI text; never blank out existing words.
        if narration:
            panel.narration = narration
        if dialogue:
            panel.dialogue = dialogue
        sidecar[panel.id] = {"model_used": used, "fallback_used": fb,
                             "cached": not fresh}
        summary["narrated" if fresh else "cached"] += 1
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
        # Deliberately NOT passing model_call: make_text_model_call is the
        # story-memory SEED adapter (it sends `_SEED_SYSTEM` as the system
        # prompt with the default output budget), so handing it to the script
        # pass replaced the narrator persona contract with the seed contract
        # and capped the response at 4096 tokens -- far too small for the
        # one-line-per-panel script. Without it, recap_script uses its own
        # SYSTEM_PROMPT + SCRIPT_MAX_TOKENS and expands the key pool itself
        # (api_key stays None, so rotation across accounts keeps working).
        script_summary = build_chapter_script(
            session, api_key=api_key, base_url=base_url, model=model,
            request_fn=request_fn,
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
