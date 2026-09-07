# adapters/narrate_gemini.py
"""Gemini narration adapter + grounding check + verbatim fallback.

Verified against the google-genai SDK README (opened 2026-09-05):
  from google import genai; from google.genai import types
  client = genai.Client(api_key=...)          # or GEMINI_API_KEY env var
  response = client.models.generate_content(model=..., contents=..., config=...)
  response.text
JSON output: response_mime_type="application/json" in GenerateContentConfig,
verified against ai.google.dev/gemini-api/docs/json-mode (updated 2026-09-02).
Grounding and verbatim modes are offline and need no API.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from .schemas import NarrationArtifact, NarrationEntry, OcrArtifact

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the narrator for a recap video of a manhwa chapter. You will receive
OCR text extracted from the chapter's panels, in reading order.

STRICT RULES - violations make the output unusable:
1. Use ONLY information present in the provided OCR text. Never invent
   events, names, motives, dialogue, or plot connections.
2. Preserve panel order. One output entry per input panel, same order.
3. If a panel has no usable text, or you cannot tell what is happening,
   emit an empty string for that panel ("text": ""). Do NOT guess.
4. Distinguish narration from quoted dialogue. Any quoted string you output
   MUST appear verbatim in the OCR text (ignore letter-case and punctuation).
5. Keep character identities consistent: the same name always refers to the
   same character. Never assign a name that does not appear in the OCR text.
6. Ignore sound effects and onomatopoeia (entries marked kind="sfx").
7. Output ONLY a JSON object matching the requested schema. No commentary.
"""

USER_PROMPT_TEMPLATE = """\
Narration mode: {mode}
{speaker_rules}
OCR text for this chapter (panel id -> text lines; "sfx" lines are sound
effects and must not be narrated):
{ocr_dump}

Return JSON: {{"entries": [{{"panel_id": "...", "speaker": null or "Name",
"text": "..."}}]}} with exactly {n_panels} entries, in panel order.
"""


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def extract_quotes(text: str) -> list[str]:
    return re.findall(r'"([^"]+)"', text)


def check_grounding(narration: NarrationArtifact,
                    ocr: OcrArtifact) -> list[str]:
    """Return quotes in narration not found (normalized) in ocr.json."""
    corpus = normalize(" ".join(r.text for r in ocr.regions))
    bad: list[str] = []
    for e in narration.entries:
        for q in e.quotes:
            if normalize(q) and normalize(q) not in corpus:
                bad.append(q)
    return bad


def build_verbatim(ocr: OcrArtifact, panel_ids: list[str]) -> NarrationArtifact:
    """No-API fallback: dialogue lines become narration entries directly."""
    by_panel: dict[str, list[str]] = {pid: [] for pid in panel_ids}
    for r in ocr.regions:
        if r.kind == "sfx" or r.panel_id is None or not r.text:
            continue
        if r.panel_id in by_panel:
            by_panel[r.panel_id].append(r.text)
    entries = [NarrationEntry(id=pid, panel_id=pid, order=i,
                              text=" ".join(by_panel[pid]),
                              quotes=[t for t in by_panel[pid]])
               for i, pid in enumerate(panel_ids)]
    return NarrationArtifact(mode="verbatim", entries=entries, meta=ocr.meta)


def generate(narration_request: str, ocr: OcrArtifact, panel_ids: list[str],
               mode: str = "narrator", *, model: str = "gemini-2.0-flash",
               max_attempts: int = 3, debug_dir: Path = Path("llm_debug")
               ) -> NarrationArtifact:
    from google import genai  # lazy import
    from google.genai import types

    from adapters._gemini_keys import from_env

    rotator = from_env()
    dump = "\n".join(
        f"{r.panel_id or '?'} [{r.kind}] conf={r.confidence}: {r.text}"
        for r in ocr.regions)
    user = USER_PROMPT_TEMPLATE.format(
        mode=mode,
        speaker_rules="- Assign each entry a speaker only if the name is "
                      "present in the OCR text; otherwise null.\n"
                      if mode == "characters" else "",
        ocr_dump=dump, n_panels=len(panel_ids))
    last_err = ""
    last_raw = ""
    last_exc: Exception | None = None
    max_rotation = rotator.total + 1
    for attempt in range(1, max(max_attempts, max_rotation) + 1):
        api_key = rotator.current()
        client = genai.Client(api_key=api_key)
        try:
            resp = client.models.generate_content(
                model=model, contents=[SYSTEM_PROMPT, user + last_err],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json"))
            last_raw = resp.text or ""
            data = json.loads(last_raw)
            narration = NarrationArtifact.model_validate(
                {**data, "mode": mode, "meta": ocr.meta})
            narration.entries.sort(key=lambda e: e.order)
            if [e.panel_id for e in narration.entries] != panel_ids:
                raise ValueError("panel ids/order do not match panels.json")
            bad = check_grounding(narration, ocr)
            if not bad:
                narration.ungrounded_quotes = []
                return narration
            narration = _grounding_retry(
                client, narration, ocr, bad, model, debug_dir)
            remaining_bad = check_grounding(narration, ocr)
            narration.ungrounded_quotes = remaining_bad
            return narration
        except Exception as exc:  # noqa: BLE001 - every failure is retried,
            # then re-raised with context; never silently swallowed
            last_exc = exc
            msg = str(exc).lower()
            is_quota = (
                "429" in msg
                or "resource_exhausted" in msg
                or "quota" in msg
            )
            if is_quota and attempt < max_rotation:
                rotator.advance()
                log.warning(
                    "gemini quota error on key ending %s; rotated to next key (%d/%d)",
                    api_key[-4:], attempt + 1, max_rotation
                )
                continue
            last_err = (
                f"\n\nPrevious attempt failed validation: {exc}\nFix it."
            )
            time.sleep(attempt)
    debug_dir.mkdir(exist_ok=True)
    (debug_dir / f"narration_raw_{int(time.time())}.txt").write_text(
        last_raw, encoding="utf-8")
    raise RuntimeError(
        f"narration failed after {max_attempts} attempts; raw "
        f"response saved under {debug_dir}") from last_exc


def _grounding_retry(client, narration: NarrationArtifact,
                      ocr: OcrArtifact, bad_quotes: list[str],
                      model: str, debug_dir: Path) -> NarrationArtifact:
    """One-shot targeted retry: feed only the offending panels' OCR regions
    and the ungrounded quotes back to the model, asking it to either ground
    each quote in the provided OCR or drop it."""
    from google.genai import types

    affected_panel_ids = {e.panel_id for e in narration.entries
                          for q in e.quotes if q in bad_quotes}
    ocr_by_panel: dict[str, list[str]] = {}
    for r in ocr.regions:
        if r.panel_id and r.text and r.panel_id in affected_panel_ids:
            ocr_by_panel.setdefault(r.panel_id, []).append(r.text)
    if not ocr_by_panel:
        return narration

    ocr_dump = "\n".join(
        f"{pid}: {' | '.join(lines)}" for pid, lines in ocr_by_panel.items())
    follow_up = (
        "GROUNDING AUDIT: the following quotes were flagged as NOT found "
        "in the full OCR corpus. For EACH quote, either:\n"
        "  a) return the exact OCR text that supports it (if present in "
        "the panel OCR below), or\n"
        "  b) remove the quote from the panel's narration entirely.\n"
        f"Flagged quotes: {json.dumps(bad_quotes, ensure_ascii=False)}\n"
        f"Panel OCR for affected panels:\n{ocr_dump}\n"
        "Return the SAME JSON schema with updated entries for ONLY the "
        "affected panels. Keep all other panels unchanged."
    )
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[follow_up],
            config=types.GenerateContentConfig(
                temperature=0.2, response_mime_type="application/json"))
        data = json.loads(resp.text)
        updated = NarrationArtifact.model_validate(
            {**data, "mode": narration.mode, "meta": narration.meta})
        updated.entries.sort(key=lambda e: e.order)
        by_id = {e.panel_id: e for e in updated.entries}
        for _i, e in enumerate(narration.entries):
            if e.panel_id in by_id:
                replacement = by_id[e.panel_id]
                e.text = replacement.text
                e.quotes = replacement.quotes
                e.speaker = replacement.speaker
        still_bad = check_grounding(narration, ocr)
        if still_bad:
            for e in narration.entries:
                e.quotes = [q for q in e.quotes if q not in still_bad]
    except Exception as exc:  # noqa: BLE001 - best-effort audit
        debug_dir.mkdir(exist_ok=True)
        (debug_dir / f"grounding_retry_{int(time.time())}.txt").write_text(
            f"Grounding retry failed: {exc}\nQuotes: {bad_quotes}\n",
            encoding="utf-8")
    return narration
