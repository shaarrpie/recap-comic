# strip_analyzer.py
"""Phase 1 — AI pre-read of a vertical manhwa strip into a per-panel plan.

Rationale
---------
Manhwa/webtoons arrive as very tall strips (often 800 x 10,000-20,000 px).
Cutting them blindly loses reading-order structure and sometimes splits a
single logical panel across two cut images. This module implements the
"read FIRST, then cut" half of the guided pipeline:

* The full strip is split into overlapping *reading chunks* (default 2000 px
  high, 200 px overlap). Chunks are ONLY for the model — they are never the
  output. Each chunk records the absolute Y of its top edge, so every panel
  boundary the model returns in chunk-local coordinates is mapped back into
  the original strip's pixel space. Nothing is lost across chunk seams.
* The model is forbidden from narrating the whole strip in one block: it must
  return a strict JSON list of {panel_index, y_start, y_end, narration,
  dialogue, panel_type, confidence, bubble_boxes}. Prose or merged panels are
  rejected; the validation error is fed back and the call is retried a
  bounded number of times; if it still fails the raw response is left in the
  backends' debug output so a human can inspect it.
* Phase-1 results are cached keyed by the strip file's SHA-256 + a config
  hash, so a later cut run does not re-spend tokens.

The vision backend is pluggable through the VisionBackend protocol: Gemini
(default, google-genai 2.22.0 verified), OpenAI (openai 3.8.0 signature-checked:
chat.completions.create(..., response_format, ...)), Anthropic (anthropic
1.4.0 signature-checked: messages.create(model, max_tokens, messages, ...),
image blocks verified against the vision docs), and a deterministic Fixture
backend for offline tests. "local" (Ollama etc.) is a TODO adapter.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, PrivateAttr, ValidationError, model_validator

from adapters.schemas import BBox

log = logging.getLogger(__name__)

DEFAULT_CHUNK_HEIGHT = 2000
DEFAULT_CHUNK_OVERLAP = 200
MAX_ATTEMPTS = 3

# Bump this whenever the prompt template, the JSON schema, or the
# normalization convention in _chunk_prompt() / parse_entries_from_json()
# changes in a way that would make a previously cached plan stale. The
# value is folded into the Phase-1 cache key so old plans are not reused.
PROMPT_VERSION = "2026-09-05a"

PANEL_TYPES = frozenset({
    "single", "tall_scenic", "transition_gutter", "multi_sub_panel", "unknown",
})


class VisionAnalysisError(RuntimeError):
    """Raised when Phase 1 cannot produce a usable plan."""


class PanelPlanEntry(BaseModel):
    panel_index: int  # reading order, top -> bottom (1-based)
    y_start: int      # pixel coordinate in the ORIGINAL strip / chunk
    y_end: int
    narration: str = ""
    dialogue: str = ""
    panel_type: str = "unknown"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    bubble_boxes: list[BBox] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_bounds(self) -> PanelPlanEntry:
        if self.y_start < 0 or self.y_end < 0:
            raise ValueError(
                f"panel {self.panel_index}: y coordinates must be >= 0, "
                f"got {self.y_start}, {self.y_end}")
        if self.y_start >= self.y_end:
            raise ValueError(
                f"panel {self.panel_index}: y_start must be < y_end, "
                f"got {self.y_start}, {self.y_end}")
        if self.panel_type not in PANEL_TYPES:
            raise ValueError(
                f"panel {self.panel_index}: invalid panel_type "
                f"{self.panel_type!r}; expected one of {sorted(PANEL_TYPES)}")
        return self


class PanelPlan(BaseModel):
    source: str
    width: int   # of the original strip
    height: int  # of the original strip
    model: str
    config_hash: str
    input_hash: str
    provenance: str = "ai"  # "ai" | "fallback" (gutter detector)
    characters: list[str] = Field(default_factory=list)
    entries: list[PanelPlanEntry]
    _gray_cache: np.ndarray | None = PrivateAttr(default=None)
    """Set by guided_cut() so bubble_detector can run on the real pixels
    without re-reading the strip; never serialized."""

    @model_validator(mode="after")
    def _check_plan(self) -> PanelPlan:
        ys = [e.y_start for e in self.entries]
        if ys != sorted(ys):
            raise ValueError("plan entries must be in top-to-bottom order")
        indices = [e.panel_index for e in self.entries]
        if indices != list(range(1, len(self.entries) + 1)):
            raise ValueError(f"panel_index must be 1..N in order, got {indices}")
        for e in self.entries:
            if e.y_start < 0 or e.y_end > self.height:
                raise ValueError(
                    f"panel {e.panel_index} range [{e.y_start},{e.y_end}] "
                    f"outside strip height {self.height}")
        return self

def make_chunks(image: Image.Image, *, chunk_height: int = DEFAULT_CHUNK_HEIGHT,
                overlap: int = DEFAULT_CHUNK_OVERLAP
                ) -> list[tuple[int, Image.Image]]:
    """Return (absolute_y0, chunk_image) tiles covering the full strip.

    Tiles overlap by `overlap` pixels so a panel straddling a tile edge is
    fully visible in at least one tile. `absolute_y0` is the tile's top edge
    in the original strip; model coordinates are offsets from this value,
    which is exactly how absolute Y positions are preserved across chunks.
    """
    if chunk_height <= 0:
        raise ValueError(f"chunk_height must be > 0, got {chunk_height}")
    if overlap < 0 or overlap >= chunk_height:
        raise ValueError("overlap must be in [0, chunk_height)")
    w, h = image.size
    chunks: list[tuple[int, Image.Image]] = []
    y0 = 0
    step = max(1, chunk_height - overlap)
    while y0 < h:
        y1 = min(y0 + chunk_height, h)
        chunks.append((y0, image.crop((0, y0, w, y1))))
        if y1 == h:
            break
        y0 += step
    return chunks


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_hash(cfg: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_atomic(path: Path, content: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, "utf-8")
    tmp.replace(path)


def extract_json(text: str) -> object:
    """Extract the JSON object/array from a model response that may wrap it
    inside fenced code blocks. Raises ValueError on prose (no JSON at all)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    obj_start, obj_end = text.find("{"), text.rfind("}")
    arr_start, arr_end = text.find("["), text.rfind("]")
    prefer_object = (
        obj_start != -1 and obj_end != -1
        and (arr_start == -1 or (obj_end - obj_start) >= (arr_end - arr_start))
    )
    if prefer_object:
        return json.loads(text[obj_start:obj_end + 1])
    if arr_start != -1 and arr_end != -1:
        return json.loads(text[arr_start:arr_end + 1])
    raise ValueError("no JSON object or array found in model output")


def _normalize_bbox(raw: object) -> BBox:
    if isinstance(raw, BBox):
        return raw
    if isinstance(raw, dict):
        if all(k in raw for k in ("x", "y", "w", "h")):
            return BBox(x=int(raw["x"]), y=int(raw["y"]),
                        w=int(raw["w"]), h=int(raw["h"]))
        if all(k in raw for k in ("x0", "y0", "x1", "y1")):
            return BBox(x=int(raw["x0"]), y=int(raw["y0"]),
                        w=int(raw["x1"]) - int(raw["x0"]),
                        h=int(raw["y1"]) - int(raw["y0"]))
        raise TypeError(f"cannot interpret bubble box {raw!r}")
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        x0, y0, x1, y1 = (int(v) for v in raw)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"degenerate bubble box {raw!r}")
        return BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)
    raise TypeError(f"cannot interpret bubble box {raw!r}")


def parse_entries_from_json(text: str, chunk_height: int, *,
                            normalized: bool = True
                            ) -> tuple[list[PanelPlanEntry], list[str]]:
    """Strict parse + validation of one chunk's model output.

    MODEL-FACING coordinates are normalized 0..1000 (y_start, y_end and
    bubble-box corners are integers proportional to the chunk's height,
    top = 0, bottom = 1000). They are converted to chunk-PIXEL coordinates
    here, so the internal PanelPlan always carries pixels. The response may
    also carry a top-level "characters" list of names seen in the chunk.

    Raises ValueError for prose / malformed / merged / out-of-range payloads
    so callers can retry with the error fed back to the model.
    """
    obj = extract_json(text)
    panels = obj if isinstance(obj, list) else (
        obj.get("panels") if isinstance(obj, dict) else None)
    if not isinstance(panels, list):
        raise TypeError("expected a JSON object with a 'panels' list")
    raw_chars = obj.get("characters") if isinstance(obj, dict) else None
    characters: list[str] = []
    if isinstance(raw_chars, list):
        characters = [str(c).strip() for c in raw_chars if str(c).strip()]

    max_unit = 1000 if normalized else chunk_height
    scale = chunk_height / 1000.0 if normalized else 1.0
    entries: list[PanelPlanEntry] = []
    for raw in panels:
        if not isinstance(raw, dict):
            raise TypeError("each panel must be a JSON object")
        normalized_entry = dict(raw)
        boxes = raw.get("bubble_boxes") or []
        if not isinstance(boxes, list):
            raise TypeError("bubble_boxes must be a list")
        normalized_entry["bubble_boxes"] = [_normalize_bbox(b) for b in boxes]
        try:
            entry = PanelPlanEntry.model_validate(normalized_entry)
        except ValidationError as exc:
            raise ValueError(f"invalid panel entry: {exc}") from exc
        if entry.y_start < 0 or entry.y_end > max_unit:
            raise ValueError(
                f"panel {entry.panel_index} y range [{entry.y_start},"
                f"{entry.y_end}] outside the valid range 0..{max_unit}")
        entry = entry.model_copy(update={
            "y_start": min(chunk_height, round(entry.y_start * scale)),
            "y_end": min(chunk_height, round(entry.y_end * scale)),
            "bubble_boxes": [
                BBox(x=min(chunk_height, round(b.x * scale)),
                     y=min(chunk_height, round(b.y * scale)),
                     w=round(b.w * scale),
                     h=round(b.h * scale))
                for b in entry.bubble_boxes
            ],
        })
        if entry.y_start >= entry.y_end:
            raise ValueError(
                f"panel {entry.panel_index}: degenerate range after scaling "
                f"[{entry.y_start},{entry.y_end}]")
        entries.append(entry)
    return entries, characters

def _dup_of(a: PanelPlanEntry, b: PanelPlanEntry) -> bool:
    """True when b looks like the same panel also seen in another chunk."""
    inter = min(a.y_end, b.y_end) - max(a.y_start, b.y_start)
    if inter <= 0:
        return False
    shorter = min(a.y_end - a.y_start, b.y_end - b.y_start)
    return inter / shorter >= 0.5


def stitch_chunk_results(results: list[list[PanelPlanEntry]],
                         bases: list[int],
                         height: int) -> list[PanelPlanEntry]:
    """Convert chunk-local coords to strip-absolute coords and merge panels
    that appear in two chunks' overlap.

    A panel straddling a chunk seam is only PARTIALLY visible in each of the
    two neighbouring chunks, so the correct merge is a UNION of the two y
    ranges; narration/dialogue prefer the longer (more complete) detection.
    Entries are renumbered 1..N in top-to-bottom order afterwards.
    """
    abs_entries: list[tuple[PanelPlanEntry, int]] = []
    for base, entries in zip(bases, results, strict=True):
        for e in entries:
            y0 = max(0, e.y_start + base)
            y1 = min(height, e.y_end + base)
            if y0 >= y1:
                log.warning("chunk-local panel [%d,%d] maps to an empty "
                            "range; dropping", e.y_start, e.y_end)
                continue
            moved = e.model_copy(update={"y_start": y0, "y_end": y1})
            abs_entries.append((moved, y1 - y0))
    abs_entries.sort(key=lambda t: (t[0].y_start, t[0].y_end))

    merged: list[tuple[PanelPlanEntry, int]] = []
    for e, length in abs_entries:
        if merged and _dup_of(merged[-1][0], e):
            prev, prev_len = merged[-1]
            winner = e if length > prev_len else prev
            merged[-1] = (winner.model_copy(update={
                "y_start": min(prev.y_start, e.y_start),
                "y_end": max(prev.y_end, e.y_end),
            }), max(prev_len, length))
        else:
            merged.append((e, length))

    entries = [e for e, _length in merged]
    for i, e in enumerate(entries, start=1):
        e.panel_index = i  # renumber in reading order after merging
    return entries


def _call_with_retry(
    backend: VisionBackend,
    image: Image.Image,
    attempts: int,
    previous_context: str = "",
) -> tuple[list[PanelPlanEntry], list[str]]:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return backend.analyze_chunk(
                image, previous_context=previous_context)
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last = exc
            log.warning("chunk analysis attempt %d/%d failed: %s",
                        attempt, attempts, exc)
            time.sleep(attempt)
    assert last is not None
    raise last


def analyze_strip(strip_path: str | Path, backend: VisionBackend, *,
                  chunk_height: int = DEFAULT_CHUNK_HEIGHT,
                  overlap: int = DEFAULT_CHUNK_OVERLAP,
                  cache_dir: str | Path | None = None,
                  force: bool = False,
                  attempts: int = MAX_ATTEMPTS,
                  chunk_dir: str | Path | None = None
                  ) -> tuple[PanelPlan, bool]:
    """Phase 1: chunk, read, stitch -> absolute-coordinate PanelPlan.

    Returns (plan, used_cache). Cache is keyed by the strip file's SHA-256 +
    the reading-config hash, so a cut re-run does not re-spend tokens.
    When `chunk_dir` is given, each chunk image sent to the model is saved
    as chunk_XX.png there (visual debug for "what the model saw").
    Every chunk after the first receives a compact continuity context
    (previous narrations + characters seen so far); characters accumulate
    into plan.characters.
    """
    path = Path(strip_path)
    if not path.is_file():
        raise FileNotFoundError(f"strip image not found: {path}")
    input_hash = _sha256_file(path)
    cfg: dict[str, object] = {
        "chunk_height": chunk_height,
        "overlap": overlap,
        "backend": getattr(backend, "name", type(backend).__name__),
    }
    cfg_hash = _config_hash(cfg)

    cache_root = Path(cache_dir) if cache_dir is not None else None
    cache_path: Path | None = None
    if cache_root is not None:
        cache_path = cache_root / f"plan_{input_hash[:16]}_{cfg_hash[:16]}.json"
        if not force and cache_path.is_file():
            plan = PanelPlan.model_validate_json(cache_path.read_text("utf-8"))
            if plan.input_hash == input_hash and plan.config_hash == cfg_hash:
                log.info("phase-1 cache hit: %s", cache_path)
                return plan, True

    chunk_dir_p = Path(chunk_dir) if chunk_dir is not None else None
    with Image.open(path) as img:
        img.load()
        width, height = img.size
        chunks = make_chunks(img, chunk_height=chunk_height, overlap=overlap)
        results: list[list[PanelPlanEntry]] = []
        bases: list[int] = []
        prev_entries: list[PanelPlanEntry] = []
        characters: list[str] = []
        for idx, (base, chunk) in enumerate(chunks):
            if chunk_dir_p is not None:
                chunk_dir_p.mkdir(parents=True, exist_ok=True)
                chunk.save(chunk_dir_p / f"chunk_{idx:02d}.png", "PNG")
            context = build_context(prev_entries, characters)
            try:
                entries, new_chars = _call_with_retry(
                    backend, chunk, attempts=attempts,
                    previous_context=context)
            except Exception as exc:
                raise VisionAnalysisError(
                    f"chunk {idx} (absolute y0={base}) failed after "
                    f"{attempts} attempt(s): {exc}") from exc
            prev_entries = entries
            for name in new_chars:
                if name not in characters:
                    characters.append(name)
            results.append(entries)
            bases.append(base)

    stitched = stitch_chunk_results(results, bases, height)
    if not stitched:
        raise VisionAnalysisError("the model returned no panels for this strip")
    plan = PanelPlan(
        source=path.name, width=width, height=height,
        model=getattr(backend, "name", type(backend).__name__),
        config_hash=cfg_hash, input_hash=input_hash, entries=stitched,
        characters=characters)
    if cache_path is not None and cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
        _write_atomic(cache_path, plan.model_dump_json(indent=2) + "\n")
    return plan, False

class VisionBackend(Protocol):
    """Pluggable vision backend for Phase 1.

    The orchestrator `analyze_strip()` implements the requested
    `analyze_strip(image_chunks) -> PanelPlan` interface; each backend
    implements the per-image primitive `analyze_chunk(image,
    previous_context="") -> (entries, characters)` in CHUNK-PIXEL
    coordinates. `characters` is the list of character names the model
    positively identified in this chunk (for cross-chunk continuity).
    Backends may also fill `usage_log` / `last_usage` (best-effort token
    counts) — the smoke test reads them if present.
    """
    name: str

    def analyze_chunk(
        self,
        image: Image.Image,
        previous_context: str = "",
    ) -> tuple[list[PanelPlanEntry], list[str]]:
        """Return (entries, characters), both for this image only."""
        ...


class FixtureVisionBackend:
    """Deterministic backend for offline tests: returns canned per-chunk
    entries (chunk-local pixel coordinates) in order."""
    name = "fixture"

    def __init__(self, entries_by_chunk: list[list[PanelPlanEntry]]
                 | None = None) -> None:
        self._plans = list(entries_by_chunk or [])
        self.calls = 0

    def analyze_chunk(self, image: Image.Image,
                      previous_context: str = ""
                      ) -> tuple[list[PanelPlanEntry], list[str]]:
        self.calls += 1
        if not self._plans:
            return [], []
        return self._plans.pop(0), []


def build_context(last_entries: list[PanelPlanEntry],
                  characters: list[str], n: int = 3) -> str:
    """Compact continuity context for the NEXT chunk: the last `n`
    narrations and every character name seen so far."""
    parts: list[str] = []
    narrations = [e.narration.strip() for e in last_entries
                  if e.narration.strip()]
    for narration in narrations[-n:]:
        parts.append(f"- narration: {narration[:400]}")
    if characters:
        parts.append(f"- characters seen so far: {', '.join(characters)}")
    return "\n".join(parts)


def _capture_usage(resp: object) -> dict | None:
    """Best-effort token-usage capture. Defensive getattr against the
    standard SDK usage objects (Gemini UsageMetadata, OpenAI Usage,
    Anthropic Usage, Ollama dict). Attribute shapes are standard training
    knowledge and were NOT verified against a live API in this session."""
    meta = getattr(resp, "usage_metadata", None)
    if meta is None:
        meta = getattr(resp, "usage", None)
    if meta is None and isinstance(resp, dict):
        meta = resp
    if meta is None:
        return None
    out: dict[str, int] = {}
    for name in (
        "prompt_token_count", "candidates_token_count", "total_token_count",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "input_tokens", "output_tokens", "prompt_eval_count", "eval_count",
    ):
        try:
            value = getattr(meta, name, None)
        except AttributeError:
            value = None
        if value is None and isinstance(meta, dict):
            value = meta.get(name)
        if value is not None:
            try:
                out[name] = int(value)
            except (TypeError, ValueError):
                continue
    return out or None


def _chunk_prompt(height: int, previous_context: str = "") -> str:
    """Strict per-chunk dissection instructions (the prompt template).

    Coordinates are NORMALIZED: y_start/y_end and bubble-box corners are
    integers in 0..1000 proportional to THIS image's height (top = 0,
    bottom = 1000), never raw pixels. Optionally includes compact narrative
    context from earlier chunks so narration stays coherent across the strip.
    """
    context_block = ""
    if previous_context.strip():
        context_block = (
            "\nPREVIOUS CONTEXT (earlier chunks of the same strip):\n"
            f"{previous_context.strip()}\n"
            "Use it ONLY for continuity of narration and character identity. "
            "Do NOT repeat, re-narrate, or borrow panels from it.\n"
        )
    return (
        "You are dissecting a vertical manhwa (webtoon) strip image into its "
        "logical panels for a narrated recap video. The image you are viewing "
        f"is a SLICE of a taller strip and is {height} pixels tall.\n\n"
        "COORDINATES ARE NORMALIZED 0..1000: every y_start, y_end and every "
        "bubble-box corner is an integer in [0, 1000] measured PROPORTIONALLY "
        "to this image's height (top = 0, bottom = 1000), NEVER raw pixels. "
        "For example, a panel occupying the top quarter of the image is "
        "y_start=0, y_end=250; a bubble around the vertical middle is "
        "bubble_boxes: [[420, 480, 580, 540]]."
        f"{context_block}\n\n"
        "Return STRICT JSON only - no prose, no commentary, no markdown, and "
        "never merge two panels into one entry. Every entry is exactly one "
        "logical panel. The JSON must match this exact shape:\n\n"
        '{"panels": [{"panel_index": 1, "y_start": 120, "y_end": 520, '
        '"narration": "...", "dialogue": "...", "panel_type": "single", '
        '"confidence": 0.95, "bubble_boxes": [[x0, y0, x1, y1], ...]}], '
        '"characters": ["Name"]}\n\n'
        "Rules:\n"
        "- panel_index starts at 1 and increases strictly top to bottom.\n"
        "- y_start < y_end and both are integers inside [0, 1000].\n"
        "- narration describes ONLY what is visible in that panel: characters, "
        "action, setting, mood. Never invent events that are not shown.\n"
        "- dialogue lists speech-bubble and SFX text verbatim, or an empty "
        "string.\n"
        "- panel_type is exactly one of: single, tall_scenic, "
        "transition_gutter, multi_sub_panel.\n"
        "- confidence (0..1) estimates how reliable the boundary estimate is.\n"
        "- bubble_boxes: normalized boxes around each speech bubble (integers "
        "in 0..1000), or an empty list.\n"
        "- characters: names positively identifiable in THIS image, or an "
        "empty list. Never invent names.\n"
    )

class GeminiVisionBackend:
    """Gemini backend (default). Verified in this session:
    - google-genai==2.22.0 import-checked: types.Part.from_bytes / from_text
      and types.GenerateContentConfig all exist;
    - generate_content(model=..., contents=[...], config=...) with
      response_mime_type="application/json" is the documented JSON mode
      (ai.google.dev/gemini-api/docs/json-mode, updated 2026-09-02).
    NOT executed against the API in this session (no API key available)."""
    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash",
                 api_key: str | None = None,
                 max_attempts: int = MAX_ATTEMPTS) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set; pass api_key or set the env var")
        self.max_attempts = max_attempts
        self.usage_log: list[dict] = []
        self.last_usage: dict | None = None

    def analyze_chunk(self, image: Image.Image,
                      previous_context: str = ""
                      ) -> tuple[list[PanelPlanEntry], list[str]]:
        from google import genai  # optional dependency, imported lazily
        from google.genai import types

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        client = genai.Client(api_key=self._api_key)
        last_err = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                prompt = _chunk_prompt(image.size[1], previous_context)
                resp = client.models.generate_content(
                    model=self.model,
                    contents=[  # type: ignore[arg-type]  # SDK stub list-variance quirk
                        types.Part.from_text(text=prompt + last_err),
                        types.Part.from_bytes(
                            data=buf.getvalue(), mime_type="image/png"),
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
                raw = resp.text
                if raw is None:
                    raw = ""
                self.last_usage = _capture_usage(resp)
                self.usage_log.append(self.last_usage or {})
                return parse_entries_from_json(raw, image.size[1])
            except Exception as exc:  # noqa: BLE001 - retried then re-raised
                last_err = (
                    f"\n\nPrevious attempt failed with: {exc}\n"
                    "Fix the JSON and return only the required shape."
                )
                time.sleep(attempt)
        raise VisionAnalysisError(
            f"gemini analysis failed after {self.max_attempts} attempts")


class OpenAIVisionBackend:
    """OpenAI backend. Verified in this session (openai==3.8.0): the SDK
    exposes client.chat.completions.create(messages, model, max_tokens,
    response_format, ...) and client.chat.completions.parse. NOT executed
    against the API in this session (no key). The parse+retry loop below is
    the safety net for any response_format quirk on specific model families."""
    name = "openai"

    def __init__(self, model: str, api_key: str | None = None,
                 max_attempts: int = MAX_ATTEMPTS) -> None:
        if not model or not model.strip():
            raise ValueError("--model is required for the openai backend")
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set; pass api_key or set the env var")
        self.max_attempts = max_attempts
        self.usage_log: list[dict] = []
        self.last_usage: dict | None = None

    def analyze_chunk(self, image: Image.Image,
                      previous_context: str = ""
                      ) -> tuple[list[PanelPlanEntry], list[str]]:
        import base64

        import openai  # optional dependency, imported lazily

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        client = openai.OpenAI(api_key=self._api_key)
        last_err = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                prompt = _chunk_prompt(image.size[1], previous_context)
                resp = client.chat.completions.create(
                    model=self.model,
                    max_tokens=4096,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system",
                         "content": prompt + last_err},
                        {"role": "user",
                         "content": [{"type": "image_url", "image_url": {
                             "url": f"data:image/png;base64,{b64}"}}]},
                    ],
                )
                raw_text = resp.choices[0].message.content
                if raw_text is None:
                    raw_text = ""
                self.last_usage = _capture_usage(resp)
                self.usage_log.append(self.last_usage or {})
                return parse_entries_from_json(raw_text, image.size[1])
            except Exception as exc:  # noqa: BLE001 - retried then re-raised
                last_err = (
                    f"\n\nPrevious attempt failed with: {exc}\n"
                    "Fix the JSON and return only the required shape."
                )
                time.sleep(attempt)
        raise VisionAnalysisError(
            f"openai analysis failed after {self.max_attempts} attempts")

class AnthropicVisionBackend:
    """Anthropic backend. Verified in this session (anthropic==1.4.0):
    messages.create(model, max_tokens, messages, ...) — image content blocks
    with source type "base64" are recorded in the current vision docs.
    Structured JSON is enforced by the strict prompt + parse/retry loop here
    (no output_config usage, so no unverified SDK surface is invoked).
    NOT executed against the API in this session (no key)."""
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None,
                 max_attempts: int = MAX_ATTEMPTS) -> None:
        if not model or not model.strip():
            raise ValueError("--model is required for the anthropic backend")
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; pass api_key or set the env var")
        self.max_attempts = max_attempts
        self.usage_log: list[dict] = []
        self.last_usage: dict | None = None

    def analyze_chunk(self, image: Image.Image,
                      previous_context: str = ""
                      ) -> tuple[list[PanelPlanEntry], list[str]]:
        import base64

        import anthropic  # optional dependency, imported lazily

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        client = anthropic.Anthropic(api_key=self._api_key)
        last_err = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                prompt = _chunk_prompt(image.size[1], previous_context)
                resp = client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image",
                             "source": {"type": "base64",
                                        "media_type": "image/png",
                                        "data": b64}},
                            {"type": "text",
                             "text": prompt + last_err},
                        ],
                    }],
                )
                text = "".join(
                    str(getattr(block, "text", None) or "")
                    for block in resp.content)
                self.last_usage = _capture_usage(resp)
                self.usage_log.append(self.last_usage or {})
                return parse_entries_from_json(text, image.size[1])
            except Exception as exc:  # noqa: BLE001 - retried then re-raised
                last_err = (
                    f"\n\nPrevious attempt failed with: {exc}\n"
                    "Fix the JSON and return only the required shape."
                )
                time.sleep(attempt)
        raise VisionAnalysisError(
            f"anthropic analysis failed after {self.max_attempts} attempts")


class OllamaVisionBackend:
    """Local, free, offline backend via Ollama's POST /api/generate.

    Verified against the Ollama API docs (fetched 2026-09-05):
    - request params: model (required), prompt, images (list of base64),
      format ("json" enables JSON mode — docs: "Enable JSON mode by setting
      the format parameter to json"), stream (false -> a single response
      object);
    - response object carries the generated "response" text.
    Structured outputs (JSON schema in `format`) are documented too; we use
    the simpler json mode plus the strict parse/retry safety net.
    NOT executed in this session (no local Ollama server/model available).
    """
    name = "ollama"

    def __init__(self, model: str = "llava", base_url: str | None = None,
                 timeout: int = 120, max_attempts: int = MAX_ATTEMPTS) -> None:
        if not model or not model.strip():
            raise ValueError("--model is required for the ollama backend")
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL")
                         or "http://localhost:11434").rstrip("/")
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.usage_log: list[dict] = []
        self.last_usage: dict | None = None

    def _build_payload(self, image: Image.Image, prompt: str) -> dict:
        import base64
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return {"model": self.model, "prompt": prompt, "images": [b64],
                "stream": False, "format": "json"}

    def analyze_chunk(self, image: Image.Image,
                      previous_context: str = ""
                      ) -> tuple[list[PanelPlanEntry], list[str]]:
        import urllib.request

        last_err = ""
        for attempt in range(1, self.max_attempts + 1):
            prompt = _chunk_prompt(image.size[1], previous_context) + last_err
            payload = self._build_payload(image, prompt)
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                text = data.get("response", "")
                self.last_usage = _capture_usage(data)
                self.usage_log.append(self.last_usage or {})
                return parse_entries_from_json(text, image.size[1])
            except Exception as exc:  # noqa: BLE001 - retried then re-raised
                last_err = (
                    f"\n\nPrevious attempt failed with: {exc}\n"
                    "Fix the JSON and return only the required shape."
                )
                time.sleep(attempt)
        raise VisionAnalysisError(
            f"ollama analysis failed after {self.max_attempts} attempts")
