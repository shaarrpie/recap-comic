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

The vision backend is pluggable through the VisionBackend protocol: Agnes
(Agnes AI gateway, OpenAI-compatible chat.completions.create(...,
response_format, ...) with image_url input) and a deterministic Fixture
backend for offline tests.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, PrivateAttr, ValidationError, model_validator

from adapters.schemas import BBox

# Set to True by the webapp (webapp.pipeline) before running jobs. The
# rich.Progress spinner in analyze_strip is a CLI affordance: inside the
# server it garbles the uvicorn console (10+ jobs each drawing their own
# 0% bar into the same terminal) and the frontend tracks progress via
# job stages anyway. When embedded, progress is logged instead.
EMBEDDED_MODE = False

try:
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

log = logging.getLogger(__name__)

DEFAULT_CHUNK_HEIGHT = 2000
DEFAULT_CHUNK_OVERLAP = 200
# Speculative-prefetch depth for the vision chunk loop. Each chunk is
# submitted with the freshest context available at submit time (the most
# recent chunk whose result has already landed), so with depth K chunk i's
# context is roughly from chunk i-K — slightly stale continuity, but panel
# boundaries come from each chunk's own image, not the context. Set 1 to
# serialize exactly as before. Tunable via MANHWA_CHUNK_CONCURRENCY.
CHUNK_CONCURRENCY = max(1, int(os.environ.get("MANHWA_CHUNK_CONCURRENCY", "8")))
MAX_ATTEMPTS = 3

# Bump this whenever the prompt template, the JSON schema, or the
# normalization convention in _chunk_prompt() / parse_entries_from_json()
# changes in a way that would make a previously cached plan stale. The
# value is folded into the Phase-1 cache key so old plans are not reused.
PROMPT_VERSION = "2026-09-06c"

PANEL_TYPES = frozenset({
    "single", "tall_scenic", "transition_gutter", "multi_sub_panel", "unknown",
})


class VisionAnalysisError(RuntimeError):
    """Raised when Phase 1 cannot produce a usable plan."""


class EmptyChunkResult(ValueError):
    """The model returned VALID JSON with zero panels for one chunk.

    This is a legitimate answer, not a failure: _chunk_prompt explicitly
    tells the model to return {"panels": [], "characters": []} when it
    cannot comply, and a genuinely blank stretch between scenes is common
    in webtoons. It is raised (not silently returned) so a primary-model
    empty still gets the fallback model's chance; if BOTH models come back
    empty, analyze_chunk accepts it as a blank chunk instead of letting the
    whole strip degrade to the no-narration gutter fallback.
    """


class TruncatedResponseError(ValueError):
    """The model stopped because it hit the output-token ceiling.

    A truncated response is NOT a JSON parse error: the payload is cut off
    mid-object, so re-sending the identical prompt at temperature 0 reproduces
    the same cut-off answer. Historically it surfaced as a generic decode
    error, was retried verbatim, failed again, and the strip silently degraded
    to the no-AI gutter fallback ("some strips come out with no narration").
    Detecting it explicitly lets the retry ask for a SHORTER answer instead.

    Derives from ValueError because that is what the backends and the xkiro
    primary->fallback wrapper already treat as "this response is unusable,
    try the next model".
    """


# Provider finish/stop reasons that mean "output token ceiling reached":
#   OpenAI-compatible endpoints (incl. Agnes)  "length" / "max_tokens"
#   (other reason strings are matched defensively; see _TRUNCATION_REASONS)
_TRUNCATION_REASONS = frozenset({
    "length",
    "max_tokens",
    "max_output_tokens",
    "max_tokens_reached",
    "token_limit",
    "string_above_max_length",
})

# Sent as retry_feedback (see _call_with_retry) when a chunk was truncated.
# The retry must change the ANSWER, not just repeat the question.
TRUNCATION_RETRY_FEEDBACK = (
    "ERROR: your previous answer was TRUNCATED — it hit the output token "
    "limit, so the JSON was cut off mid-object and could not be read. Send a "
    "COMPLETE, valid JSON object this time. Keep every narration and dialogue "
    "under 25 words, do not repeat the schema, and do not omit any required "
    "field. If this chunk holds more panels than you can describe in one "
    "answer, describe FEWER panels with short text rather than letting the "
    "trailing JSON be cut off."
)


def finish_reason_label(reason: object) -> str:
    """Normalize a provider finish/stop reason to a lowercase label.

    Gemini returns a `types.FinishReason` enum whose str() is
    "FinishReason.MAX_TOKENS", so the enum NAME is what matters; the
    OpenAI-compatible endpoints return the bare string ("length").
    """
    name = getattr(reason, "name", None) or str(reason)
    return name.rsplit(".", 1)[-1].strip().lower()


def is_truncated_response(reason: object) -> bool:
    """True when `reason` says the model hit its output-token ceiling."""
    if reason is None:
        return False
    return finish_reason_label(reason) in _TRUNCATION_REASONS


def raise_if_truncated(reason: object, *, backend: str, model: str = "",
                       max_tokens: int | None = None,
                       image_size: tuple[int, int] | None = None) -> None:
    """Raise TruncatedResponseError when the provider reported truncation."""
    if not is_truncated_response(reason):
        return
    bits = [f"{backend} response was TRUNCATED (finish reason "
            f"{finish_reason_label(reason)!r})"]
    if model:
        bits.append(f"model={model}")
    if max_tokens:
        bits.append(f"max_output_tokens={max_tokens}")
    if image_size is not None:
        bits.append(f"chunk={image_size[0]}x{image_size[1]}")
    raise TruncatedResponseError(
        ", ".join(bits) + ": the model hit the output token ceiling and the "
        "JSON is incomplete. This is NOT a parse bug — the answer was cut off.")


def _first_choice(resp: object) -> object | None:
    """resp.choices[0] for the OpenAI-compatible SDKs, else None."""
    choices = getattr(resp, "choices", None)
    if not choices:
        return None
    try:
        return choices[0]
    except (IndexError, TypeError):
        return None


def _openai_finish_reason(resp: object | None) -> object | None:
    choice = _first_choice(resp) if resp is not None else None
    return getattr(choice, "finish_reason", None)


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


def write_atomic(path: Path, content: str) -> None:
    """Write `content` to `path` atomically (write tmp, then rename)."""
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
            x, y, w, h = int(raw["x"]), int(raw["y"]), int(raw["w"]), int(raw["h"])
            if w <= 0 or h <= 0:
                raise ValueError(f"degenerate bubble box {raw!r}: non-positive width/height")
            return BBox(x=x, y=y, w=w, h=h)
        if all(k in raw for k in ("x0", "y0", "x1", "y1")):
            x0, y0, x1, y1 = int(raw["x0"]), int(raw["y0"]), int(raw["x1"]), int(raw["y1"])
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"degenerate bubble box {raw!r}: x1<=x0 or y1<=y0")
            return BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)
        raise TypeError(f"cannot interpret bubble box {raw!r}")
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        x0, y0, x1, y1 = (int(v) for v in raw)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"degenerate bubble box {raw!r}")
        return BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)
    raise TypeError(f"cannot interpret bubble box {raw!r}")


def parse_entries_from_json(text: str, chunk_height: int, *,
                            chunk_width: int | None = None,
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
    scale_y = chunk_height / 1000.0 if normalized else 1.0
    scale_x = (chunk_width if chunk_width is not None else chunk_height) / 1000.0 if normalized else 1.0
    cw = chunk_width if chunk_width is not None else chunk_height
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
            log.debug("panel %d y range [%d,%d] outside 0..%d; clamping",
                      entry.panel_index, entry.y_start, entry.y_end, max_unit)
            normalized_entry["y_start"] = max(0, entry.y_start)
            normalized_entry["y_end"] = min(max_unit, entry.y_end)
            try:
                entry = PanelPlanEntry.model_validate(normalized_entry)
            except ValidationError as exc:
                raise ValueError(f"invalid panel entry after clamping: {exc}") from exc
        entry = entry.model_copy(update={
            "y_start": min(chunk_height, round(entry.y_start * scale_y)),
            "y_end": min(chunk_height, round(entry.y_end * scale_y)),
            "bubble_boxes": [
                BBox(x=min(cw, round(b.x * scale_x)),
                     y=min(chunk_height, round(b.y * scale_y)),
                     w=round(b.w * scale_x),
                     h=round(b.h * scale_y))
                for b in entry.bubble_boxes
            ],
        })
        if entry.y_start >= entry.y_end:
            raise ValueError(
                f"panel {entry.panel_index}: degenerate range after scaling "
                f"[{entry.y_start},{entry.y_end}]")
        entries.append(entry)
    log.debug("parsed %d panels + %d characters from model response",
              len(entries), len(characters))
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
            # Offset bubble_boxes to strip coordinates
            moved_bubbles = [
                BBox(x=b.x, y=b.y + base, w=b.w, h=b.h)
                for b in e.bubble_boxes
            ]
            moved = e.model_copy(update={"y_start": y0, "y_end": y1, "bubble_boxes": moved_bubbles})
            abs_entries.append((moved, y1 - y0))
    abs_entries.sort(key=lambda t: (t[0].y_start, t[0].y_end))

    merged: list[tuple[PanelPlanEntry, int]] = []
    for e, length in abs_entries:
        if merged and _dup_of(merged[-1][0], e):
            prev, prev_len = merged[-1]
            winner = e if length > prev_len else prev
            # Union bubble_boxes from both entries
            union_bubbles = list(prev.bubble_boxes) + list(e.bubble_boxes)
            merged[-1] = (winner.model_copy(update={
                "y_start": min(prev.y_start, e.y_start),
                "y_end": max(prev.y_end, e.y_end),
                "bubble_boxes": union_bubbles,
            }), max(prev_len, length))
        else:
            merged.append((e, length))

    entries = [e for e, _length in merged]
    # Renumber panel_index using model_copy to avoid mutating Pydantic models
    entries = [e.model_copy(update={"panel_index": i}) for i, e in enumerate(entries, start=1)]
    return entries


def _parse_duration(raw: str) -> float:
    """Parse a duration string like '51s', '1.5s', '2m' to seconds."""
    raw = raw.strip()
    if raw.endswith("ms"):
        return float(raw[:-2]) / 1000.0
    if raw.endswith("s"):
        return float(raw[:-1])
    if raw.endswith("m"):
        return float(raw[:-1]) * 60.0
    return float(raw)


def _parse_retry_delay(exc: Exception) -> float | None:
    """Extract retry delay in seconds from a quota/429 exception if present."""
    msg = str(exc)
    try:
        data = json.loads(msg)
        details = data.get("error", {}).get("details", [])
        for d in details:
            if isinstance(d, dict) and d.get("@type", "").endswith("RetryInfo"):
                raw = d.get("retryDelay", "")
                if raw:
                    return _parse_duration(raw)
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)\s*s"', msg)
    if m:
        return float(m.group(1))
    m = re.search(r'[Rr]etry [Ii]n (\d+(?:\.\d+)?)\s*s', msg)
    if m:
        return float(m.group(1))
    return None


def _backend_accepts_feedback(backend: VisionBackend) -> bool:
    """True when backend.analyze_chunk accepts a retry_feedback kwarg."""
    import inspect
    try:
        sig = inspect.signature(backend.analyze_chunk)
    except (TypeError, ValueError):
        return False
    return any(p.kind == inspect.Parameter.VAR_KEYWORD
               or p.name == "retry_feedback" for p in sig.parameters.values())


def _call_with_retry(
    backend: VisionBackend,
    image: Image.Image,
    attempts: int,
    previous_context: str = "",
) -> tuple[list[PanelPlanEntry], list[str]]:
    last: Exception | None = None
    retry_feedback = ""  # parse/validation error fed back to the model
    supports_feedback = _backend_accepts_feedback(backend)
    for attempt in range(1, attempts + 1):
        try:
            kwargs: dict[str, object] = {"previous_context": previous_context}
            if retry_feedback and attempt > 1 and supports_feedback:
                kwargs["retry_feedback"] = retry_feedback
            return backend.analyze_chunk(image, **kwargs)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last = exc
            msg = str(exc).lower()
            is_quota = (
                "429" in msg
                or "resource_exhausted" in msg
                or "rate_limit" in msg
            )
            # Quota exhaustion: don't waste 60s sleeping per attempt — surface
            # the error immediately so the gutter-detector fallback can take
            # over and the user sees a clear message instead of a 3-attempt
            # stall that still fails.
            if is_quota and attempt < attempts:
                quota_delay = _parse_retry_delay(exc) or 0
                log.warning(
                    "chunk analysis attempt %d/%d hit a quota/rate-limit "
                    "(%s); not retrying on the same key — failing fast so "
                    "the fallback can take over (retry-after ~%.0fs)",
                    attempt, attempts, exc, quota_delay)
                raise
            # Feed parse/validation errors back to the model on the next
            # attempt (temperature=0 means an identical prompt will produce
            # an identical bad answer; the feedback is what changes it).
            retry_feedback = str(exc)
            if isinstance(exc, TruncatedResponseError):
                # An identical prompt at temperature 0 reproduces the same
                # truncation byte-for-byte, so the retry must ask for a
                # SHORTER answer. Log it loudly: a truncation that survives
                # all attempts must never quietly become "no narration".
                retry_feedback = TRUNCATION_RETRY_FEEDBACK
                log.warning(
                    "chunk analysis attempt %d/%d was TRUNCATED (%s); "
                    "retrying with a 'send a shorter, complete answer' "
                    "instruction instead of the identical prompt",
                    attempt, attempts, exc)
            retry_delay = _parse_retry_delay(exc)
            if retry_delay is not None and attempt < attempts:
                log.warning("chunk analysis attempt %d/%d failed: %s; retrying in %.1fs",
                            attempt, attempts, exc, retry_delay)
                time.sleep(retry_delay)
                continue
            log.warning("chunk analysis attempt %d/%d failed: %s",
                        attempt, attempts, exc)
            time.sleep(attempt)
    if last is not None:
        raise last
    raise RuntimeError("chunk analysis failed with no recorded exception")


class ChunkCacheRecord(BaseModel):
    """One cached per-chunk model answer — the Phase-1 resume unit.

    Phase-1 tokens are spent per CHUNK, not per strip, so a single failing
    chunk used to throw away every token already paid for (plans were only
    cached on FULL success). Each successful chunk is written here so a
    re-run reads it back and only re-sends the chunks that actually failed.
    """

    chunk_height: int
    context_hash: str
    image_hash: str
    entries: list[PanelPlanEntry]
    characters: list[str] = Field(default_factory=list)


def _read_chunk_cache(
        path: Path, *, chunk_height: int, context_hash: str,
        image_hash: str) -> tuple[list[PanelPlanEntry], list[str]] | None:
    """Validated per-chunk cache read.

    A corrupt or stale record is never fatal: it returns None so the chunk is
    re-analyzed (a truncated cache file used to be able to kill every later
    run of the strip — the same class of bug as the plan cache).
    """
    if not path.is_file():
        return None
    try:
        rec = ChunkCacheRecord.model_validate_json(path.read_text("utf-8"))
    except Exception as exc:  # noqa: BLE001 - corrupt cache must never be fatal
        log.warning("ignoring unreadable Phase-1 chunk cache %s: %s",
                    path.name, exc)
        return None
    if (rec.chunk_height != chunk_height
            or rec.context_hash != context_hash
            or rec.image_hash != image_hash):
        log.debug("Phase-1 chunk cache %s does not match this chunk; "
                  "re-analyzing", path.name)
        return None
    return rec.entries, rec.characters


def _write_chunk_cache(path: Path, record: ChunkCacheRecord) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(path, record.model_dump_json(indent=2) + "\n")
    except OSError as exc:
        log.warning("could not write Phase-1 chunk cache %s: %s", path, exc)


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
        "model": getattr(backend, "model", ""),
        "prompt_version": PROMPT_VERSION,
    }
    cfg_hash = _config_hash(cfg)

    cache_root = Path(cache_dir) if cache_dir is not None else None
    cache_path: Path | None = None
    chunk_cache_root: Path | None = None
    if cache_root is not None:
        cache_path = cache_root / f"plan_{input_hash[:16]}_{cfg_hash[:16]}.json"
        chunk_cache_root = cache_root / "chunks"
        if not force and cache_path.is_file():
            # A truncated / hand-corrupted cache file must never kill every
            # later run of this strip: it is discarded and the strip is
            # re-analyzed instead.
            cached_plan: PanelPlan | None = None
            try:
                cached_plan = PanelPlan.model_validate_json(
                    cache_path.read_text("utf-8"))
            except Exception as exc:  # noqa: BLE001 - corrupt cache is not fatal
                log.warning(
                    "ignoring unreadable Phase-1 cache %s (%s); re-analyzing",
                    cache_path.name, exc)
            if cached_plan is not None:
                if (cached_plan.input_hash == input_hash
                        and cached_plan.config_hash == cfg_hash):
                    log.info("phase-1 cache hit: %s", cache_path)
                    return cached_plan, True
                log.warning(
                    "Phase-1 cache %s was produced from different inputs "
                    "(input_hash/config_hash mismatch); re-analyzing",
                    cache_path.name)

    chunk_dir_p = Path(chunk_dir) if chunk_dir is not None else None
    with Image.open(path) as img:
        img.load()
        width, height = img.size
        chunks = make_chunks(img, chunk_height=chunk_height, overlap=overlap)
        log.info("strip loaded file=%s size=%dx%d backend=%s model=%s chunks=%d",
                 path.name, width, height, cfg["backend"], cfg["model"],
                 len(chunks))
        results: list[list[PanelPlanEntry]] = []
        bases: list[int] = []
        prev_entries: list[PanelPlanEntry] = []
        characters: list[str] = []
        if _HAS_RICH and not EMBEDDED_MODE:
            progress_ctx = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
            )
            progress_ctx.start()
            task_id = progress_ctx.add_task(
                f"[cyan]Analyzing {len(chunks)} strip chunks via LLM...",
                total=len(chunks),
            )
        else:
            progress_ctx = None
            task_id = None
        try:
            # Speculative prefetch: submit up to CHUNK_CONCURRENCY chunks at
            # once. Each is given the freshest context known at submit time
            # (the most recent chunk already folded), so context lags by up
            # to the pool depth — panel boundaries come from each chunk's own
            # image, so this only slightly relaxes narration continuity.
            pool = ThreadPoolExecutor(max_workers=CHUNK_CONCURRENCY,
                                      thread_name_prefix="phase1-chunk")
            inflight: list[tuple[int, int, Path | None, Future, str]] = []
            collected: dict[int, tuple[int, list[PanelPlanEntry], list[str]]] = {}

            def _submit(idx: int, base: int, chunk: Image.Image,
                        context: str) -> str:
                context_hash = _config_hash({"context": context})
                image_hash = hashlib.sha256(chunk.tobytes()).hexdigest()
                chunk_cache_path: Path | None = None
                if chunk_cache_root is not None:
                    chunk_cache_path = chunk_cache_root / (
                        f"chunk_{input_hash[:8]}_{cfg_hash[:8]}_{idx:04d}"
                        f"_{context_hash[:8]}_{image_hash[:8]}.json")
                cached_chunk = (
                    None if (force or chunk_cache_path is None)
                    else _read_chunk_cache(
                        chunk_cache_path, chunk_height=chunk_height,
                        context_hash=context_hash, image_hash=image_hash))
                if cached_chunk is not None:
                    # Resume: this chunk already succeeded in an earlier run.
                    entries, new_chars = cached_chunk
                    log.info("phase-1 chunk %d/%d cache hit panels=%d",
                             idx + 1, len(chunks), len(entries))
                    collected[idx] = (base, entries, new_chars)
                    return context_hash
                fut = pool.submit(_call_with_retry, backend, chunk,
                                  attempts=attempts, previous_context=context)
                inflight.append((idx, base, chunk_cache_path, fut, context_hash))
                return context_hash

            def _fold(idx: int, base: int, entries: list[PanelPlanEntry],
                      new_chars: list[str]) -> None:
                nonlocal prev_entries
                collected[idx] = (base, entries, new_chars)
                for name in new_chars:
                    if name not in characters:
                        characters.append(name)
                prev_entries = entries
                if progress_ctx is not None and task_id is not None:
                    progress_ctx.update(task_id, advance=1)

            def _drain_one() -> None:
                """Wait for the oldest in-flight chunk and fold its result."""
                idx, base, chunk_cache_path, fut, context_hash = inflight.pop(0)
                try:
                    entries, new_chars = fut.result()
                except Exception as exc:
                    done = len(collected)
                    hint = (""
                            if not (chunk_cache_root is not None and done)
                            else f"; {done} earlier chunk(s) succeeded and "
                                 "are cached — re-run (without --force) to "
                                 "resume and re-spend tokens only on the "
                                 "failed chunk(s)")
                    raise VisionAnalysisError(
                        f"chunk {idx} (absolute y0={base}) failed after "
                        f"{attempts} attempt(s): {exc}{hint}") from exc
                if chunk_cache_path is not None:
                    _write_chunk_cache(chunk_cache_path, ChunkCacheRecord(
                        chunk_height=chunk_height,
                        context_hash=context_hash,
                        image_hash=hashlib.sha256(
                            chunks[idx][1].tobytes()).hexdigest(),
                        entries=entries, characters=new_chars))
                log.info("chunk %d/%d panels=%d new_chars=%s",
                         idx + 1, len(chunks), entries, new_chars)
                _fold(idx, base, entries, new_chars)

            for idx, (base, chunk) in enumerate(chunks):
                if chunk_dir_p is not None:
                    chunk_dir_p.mkdir(parents=True, exist_ok=True)
                    chunk.save(chunk_dir_p / f"chunk_{idx:02d}.png", "PNG")
                context = build_context(prev_entries, characters)
                log.debug("chunk %d/%d base_y=%d size=%dx%d context_len=%d",
                           idx + 1, len(chunks), base, chunk.width,
                           chunk.height, len(context))
                _submit(idx, base, chunk, context)
                # Keep the pool saturated but bounded: fold the oldest
                # result before submitting the next chunk.
                while len(inflight) >= CHUNK_CONCURRENCY:
                    _drain_one()
            while inflight:
                _drain_one()
            pool.shutdown(wait=False)
            # Emit in chunk order (stitch_chunk_results is order-sensitive).
            for idx in sorted(collected):
                base, entries, _chars = collected[idx]
                results.append(entries)
                bases.append(base)
        finally:
            if progress_ctx is not None:
                progress_ctx.stop()

    stitched = stitch_chunk_results(results, bases, height)
    if not stitched:
        raise VisionAnalysisError("the model returned no panels for this strip")
    last_panel_bottom = max(e.y_end for e in stitched)
    if last_panel_bottom < height * 0.95:
        log.warning("phase-1 coverage check: last panel ends at y=%d, "
                    "only %.0f%% of strip height %d covered — strip tail "
                    "may be missing panels (truncated model output)",
                    last_panel_bottom, last_panel_bottom / height * 100, height)
    plan = PanelPlan(
        source=path.name, width=width, height=height,
        model=getattr(backend, "name", type(backend).__name__),
        config_hash=cfg_hash, input_hash=input_hash, entries=stitched,
        characters=characters)
    log.info("stitched panels=%d characters=%s", len(stitched), characters)
    if cache_path is not None and cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
        write_atomic(cache_path, plan.model_dump_json(indent=2) + "\n")
    return plan, False

class VisionBackend(Protocol):
    """Pluggable vision backend for Phase 1.

    The orchestrator `analyze_strip()` implements the requested
    `analyze_strip(image_chunks) -> PanelPlan` interface; each backend
    implements the per-image primitive `analyze_chunk(image,
    previous_context="") -> (entries, characters)` in CHUNK-PIXEL
    coordinates. `characters` is the list of character names the model
    positively identified in this chunk (for cross-chunk continuity).
    `retry_feedback` (optional) carries the parse/validation error from
    the previous attempt so backends can append it to the prompt — with
    temperature=0 an identical prompt would repeat the same bad output.
    Backends may also fill `usage_log` / `last_usage` (best-effort token
    counts) — the smoke test reads them if present.
    """
    name: str

    def analyze_chunk(
        self,
        image: Image.Image,
        previous_context: str = "",
        retry_feedback: str = "",
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
                      previous_context: str = "",
                      retry_feedback: str = ""
                      ) -> tuple[list[PanelPlanEntry], list[str]]:
        self.calls += 1
        log.debug("fixture backend call=%d image=%dx%d feedback=%r",
                  self.calls, image.width, image.height,
                  bool(retry_feedback))
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
    standard OpenAI-compatible usage objects (verified live against the
    Agnes gateway, which returns OpenAI Usage shape)."""
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


def _chunk_prompt(height: int, width: int | None = None, previous_context: str = "",
                  retry_feedback: str = "") -> str:
    """Strict per-chunk dissection instructions (the prompt template).

    Coordinates are NORMALIZED: y_start/y_end and bubble-box Y corners are
    integers in 0..1000 proportional to THIS image's height (top = 0,
    bottom = 1000); bubble-box X corners are integers in 0..1000 proportional
    to THIS image's width (left = 0, right = 1000). Never raw pixels.
    Optionally includes compact narrative context from earlier chunks so
    narration stays coherent across the strip.
    `retry_feedback` carries the parse/validation error from the previous
    attempt so the model can correct its output format.
    """
    context_block = ""
    if previous_context.strip():
        context_block = (
            "\nPREVIOUS CONTEXT (earlier chunks of the same strip):\n"
            f"{previous_context.strip()}\n"
            "Use it ONLY for continuity of narration and character identity. "
            "Do NOT repeat, re-narrate, or borrow panels from it.\n"
        )
    feedback_block = ""
    if retry_feedback.strip():
        feedback_block = (
            "\nYOUR PREVIOUS ATTEMPT FAILED with this error:\n"
            f"{retry_feedback.strip()[:500]}\n"
            "Return STRICT JSON matching the exact shape below. Fix the "
            "reported problem and output nothing else.\n"
        )
    return (
        "OUTPUT CONTRACT: respond with ONE JSON object ONLY. "
        "No prose, no markdown, no code fences, no explanations. "
        "Start with { and end with }. If you cannot comply, return "
        '{"panels":[],"characters":[]}.\n\n'
        "You are dissecting a vertical manhwa (webtoon) strip image into its "
        "logical panels for a narrated recap video. The image you are viewing "
        f"is a SLICE of a taller strip and is {height} pixels tall"
        + (f" and {width} pixels wide" if width else "")
        + ".\n\n"
        "COORDINATES ARE NORMALIZED 0..1000: y_start/y_end and every bubble-box "
        "Y corner are integers in [0, 1000] measured PROPORTIONALLY to this "
        "image's HEIGHT (top = 0, bottom = 1000). Bubble-box X corners are "
        "integers in [0, 1000] measured PROPORTIONALLY to this image's WIDTH "
        "(left = 0, right = 1000). NEVER raw pixels. For example, a panel "
        "occupying the top quarter of the image is y_start=0, y_end=250; a "
        "bubble centered horizontally at the vertical middle is "
        "bubble_boxes: [[420, 480, 580, 540]] (x0=420, y0=480, x1=580, y1=540)."
        f"{context_block}{feedback_block}\n\n"
        "Return STRICT JSON only. Every entry is exactly one logical panel. "
        "The JSON must match this exact shape:\n\n"
        '{"panels": [{"panel_index": 1, "y_start": 120, "y_end": 520, '
        '"narration": "...", "dialogue": "...", "panel_type": "single", '
        '"confidence": 0.95, "bubble_boxes": [[x0, y0, x1, y1], ...]}], '
        '"characters": ["Name"]}\n\n'
        "Rules:\n"
        "- panel_index starts at 1 and increases strictly top to bottom.\n"
        "- y_start < y_end and both are integers inside [0, 1000].\n"
        "- narration describes ONLY what is visible in that panel: characters, "
        "action, setting, mood. Never invent events that are not shown.\n"
        "- Keep narration SHORT and SIMPLE: 1 sentence max, plain language, "
        "no flowery description. This is for a fast-paced recap.\n"
        "- dialogue lists speech-bubble and SFX text verbatim, or an empty "
        "string.\n"
        "- panel_type is exactly one of: single, tall_scenic, "
        "transition_gutter, multi_sub_panel.\n"
        "- confidence (0..1) estimates how reliable the boundary estimate is.\n"
        "- bubble_boxes: normalized boxes around each speech bubble "
        "(x0, y0, x1, y1 as integers in 0..1000 proportional to width/height), "
        "or an empty list.\n"
        "- characters: names positively identifiable in THIS image, or an "
        "empty list. Never invent names.\n"
    )

class AgnesVisionBackend:
    """OpenAI-compatible vision backend: Agnes 2.5 Flash primary + 2.0 Flash
    fallback.

    Default endpoint https://apihub.agnes-ai.com/v1 (AGNES_BASE_URL override),
    key from adapters.ai_models.api_key_from_env: explicit param, then
    manual webapp settings.json, then AGNES_API_KEY / AGNES_API_KEYS.

    The SAME prompt + SAME chunk image is sent to both models; the input /
    output contract (parse_entries_from_json) is identical, so task
    compatibility is preserved. Empty / invalid-JSON / image-rejection /
    timeout / HTTP errors from the primary all trigger the fallback retry
    via the shared adapters.ai_models wrapper. If both fail,
    VisionAnalysisError surfaces both causes (never fabricated panels).

    AI scope: semantic panel understanding only. Physical crop coordinates
    remain advisory here — guided_cutter snaps every boundary to
    deterministically detected gutters and blank_detector stays pure CV.
    """

    name = "agnes"

    def __init__(self, model: str | None = None,
                 api_key: str | None = None,
                 base_url: str | None = None,
                 timeout: int = 120,
                 primary_model: str | None = None,
                 fallback_model: str | None = None,
                 max_tokens: int = 4096,
                 request_fn: object = None) -> None:
        from adapters import ai_models as _ai
        self.primary_model = _ai.resolve_model_id(
            primary_model or model or _ai.PRIMARY_MODEL)
        self.fallback_model = _ai.resolve_model_id(
            fallback_model or _ai.FALLBACK_MODEL)
        if not self.primary_model.strip():
            raise ValueError("primary model id is required")
        # Cache key must distinguish the model pair (plus prompt/image,
        # which analyze_strip already folds in via input/config hashes).
        self.model = f"{self.primary_model}=>{self.fallback_model}"
        self._explicit_key = api_key.strip() if api_key and api_key.strip() else None
        self._api_key = self._explicit_key or _ai.api_key_from_env()
        if not self._api_key and request_fn is None:
            raise RuntimeError(
                "AGNES_API_KEY is not set; set it in .env or pass api_key "
                "(use --backend none for offline mode)")
        self._base_url = (base_url or _ai.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._request_fn = request_fn
        self.usage_log: list[dict] = []
        self.last_usage: dict | None = None
        self.last_model_used: str | None = None
        self.fallback_used = False
        self.last_primary_error: Exception | None = None

    def _request_once(self, model_id: str, prompt: str, b64: str,
                      image: Image.Image, api_key: str | None = None):
        """Single raw request for `model_id`. Raises on any failure,
        including empty or non-JSON responses (callers treat as fallback
        triggers, never as usable output)."""
        if self._request_fn is not None:
            raw_text = self._request_fn(model_id, prompt, b64, image)  # type: ignore[operator]
            if raw_text is None or not str(raw_text).strip():
                raise ValueError(f"model {model_id} returned an empty response")
            return raw_text, None
        import openai  # optional dependency, imported lazily

        client = openai.OpenAI(api_key=api_key or self._api_key,
                               base_url=self._base_url,
                               timeout=self.timeout)
        resp = client.chat.completions.create(
            model=model_id,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64}"}}]},
            ],
        )
        raw_text = resp.choices[0].message.content
        if raw_text is None or not raw_text.strip():
            raise ValueError(f"model {model_id} returned an empty response")
        # Truncation check BEFORE parsing: a "length" answer is cut mid-JSON
        # and would otherwise surface as a misleading parse failure. It is
        # raised as a response-quality error, so the fallback model
        # gets its own chance at the same chunk.
        raise_if_truncated(_openai_finish_reason(resp), backend="agnes",
                           model=model_id, max_tokens=self.max_tokens,
                           image_size=image.size)
        return raw_text, resp

    def _analyze_with(self, model_id: str, image: Image.Image,
                      prompt: str, b64: str, api_key: str | None = None):
        raw_text, resp = self._request_once(model_id, prompt, b64, image,
                                            api_key=api_key)
        if resp is not None:
            self.last_usage = _capture_usage(resp)
            self.usage_log.append(self.last_usage or {})
        # Raises ValueError on invalid JSON / schema violations -> fallback.
        entries, characters = parse_entries_from_json(
            raw_text, image.size[1], chunk_width=image.size[0])
        if not entries:
            # Valid JSON but zero panels: a legitimate blank stretch (see
            # EmptyChunkResult). Raise so the fallback model still gets a
            # chance; if it is ALSO empty, analyze_chunk accepts the chunk.
            raise EmptyChunkResult(f"model {model_id} returned no panels")
        return entries, characters

    def analyze_chunk(self, image: Image.Image,
                      previous_context: str = "",
                      retry_feedback: str = ""
                      ) -> tuple[list[PanelPlanEntry], list[str]]:
        import base64

        from adapters import ai_models as _ai

        # Always send the ACTUAL chunk image; never text-only. Both models
        # share this OpenAI-compatible image_url shape.
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        prompt = _chunk_prompt(image.size[1], image.size[0], previous_context,
                               retry_feedback=retry_feedback)
        operation = f"agnes-analyze-chunk:{image.size[0]}x{image.size[1]}"
        # Key pool: rotate across AGNES_API_KEY(S) so one rate-limited key
        # degrades to the next instead of failing the chunk. The start key
        # round-robins per chunk so parallel chunks spread load (3 keys x
        # 20 RPM each). Key values never appear in logs (index only).
        keys = _ai.api_key_pool(self._explicit_key)
        if not keys and self._request_fn is None:
            raise VisionAnalysisError(
                "agnes vision failed: no API key available "
                "(set AGNES_API_KEY or use --backend none)")
        if not keys:
            keys = [""]
        start = _ai.pool_start_index(len(keys))
        wait_s, rounds = _ai._rate_wait_config()
        last_exc: _ai.AIFallbackError | None = None
        for rnd in range(rounds + 1):
            for attempt in range(len(keys)):
                key = keys[(start + attempt) % len(keys)]
                try:
                    outcome = _ai.call_ai_with_fallback(
                        operation,
                        lambda key=key: self._analyze_with(
                            self.primary_model, image, prompt, b64,
                            api_key=key or None),
                        lambda key=key: self._analyze_with(
                            self.fallback_model, image, prompt, b64,
                            api_key=key or None),
                        primary_model=self.primary_model,
                        fallback_model=self.fallback_model)
                except _ai.AIFallbackError as exc:
                    # Both models returned VALID but EMPTY panel lists: a
                    # legitimately blank chunk (white space between scenes),
                    # not a failure. Accept it — the whole-strip coverage
                    # check and the fallback ratio decide downstream whether
                    # the plan is usable.
                    if (isinstance(exc.primary_error, EmptyChunkResult)
                            and isinstance(exc.fallback_error, EmptyChunkResult)):
                        return [], []
                    last_exc = exc
                    if _ai.is_rate_limit_error(exc):
                        if attempt + 1 < len(keys):
                            log.warning(
                                "agnes rate-limit on key %d/%d; rotating to "
                                "next key", attempt + 1, len(keys))
                            continue
                        break  # whole pool throttled: wait below, or fail
                    raise VisionAnalysisError(
                        f"agnes vision failed (primary {self.primary_model}: "
                        f"{exc.primary_error}; fallback {self.fallback_model}: "
                        f"{exc.fallback_error})") from exc
                self.last_model_used = outcome.model_used
                self.fallback_used = outcome.fallback_used
                self.last_primary_error = outcome.primary_error
                return outcome.result
            # Whole pool throttled this round: Agnes free tier recovers by
            # waiting ("pause a few minutes and retry"), so sleep with linear
            # backoff instead of failing the chunk — unless out of rounds.
            assert last_exc is not None  # loop always runs >= 1 key
            if rnd >= rounds:
                break
            delay = wait_s * (rnd + 1)
            log.warning("agnes pool throttled (all %d key(s)); waiting %.0fs "
                        "(round %d/%d, %s)", len(keys), delay, rnd + 1,
                        rounds, operation)
            time.sleep(delay)
        assert last_exc is not None
        raise VisionAnalysisError(
            f"agnes vision failed on all {len(keys)} key(s): {last_exc}") \
            from last_exc



