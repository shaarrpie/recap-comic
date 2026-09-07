# adapters/schemas.py
"""Artifact schemas for the manhwa-recap IR pipeline (pydantic v2).

Every artifact records: schema_version, generator, config_hash (sha256 of the
stage's canonical config dict) and input_hashes (sha256 per input artifact).
Cache rule: a stage is skipped iff its output exists and all recorded hashes
match. --force bypasses the check.
Coordinate system: pixel coordinates, origin top-left of the page image.
ID scheme: panel id = "<page:03d>.<index:02d>" (e.g. "003.01"), page = 1-based
position in the sorted page list, index = 1-based panel position within the
page in the configured reading order.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

SCHEMA_VERSION = 1
PanelId = str


class Meta(BaseModel):
    schema_version: int
    generator: str  # "<module>.<adapter>"
    config_hash: str
    input_hashes: dict[str, str]


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class Panel(BaseModel):
    id: PanelId
    page: int
    index: int
    bbox: BBox
    source_image: str  # relative path inside pages/


class Bubble(BaseModel):  # optional metadata
    id: str
    panel_id: PanelId
    bbox: BBox
    kind: Literal["dialogue", "thought", "narration", "sfx", "unknown"] = "unknown"


class PanelsArtifact(BaseModel):
    meta: Meta
    reading_order: Literal["top_to_bottom", "right_to_left_rows"]
    pages: list[str]
    panels: list[Panel]
    bubbles: list[Bubble] = []


class OcrRegion(BaseModel):
    id: str
    panel_id: PanelId | None
    page: int
    bbox: BBox
    text: str
    confidence: float | None  # 0..100 as reported by the engine
    kind: Literal["dialogue", "narration", "sfx", "unknown"] = "unknown"


class OcrArtifact(BaseModel):
    meta: Meta
    backend: str
    regions: list[OcrRegion]


class NarrationEntry(BaseModel):
    id: PanelId  # == panel id: one narration entry per panel
    panel_id: PanelId
    order: int
    speaker: str | None = None
    text: str = ""
    quotes: list[str] = []  # verbatim dialogue, grounding-checked


class NarrationArtifact(BaseModel):
    meta: Meta
    mode: Literal["narrator", "characters", "verbatim"]
    entries: list[NarrationEntry]
    ungrounded_quotes: list[str] = []


class AudioEntry(BaseModel):
    entry_id: PanelId
    path: str
    duration_seconds: float  # measured (ffprobe or wave header), never estimated
    words: list[dict] = []  # {start, end, text}, seconds relative to clip start


class AudioArtifact(BaseModel):
    meta: Meta
    voice: str
    entries: list[AudioEntry]


class PanSpec(BaseModel):
    kind: Literal["pan_down", "pan_right", "pan_left", "pan_up", "zoom_in", "zoom_out", "static"]
    scaled_w: int  # panel scaled to this size before cropping to 1080x1920
    scaled_h: int
    travel_px: int  # total pan distance along the pan axis (0 when static)


class TimelineEntry(BaseModel):
    panel_id: PanelId
    order: int
    source_image: str
    bbox: BBox
    start_seconds: float
    duration_seconds: float
    audio_path: str | None = None  # None => silent, min display duration
    pan: PanSpec


class TimelineArtifact(BaseModel):
    meta: Meta
    width: int = 1080
    height: int = 1920
    fps: int = 30
    gap_seconds: float  # default 0.35 (see report)
    min_display_seconds: float  # default 2.0 (see report)
    entries: list[TimelineEntry]
