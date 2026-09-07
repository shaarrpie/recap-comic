# adapters/editor.py
"""Editable project state for the recap-comic video editor.

The editor works on top of the existing automated pipeline output.  It does NOT
replace automation; it provides manual overrides for:
  - panel order
  - panel duration
  - visual effect / Ken Burns
  - transitions between panels
  - caption text and timing

Every user mutation is recorded so undo/redo is possible.  Automated values
are preserved alongside user overrides so the user can always reset to the
automated baseline.
"""
from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from adapters.schemas import BBox, PanSpec, TimelineEntry

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
class EditorCaption(BaseModel):
    id: str
    panel_id: str
    text: str = ""
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    automated_text: str = ""
    automated_start_seconds: float = 0.0
    automated_end_seconds: float = 0.0


class EditorTransition(BaseModel):
    from_panel_id: str
    to_panel_id: str
    type: str = "cut"       # cut | fade | crossfade
    duration: float = 0.5


class EditorEffect(BaseModel):
    panel_id: str
    kind: str = "static"    # static | zoom_in | zoom_out | pan_left | pan_right | pan_up | pan_down
    duration: float = 0.0


class UndoAction(BaseModel):
    type: str               # reorder | duration | effect | caption | transition | remove | add
    before: dict[str, Any]
    after: dict[str, Any]
    ts: float = field(default_factory=time.time)


class EditorProject(BaseModel):
    version: int = 1
    session: str = ""
    project: dict[str, Any] = {"width": 1080, "height": 1920, "fps": 30}
    original_timeline: list[dict[str, Any]] = []
    edited_timeline: list[dict[str, Any]] = []
    captions: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    history_index: int = -1
    needs_render: bool = True
    last_rendered_at: float | None = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _panel_id_from_image(image_file: str) -> str:
    stem = Path(image_file).stem
    if stem.startswith("panel_"):
        return stem[len("panel_"):]
    return stem


def timeline_from_cut(artifact, panels_dir: Path, cfg) -> list[dict[str, Any]]:
    """Build an initial edited_timeline from a CutArtifact."""
    from recap_video import build_narration, synthesize_audio, build_timeline
    from adapters.schemas import TimelineArtifact

    panels_hash = _sha256_text(json.dumps(artifact.model_dump(), sort_keys=True))
    narration = build_narration(artifact, cfg, panels_hash=panels_hash)
    audio = synthesize_audio(narration, panels_dir, cfg, force=False)
    tl = build_timeline(artifact, panels_dir, narration, audio, panels_dir, cfg,
                        panels_hash=panels_hash)
    return [e.model_dump() for e in tl.entries]


def captions_from_timeline(tl_entries, narration, audio) -> list[dict[str, Any]]:
    """Build initial captions from timeline + narration + audio."""
    from recap_video import _cues_for_entry, srt_time
    by_text = {n.id: n.text for n in narration.entries}
    by_audio = {a.entry_id: a for a in audio.entries}
    captions = []
    cid = 0
    for e in tl_entries:
        pid = e["panel_id"]
        text = by_text.get(pid, "")
        audio_entry = by_audio.get(pid)
        cues = _cues_for_entry(
            TimelineEntry(**e), text, audio_entry
            if audio_entry else None
        )
        for start, end, txt in cues:
            cid += 1
            captions.append({
                "id": f"cap_{cid:03d}",
                "panel_id": pid,
                "text": " ".join(txt.split()),
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "automated_text": " ".join(txt.split()),
                "automated_start_seconds": round(start, 3),
                "automated_end_seconds": round(end, 3),
            })
    return captions


def transitions_from_timeline(tl_entries) -> list[dict[str, Any]]:
    trans = []
    for i in range(len(tl_entries) - 1):
        trans.append({
            "from_panel_id": tl_entries[i]["panel_id"],
            "to_panel_id": tl_entries[i + 1]["panel_id"],
            "type": "cut",
            "duration": 0.5,
        })
    return trans


def effects_from_timeline(tl_entries) -> list[dict[str, Any]]:
    return [{
        "panel_id": e["panel_id"],
        "kind": e.get("pan", {}).get("kind", "static"),
        "duration": e.get("duration_seconds", 0.0),
    } for e in tl_entries]


def _sha256_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Editor logic
# --------------------------------------------------------------------------- #
class Editor:
    def __init__(self, project: EditorProject):
        self.project = project

    @classmethod
    def load(cls, path: Path) -> Editor:
        data = json.loads(path.read_text("utf-8"))
        return cls(EditorProject.model_validate(data))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.project.model_dump_json(indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)

    def _record(self, action_type: str, before: dict, after: dict) -> None:
        act = UndoAction(type=action_type, before=before, after=after).model_dump()
        # truncate any forward history
        self.project.history = self.project.history[: self.project.history_index + 1]
        self.project.history.append(act)
        self.project.history_index = len(self.project.history) - 1
        # keep history bounded
        if len(self.project.history) > 200:
            self.project.history = self.project.history[-200:]
            self.project.history_index = len(self.project.history) - 1
        self.project.needs_render = True

    def undo(self) -> bool:
        if self.project.history_index < 0:
            return False
        act = self.project.history[self.project.history_index]
        self._apply_action(act["type"], act["before"])
        self.project.history_index -= 1
        self.project.needs_render = True
        return True

    def redo(self) -> bool:
        if self.project.history_index >= len(self.project.history) - 1:
            return False
        self.project.history_index += 1
        act = self.project.history[self.project.history_index]
        self._apply_action(act["type"], act["after"])
        self.project.needs_render = True
        return True

    def _apply_action(self, action_type: str, payload: dict) -> None:
        if action_type == "reorder":
            self.project.edited_timeline = payload["timeline"]
            self._recalc_starts()
            self._rebuild_transitions()
        elif action_type == "duration":
            entry = next((e for e in self.project.edited_timeline if e["panel_id"] == payload["panel_id"]), None)
            if entry:
                entry["duration_seconds"] = payload["duration_seconds"]
                self._recalc_starts()
                self._recalc_caption_timings_for_panel(payload["panel_id"])
        elif action_type == "effect":
            eff = next((e for e in self.project.effects if e["panel_id"] == payload["panel_id"]), None)
            if eff:
                eff.update(payload)
        elif action_type == "caption":
            cap = next((c for c in self.project.captions if c["id"] == payload["id"]), None)
            if cap:
                cap.update(payload)
        elif action_type == "transition":
            tr = next((t for t in self.project.transitions
                       if t["from_panel_id"] == payload["from_panel_id"]
                       and t["to_panel_id"] == payload["to_panel_id"]), None)
            if tr:
                tr.update(payload)
        elif action_type in ("remove", "add"):
            self.project.edited_timeline = payload["timeline"]
            self._recalc_starts()
            self._rebuild_transitions()

    def _recalc_starts(self) -> None:
        t = 0.0
        for e in self.project.edited_timeline:
            e["start_seconds"] = round(t, 3)
            t += e["duration_seconds"]

    def _rebuild_transitions(self) -> None:
        existing = {(t["from_panel_id"], t["to_panel_id"]): t
                    for t in self.project.transitions}
        new_trans = []
        for i in range(len(self.project.edited_timeline) - 1):
            a = self.project.edited_timeline[i]["panel_id"]
            b = self.project.edited_timeline[i + 1]["panel_id"]
            key = (a, b)
            if key in existing:
                new_trans.append(existing[key])
            else:
                new_trans.append({
                    "from_panel_id": a,
                    "to_panel_id": b,
                    "type": "cut",
                    "duration": 0.5,
                })
        self.project.transitions = new_trans

    def _recalc_caption_timings_for_panel(self, panel_id: str) -> None:
        entry = next((e for e in self.project.edited_timeline if e["panel_id"] == panel_id), None)
        if not entry:
            return
        start = entry["start_seconds"]
        end = start + entry["duration_seconds"]
        for cap in self.project.captions:
            if cap["panel_id"] == panel_id:
                auto_start = cap["automated_start_seconds"]
                auto_end = cap["automated_end_seconds"]
                auto_dur = max(auto_end - auto_start, 0.01)
                scale = (end - start) / max(entry.get("automated_duration", end - start), 0.01)
                if scale != 1.0:
                    rel_start = auto_start - (entry.get("automated_start_seconds", start) or start)
                    rel_end = auto_end - (entry.get("automated_end_seconds", end) or end)
                    cap["start_seconds"] = round(start + rel_start * scale, 3)
                    cap["end_seconds"] = round(end + rel_end * scale, 3)
                else:
                    cap["start_seconds"] = round(start + (auto_start - entry.get("automated_start_seconds", 0)), 3)
                    cap["end_seconds"] = round(end + (auto_end - entry.get("automated_end_seconds", 0)), 3)

    # -- public mutation API ------------------------------------------------- #
    def reorder_panels(self, new_order: list[str]) -> None:
        before = copy.deepcopy(self.project.edited_timeline)
        by_id = {e["panel_id"]: e for e in self.project.edited_timeline}
        self.project.edited_timeline = [by_id[pid] for pid in new_order if pid in by_id]
        self._recalc_starts()
        self._rebuild_transitions()
        self._record("reorder", {"timeline": before}, {"timeline": copy.deepcopy(self.project.edited_timeline)})

    def set_duration(self, panel_id: str, duration: float) -> None:
        before = next((e["duration_seconds"] for e in self.project.edited_timeline if e["panel_id"] == panel_id), None)
        entry = next((e for e in self.project.edited_timeline if e["panel_id"] == panel_id), None)
        if entry is None or before is None:
            return
        entry["duration_seconds"] = round(max(duration, 0.1), 3)
        self._recalc_starts()
        self._recalc_caption_timings_for_panel(panel_id)
        self._record("duration",
                     {"panel_id": panel_id, "duration_seconds": before},
                     {"panel_id": panel_id, "duration_seconds": entry["duration_seconds"]})

    def set_effect(self, panel_id: str, kind: str, duration: float) -> None:
        before = next((e for e in self.project.effects if e["panel_id"] == panel_id), None)
        payload = {"panel_id": panel_id, "kind": kind, "duration": round(duration, 3)}
        if before is None:
            self.project.effects.append(payload)
        else:
            before.update(payload)
        self._record("effect",
                     {"panel_id": panel_id, "kind": before.get("kind", "static") if before else "static",
                      "duration": before.get("duration", 0.0) if before else 0.0},
                     payload)

    def update_caption(self, caption_id: str, **kwargs) -> None:
        cap = next((c for c in self.project.captions if c["id"] == caption_id), None)
        if cap is None:
            return
        before = {k: cap[k] for k in kwargs if k in cap}
        cap.update({k: v for k, v in kwargs.items() if k in cap})
        self._record("caption", {"id": caption_id, **before}, {"id": caption_id, **kwargs})

    def set_transition(self, from_panel_id: str, to_panel_id: str, type: str, duration: float) -> None:
        before = next((t for t in self.project.transitions
                       if t["from_panel_id"] == from_panel_id and t["to_panel_id"] == to_panel_id), None)
        payload = {"from_panel_id": from_panel_id, "to_panel_id": to_panel_id,
                    "type": type, "duration": round(duration, 3)}
        if before is None:
            self.project.transitions.append(payload)
        else:
            before.update(payload)
        self._record("transition",
                     {"from_panel_id": from_panel_id, "to_panel_id": to_panel_id,
                      "type": before.get("type", "cut") if before else "cut",
                      "duration": before.get("duration", 0.5) if before else 0.5},
                     payload)

    def remove_panel(self, panel_id: str) -> None:
        before = copy.deepcopy(self.project.edited_timeline)
        self.project.edited_timeline = [e for e in self.project.edited_timeline if e["panel_id"] != panel_id]
        self.project.captions = [c for c in self.project.captions if c["panel_id"] != panel_id]
        self.project.effects = [e for e in self.project.effects if e["panel_id"] != panel_id]
        self._recalc_starts()
        self._rebuild_transitions()
        self._record("remove", {"timeline": before}, {"timeline": copy.deepcopy(self.project.edited_timeline)})

    def add_panel(self, panel_id: str, after_panel_id: str | None = None) -> None:
        # find the panel in original timeline
        orig = next((e for e in self.project.original_timeline if e["panel_id"] == panel_id), None)
        if orig is None:
            return
        before = copy.deepcopy(self.project.edited_timeline)
        new_entry = copy.deepcopy(orig)
        new_entry["automated_duration"] = new_entry["duration_seconds"]
        new_entry["automated_start_seconds"] = new_entry["start_seconds"]
        new_entry["automated_end_seconds"] = new_entry["start_seconds"] + new_entry["duration_seconds"]
        new_entry["automated_pan"] = new_entry.get("pan", {})
        if after_panel_id:
            idx = next((i for i, e in enumerate(self.project.edited_timeline) if e["panel_id"] == after_panel_id), -1)
            self.project.edited_timeline.insert(idx + 1, new_entry)
        else:
            self.project.edited_timeline.append(new_entry)
        self._recalc_starts()
        self._rebuild_transitions()
        self._record("add", {"timeline": before}, {"timeline": copy.deepcopy(self.project.edited_timeline)})

    def reset_to_automated(self) -> None:
        self.project.edited_timeline = copy.deepcopy(self.project.original_timeline)
        self._recalc_starts()
        self._rebuild_transitions()
        self.project.history = []
        self.project.history_index = -1
        self.project.needs_render = True
