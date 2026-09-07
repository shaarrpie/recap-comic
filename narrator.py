# narrator.py
"""Recap / literal script generation from a per-panel narration plan.

Two styles:
  recap   — a single flowing recap script (paragraph form), joining the
            per-panel narrations in reading order. Suitable for TTS.
  literal — the per-panel narrations concatenated verbatim, separated by
            blank lines. Useful for subtitles or debugging.

Offline: no API call, no model load. Pure string joining.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import strip_analyzer as sa
from guided_cutter import CutArtifact

log = logging.getLogger(__name__)


def _flow_join(parts: list[str]) -> str:
    """Join narration fragments into one flowing paragraph."""
    cleaned: list[str] = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        # Ensure each fragment ends with sentence punctuation so the recap
        # reads naturally when fed to TTS.
        if s[-1] not in ".!?":
            s += "."
        cleaned.append(s)
    text = " ".join(cleaned)
    # Collapse any accidental double-spaces from fragments that already
    # included trailing whitespace.
    return " ".join(text.split())


def make_script_from_plan(plan: sa.PanelPlan, style: str = "recap") -> str:
    """Produce a single script string from a panel plan.

    style="recap"   -> flowing paragraph (TTS-friendly).
    style="literal" -> verbatim per-panel text, panels separated by blank
                       lines.
    """
    entries = sorted(plan.entries, key=lambda e: e.panel_index)
    parts = [e.narration for e in entries if e.narration.strip()]
    if not parts:
        return ""
    if style == "literal":
        return "\n\n".join(p.strip() for p in parts)
    if style == "recap":
        return _flow_join(parts)
    raise ValueError(f"unknown style {style!r}; expected 'recap' or 'literal'")


make_script = make_script_from_plan


def make_script_from_cut(artifact: CutArtifact, style: str = "recap") -> str:
    """Produce a script string from a CutArtifact (panels.json).

    Uses the post-cut panel order and narrations so the script maps 1:1 to
    the output PNGs.
    """
    panels = sorted(artifact.panels, key=lambda p: p.y_start)
    parts = []
    _prev = None
    for p in panels:
        txt = p.narration.strip()
        if not txt or txt == _prev:
            continue
        parts.append(txt)
        _prev = txt
    if not parts:
        return ""
    if style == "literal":
        return "\n\n".join(p.strip() for p in parts)
    if style == "recap":
        return _flow_join(parts)
    raise ValueError(f"unknown style {style!r}; expected 'recap' or 'literal'")


def _script_from_panels(artifact, style: str = "recap") -> str:
    """Produce a script from a PanelsArtifact (Stack B IR).

    PanelsArtifact carries no narration text in the current schema; this
    path is a placeholder that raises until the OCR/narration stage fills
    it in. Use CutArtifact/PanelPlan for narration-capable inputs.
    """
    raise NotImplementedError(
        "Stack-B PanelsArtifact has no narration text yet; "
        "use CutArtifact or PanelPlan instead")


def narrate_plan(plan_path: Path, out_path: Path, *,
                 style: str = "recap") -> str:
    """Read a plan JSON (PanelPlan or CutArtifact), write the script, return it."""
    raw = plan_path.read_text("utf-8")
    data = json.loads(raw)
    if "panels" in data and isinstance(data["panels"], list) and data["panels"]:
        first = data["panels"][0]
        if "image_file" in first:
            from guided_cutter import CutArtifact
            artifact = CutArtifact.model_validate_json(raw)
            script = make_script_from_cut(artifact, style=style)
            index = [{"panel_id": p.id, "panel_index": p.panel_index,
                      "narration": p.narration, "dialogue": p.dialogue,
                      "image_file": p.image_file}
                     for p in sorted(artifact.panels, key=lambda p: p.y_start)]
            log.info("narrate_plan from CutArtifact panels=%d style=%s",
                     len(artifact.panels), style)
        elif "bbox" in first:
            from adapters.schemas import PanelsArtifact
            artifact = PanelsArtifact.model_validate_json(raw)
            script = _script_from_panels(artifact, style=style)
            index = [{"panel_id": p.id, "panel_index": p.index,
                      "narration": "", "dialogue": ""}
                     for p in artifact.panels]
            log.info("narrate_plan from PanelsArtifact panels=%d style=%s",
                     len(artifact.panels), style)
        else:
            raise ValueError(
                f"unrecognised panels format in {plan_path}: "
                f"expected 'image_file' (CutArtifact) or 'bbox' (PanelsArtifact)")
    elif "entries" in data and isinstance(data.get("entries"), list):
        plan = sa.PanelPlan.model_validate_json(raw)
        script = make_script_from_plan(plan, style=style)
        index = [{"panel_id": e.panel_index,
                  "narration": e.narration, "dialogue": e.dialogue}
                 for e in sorted(plan.entries, key=lambda e: e.panel_index)]
        log.info("narrate_plan from PanelPlan entries=%d style=%s",
                 len(plan.entries), style)
    else:
        raise ValueError(
            f"unrecognised plan format in {plan_path}: expected PanelPlan "
            f"or CutArtifact/PanelsArtifact with panels/entries list")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sa.write_atomic(out_path, script)
    index_path = out_path.with_suffix(".index.json")
    sa.write_atomic(index_path, json.dumps(index, indent=2))
    log.info("narrate_plan wrote script=%d chars index=%d entries",
             len(script), len(index))
    return script
