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
from pathlib import Path

import strip_analyzer as sa


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


def make_script(plan: sa.PanelPlan, style: str = "recap") -> str:
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


def narrate_plan(plan_path: Path, out_path: Path, *,
                 style: str = "recap") -> str:
    """Read a plan JSON, write the script, return the script text."""
    plan = sa.PanelPlan.model_validate_json(plan_path.read_text("utf-8"))
    script = make_script(plan, style=style)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script, encoding="utf-8")
    # Also write a tiny sidecar with per-panel offsets for TTS chunking.
    entries = sorted(plan.entries, key=lambda e: e.panel_index)
    index = [{"panel_id": e.panel_index,
              "narration": e.narration,
              "dialogue": e.dialogue} for e in entries]
    index_path = out_path.with_suffix(".index.json")
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return script
