# adapters/panels_guided.py
"""Bridge from Stack A (AI-guided cutter) to Stack B (IR pipeline).

Converts a `CutArtifact` (panels.json written by `guided cut` / `guided run`)
into a `PanelsArtifact` that every downstream Stack B stage understands.

The guided cutter operates on one tall strip, so the output is a single-page
PanelsArtifact: each panel's bbox is `(0, y_start, width, y_end-y_start)`,
and the narration/dialogue/confidence from the CutPanel are carried across
as panel-level metadata (the OCR stage can pick them up).
"""
from __future__ import annotations

from pathlib import Path

from guided_cutter import CutArtifact

from .schemas import SCHEMA_VERSION, BBox, Meta, Panel, PanelsArtifact


def _meta(cfg: dict, input_hashes: dict) -> Meta:
    return Meta(
        schema_version=SCHEMA_VERSION,
        generator="panels_guided",
        config_hash="",
        input_hashes=input_hashes,
    )


def cut_artifact_to_panels(artifact: CutArtifact) -> PanelsArtifact:
    """Convert a CutArtifact into a single-page PanelsArtifact."""
    panels: list[Panel] = []
    for p in artifact.panels:
        h = p.y_end - p.y_start
        panels.append(Panel(
            id=p.id,
            page=1,
            index=p.panel_index,
            bbox=BBox(x=0, y=p.y_start, w=artifact.width, h=h),
            source_image=p.image_file,
        ))
    return PanelsArtifact(
        meta=_meta(
            artifact.config,
            {"panels.json": artifact.plan_hash},
        ),
        reading_order="top_to_bottom",
        pages=[artifact.source],
        panels=panels,
    )


def load_and_convert(panels_json: Path) -> PanelsArtifact:
    """Load panels.json written by `guided cut` and return PanelsArtifact."""
    data = panels_json.read_text("utf-8")
    artifact = CutArtifact.model_validate_json(data)
    return cut_artifact_to_panels(artifact)
