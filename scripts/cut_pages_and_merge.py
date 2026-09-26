#!/usr/bin/env python
"""Cut oversized multi-page strips per page, then merge into one panels.json.

A stitched webtoon strip taller than PIL's decompression-bomb limit (~179 MP)
and guided_cutter's own 80 MP safety cap cannot be cut in one pass. This
script processes each source PAGE image separately (each is well under every
limit), runs the deterministic gutter-detector plan per page, cuts it, and
merges everything into a single CutArtifact whose y_start/y_end are absolute
coordinates in the virtual stitched strip (same order, same widths).

Usage:
    python scripts/cut_pages_and_merge.py PAGES_DIR OUT_DIR [--ext webp]

PAGES_DIR must contain the primary page images (duplicates like *_1.webp
are skipped, and so are non-portrait pages -- the scanlation credits/title
card every chapter opens with; see --keep-landscape-pages). OUT_DIR receives
panel_NNN.png, panels.json, per-page plan/pieces JSON under pieces/, and a
copy of each page's cut outputs.

The merged panels.json is a valid CutArtifact (guided_cutter.CutArtifact):
- source/width/height describe the virtual stitched strip
- y_start/y_end are absolute stitched coordinates
- strip_width is set only when the page's width differs from the artifact
  width (legacy full-res crop support)
- panel PNGs are the cutter's normalized 390x[760,800] outputs, so Phase 3
  (recap_video) uses each panel's output_width/output_height for pan math
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # noqa: E402

import strip_analyzer as sa  # noqa: E402
from guided_cutter import CutArtifact, CutPanel, CutterConfig, guided_cut  # noqa: E402
from guided_pipeline import fallback_plan_from_gutter_detector  # noqa: E402

log = logging.getLogger("cut_pages_and_merge")

MIN_PANEL_PX = 5  # below this a piece is dust, not a panel


def natural_key(name: str) -> list[str | int]:
    import re
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", name)]


def collect_pages(pages_dir: Path, ext: str, *,
                  skip_names: set[str] = frozenset(),
                  drop_landscape: bool = True,
                  ) -> tuple[list[Path], list[str]]:
    """Primary pages only: skip *_1.* duplicates (same page re-downloaded).

    Returns (kept pages in reading order, human-readable skip notes).

    Scanlation groups ship a non-story credits/title card as the first page of
    every chapter (group logo, TL/PR/CL/RD/TS/QC names, the chapter number).
    It is not story art, and cutting it produces the same fake "panel" at the
    top of every chapter of the recap. Webtoon pages are always far taller
    than wide, so any page that is not strictly portrait is dropped; pass
    --keep-landscape-pages to opt out (e.g. a chapter that legitimately opens
    on a landscape splash).
    """
    pages = [p for p in pages_dir.iterdir()
             if p.is_file()
             and p.suffix.lower() == f".{ext.lstrip('.')}"
             and not p.stem.endswith("_1")]
    pages.sort(key=lambda p: natural_key(p.name))
    if not pages:
        raise SystemExit(f"no *.{ext} pages found in {pages_dir}")
    kept: list[Path] = []
    skipped: list[str] = []
    for page in pages:
        if page.name in skip_names:
            skipped.append(f"{page.name} (explicit --skip-page)")
            continue
        if drop_landscape:
            with Image.open(page) as img:
                w, h = img.size
            if w >= h:
                skipped.append(f"{page.name} (landscape {w}x{h}: "
                               f"credits/ad card, not story art)")
                continue
        kept.append(page)
    return kept, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pages_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--ext", default="webp")
    ap.add_argument("--max-panel-height", type=int, default=1600)
    ap.add_argument("--variance-threshold", type=float, default=6.0)
    ap.add_argument("--edge-threshold", type=float, default=30.0)
    ap.add_argument("--tolerance", type=int, default=80)
    ap.add_argument("--blank-sensitivity", default="conservative",
                    choices=["low", "conservative", "high"])
    ap.add_argument("--no-blank", action="store_true")
    ap.add_argument("--no-filter", action="store_true",
                    help="skip the Phase 2.5 panel-content filter. By default "
                         "(mirroring `guided run --filter`) blank panels are "
                         "removed and text-only / solid-colour card crops "
                         "(speech-bubble fields, title cards) are demoted to "
                         "context_only so they get no frame and no narration.")
    ap.add_argument("--skip-page", action="append", default=[], metavar="NAME",
                    help="exclude this page filename from the cut "
                         "(repeatable), e.g. --skip-page page_001.webp")
    ap.add_argument("--keep-landscape-pages", action="store_true",
                    help="do not auto-drop pages that are not strictly "
                         "portrait (default drops them: they are scanlation "
                         "credits/ad cards, not story art)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    pages, skipped = collect_pages(
        args.pages_dir, args.ext,
        skip_names=set(args.skip_page),
        drop_landscape=not args.keep_landscape_pages)
    for note in skipped:
        log.info("page excluded: %s", note)
    if not pages:
        raise SystemExit(f"every page in {args.pages_dir} was excluded; "
                         f"nothing to cut")
    log.info("pages=%d ext=%s excluded=%d", len(pages), args.ext, len(skipped))

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pieces_dir = out / "pieces"
    if args.force:
        import shutil
        if pieces_dir.exists():
            shutil.rmtree(pieces_dir)
        for p in out.glob("panel_*.png"):
            p.unlink()
    pieces_dir.mkdir(parents=True, exist_ok=True)

    merged_panels: list[CutPanel] = []
    y_offset = 0
    art_w = 0
    total_h = 0
    max_h_seen = 0

    for n, page in enumerate(pages, start=1):
        with Image.open(page) as img:
            w, h = img.size
        art_w = max(art_w, w)
        total_h += h
        log.info("page %02d/%d %s %dx%d (y_offset=%d)",
                 n, len(pages), page.name, w, h, y_offset)

        # per-page deterministic plan (each page is a normal-size strip)
        plan = fallback_plan_from_gutter_detector(
            page,
            variance_threshold=args.variance_threshold,
            edge_threshold=args.edge_threshold,
            max_panel_height=args.max_panel_height)
        log.info("  page plan panels=%d", len(plan.entries))

        # cut the page with the same config `guided run --backend none` uses
        config = CutterConfig(
            tolerance=args.tolerance,
            max_panel_height=args.max_panel_height,
            variance_threshold=args.variance_threshold,
            edge_threshold=args.edge_threshold,
            blank_detection=not args.no_blank,
            blank_preset=args.blank_sensitivity,
        )
        page_out = pieces_dir / f"page_{n:02d}"
        page_out.mkdir(parents=True, exist_ok=True)
        artifact = guided_cut(page, plan, page_out, config=config,
                              force=args.force)
        log.info("  page cut panels=%d -> %s",
                 len(artifact.panels), page_out)

        # keep a copy of the per-page plan for provenance
        (page_out / "plan.json").write_text(
            plan.model_dump_json(indent=2) + "\n", "utf-8")

        # absorb panels
        for p in sorted(artifact.panels, key=lambda p: p.y_start):
            if p.y_end - p.y_start < MIN_PANEL_PX:
                continue
            new_index = len(merged_panels) + 1
            new_id = f"{new_index:03d}"
            new_file = f"panel_{new_index:03d}.png"
            # copy the normalized PNG under its merged name
            src = page_out / p.image_file
            if src.is_file():
                Image.open(src).save(out / new_file, "PNG")
            else:  # skipped PNG during cut; drop the entry entirely
                log.warning("  page %d panel image missing: %s",
                            n, p.image_file)
                continue
            merged_panels.append(CutPanel(
                id=new_id,
                panel_index=new_index,
                y_start=y_offset + p.y_start,
                y_end=y_offset + p.y_end,
                narration=p.narration,
                dialogue=p.dialogue,
                panel_type=p.panel_type,
                confidence=p.confidence,
                image_file=new_file,
                output_width=p.output_width,
                output_height=p.output_height,
                strip_width=w if w != art_w else None,
                split_of=p.split_of,
                merged_with=p.merged_with,
                snap_distances=p.snap_distances,
                snap_measured=p.snap_measured,
                blank_score=p.blank_score,
                blank_flag=p.blank_flag,
                blank_reasons=p.blank_reasons,
            ))
        y_offset += h
        max_h_seen = max(max_h_seen, h)

    if not merged_panels:
        raise SystemExit("no panels were produced across all pages")

    merged = CutArtifact(
        source="stitched:" + "+".join(p.name for p in pages),
        width=art_w,
        height=total_h,
        plan_hash="per-page-gutter-fallback",
        config={
            "mode": "per-page-deterministic",
            "pages": [p.name for p in pages],
            "skipped_pages": skipped,
            "max_panel_height": args.max_panel_height,
            "variance_threshold": args.variance_threshold,
            "edge_threshold": args.edge_threshold,
            "tolerance": args.tolerance,
            "blank_sensitivity": (None if args.no_blank
                                  else args.blank_sensitivity),
        },
        panels=merged_panels,
    )
    sa.write_atomic(out / "panels.json",
                    merged.model_dump_json(indent=2) + "\n")
    blank = sum(1 for p in merged_panels if p.blank_flag == "blank")
    log.info("merged panels=%d blank=%d out=%s",
             len(merged_panels), blank, out / "panels.json")
    print(f"merged {len(merged_panels)} panels "
          f"({blank} blank-flagged) into {out / 'panels.json'}")

    # Phase 2.5: deterministic panel-content filter (blank removal +
    # text-only / solid-colour-card demotion), in place on the merged
    # artifact. Mirrors `guided run --filter` and the webapp build. A filter
    # failure must never lose the cut, so it degrades to an unfiltered
    # panels.json (the safe direction).
    if not args.no_filter:
        try:
            from panel_filter import filter_panels_inplace
            fres = filter_panels_inplace(out)
            log.info("Phase-2.5 panel filter: kept=%d context_only=%d "
                     "removed_blank=%d [%s]",
                     fres["kept"], fres["context_only"],
                     fres["removed_blank"], fres["thresholds"]["method"])
            print(f"filtered: kept={fres['kept']} "
                  f"context_only={fres['context_only']} "
                  f"removed_blank={fres['removed_blank']} "
                  f"[{fres['thresholds']['method']}]")
        except Exception as exc:  # noqa: BLE001 - filter must never kill the cut
            log.warning("panel filter skipped (%s); panels.json left unfiltered",
                        exc)


if __name__ == "__main__":
    main()
