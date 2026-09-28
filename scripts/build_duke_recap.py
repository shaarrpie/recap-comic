#!/usr/bin/env python
"""Autonomous multi-chapter recap builder for demo_duke (chapters 1..25).

Per chapter: deterministic page cut -> chained story-memory (carry_forward from
the previous chapter) -> AI narration + FULL-COVERAGE recap script (one spoken
line per panel) -> 16:9 video, no color grade, no vignette, no sound effects,
edge TTS en-US-BrianNeural @ +20%. Finally concat every chapter recap into one
continuous recap_full_16x9[_chA-B].mp4.

Fully resumable: each stage is skipped when its output already exists (the
narration stage also re-runs when script.json was written by an older script
contract), so a re-run after a crash continues where it left off.
--force-video re-renders the Phase-3 videos from the existing cut/narration (no
AI calls); --force also re-cuts and re-narrates (expensive: it invalidates the
narration cache). --chapters 1-22 restricts the build to a chapter range.
Best-effort: a chapter that fails is logged and skipped; the build never aborts
the whole run. Offline/no-key: the AI stages fail per-chapter and are reported,
nothing crashes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "demo_duke"
OUT = REPO / "demo_duke_out"
# NOTE: no SFX bank on purpose -- the recap is narrated over the art with the
# narrator's voice only (the user asked for no sound effects). Passing
# --sfx-dir is what enables SFX mixing in Phase 3, so it is simply omitted.
# `python scripts/build_duke_recap.py` puts scripts/ (not the repo root) on
# sys.path, so the in-process `story_context` import below used to fail with
# "No module named 'story_context'" and silently disabled cross-chapter memory
# chaining. The repo root must be importable for the carry-forward step.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PY = str(Path(sys.executable))
VOICE = "en-US-BrianNeural"
RATE = "+20%"
# Look: 720p canvas (faster render), flat colours (no moody grade), no dark
# vignette; panels still float on the blurred fill (the preferred mode).
# --min-silent 0 keeps the timeline speech-driven: any panel that somehow ends
# up with no audio is dropped rather than held as a 1s silent beat (silent
# filler beats = dead air + rushed pans; see the recap pitfall). Full coverage
# (recap_script) already speaks every panel, so this drops nothing real and just
# shaves residual dead air -> shorter video + faster render.
VIDEO_LOOK = ["--canvas", "1280x720", "--no-color-grade", "--no-vignette",
              "--min-silent", "0", "--render-preset", "ultrafast"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(args: list[str], label: str) -> int:
    log(f"    $ {' '.join(args)}")
    proc = subprocess.run(args, cwd=str(REPO))
    if proc.returncode != 0:
        log(f"    ! {label} exited {proc.returncode}")
    return proc.returncode


def _tag(n: float) -> str:
    """Chapter number -> filename fragment (0.5 -> '0-5', 6.0 -> '6')."""
    return f"{n:g}".replace(".", "-")


def chapter_number(name: str) -> float:
    """`chapter_0007` -> 7.0, `chapter_01_5` -> 1.5, `chapter_00_5` -> 0.5.

    The downloader writes fractional chapter numbers with an underscore instead
    of a dot (0.5 becomes `chapter_00_5`), so reading only the last underscore
    segment would call the prologue "chapter 5" and sort it after chapter 4 --
    silently reordering the story and mis-chaining the memory carry-forward.
    """
    tail = name.lower().split("chapter_")[-1].replace("-", "_")
    parts = [p for p in tail.split("_") if p.isdigit()]
    if len(parts) >= 2:
        try:
            return float(f"{int(parts[0])}.{parts[1]}")
        except ValueError:
            pass
    try:
        return float(parts[0]) if parts else 0.0
    except ValueError:
        return 0.0


def chapter_dirs(src: Path, select: str | None = None) -> list[Path]:
    """Source chapter dirs, ordered by chapter number (fractions included).

    `select` is an optional chapter-number spec: "1-22", "0.5-6", "3,5,9-11" or
    a plain "7". Without it every chapter found in the source folder is built.
    """
    dirs = sorted(src.glob("chapter_*"), key=lambda p: chapter_number(p.name))
    if not select:
        return dirs
    exact: set[float] = set()
    spans: list[tuple[float, float]] = []
    for part in select.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            spans.append((float(lo or 0), float(hi or max(
                (chapter_number(d.name) for d in dirs), default=0))))
        else:
            exact.add(float(part))

    def keep(n: float) -> bool:
        return n in exact or any(a <= n <= b for a, b in spans)

    picked = [d for d in dirs if keep(chapter_number(d.name))]
    missing = {n for n in exact if not any(chapter_number(d.name) == n
                                          for d in picked)}
    if missing:
        log(f"  ! requested chapters not found in {src.name}: "
            f"{sorted(missing)}")
    return picked


def cut(src_dir: Path, out_dir: Path, *, force: bool = False) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "panels.json").is_file() and not force:
        log("  cut: panels.json exists -> skip")
        return True
    rc = run([PY, "scripts/cut_pages_and_merge.py", str(src_dir),
              str(out_dir), "--ext", "webp"]
             + (["--force"] if force else []), "cut")
    return (out_dir / "panels.json").is_file() and rc == 0


def chain_memory(prev_out: Path | None, out_dir: Path, n: int) -> None:
    """Carry the previous chapter's cast/threads into this one (idempotent)."""
    if prev_out is None or not (prev_out / "story_context.json").is_file():
        return
    if (out_dir / "story_context.json").is_file():
        return  # already seeded/merged for this chapter
    try:
        from story_context import carry_forward
        carry_forward(prev_out, out_dir, chapter=n)
        log(f"  memory: carried forward from chapter {n - 1}")
    except Exception as exc:  # noqa: BLE001 - chaining is best-effort
        log(f"  ! memory carry failed (continuing): {exc}")


def narrate(out_dir: Path, *, force: bool = False) -> bool:
    """Run the AI narration + chapter-script pass for one chapter.

    Skipped only when script.json exists AND was written by the current
    script contract. A stale version (e.g. the v4 "3-12 lines, skip panels"
    scripts) must be rebuilt, otherwise the full-coverage work in
    recap_script v5 never reaches the video. This does NOT pass --force, so
    the per-panel vision cache is reused and only genuinely missing captions
    cost a new call.
    """
    sp = out_dir / "script.json"
    if not force and sp.is_file():
        try:
            version = json.loads(sp.read_text("utf-8")).get("version")
        except (OSError, ValueError):
            version = None
        from recap_script import SCRIPT_VERSION
        if version == SCRIPT_VERSION:
            log(f"  narrate: script.json v{version} current -> skip")
            return True
        log(f"  narrate: script.json is v{version}, need "
            f"v{SCRIPT_VERSION} -> rebuilding (vision cache reused)")
    rc = run([PY, "-m", "cli", "guided", "narrate-ai", str(out_dir),
              "--narration-concurrency", "16"]
             + (["--force"] if force else []), "narrate-ai")
    return rc == 0


def render(out_dir: Path, *, force: bool = False) -> bool:
    recap = out_dir / "recap.mp4"
    if (recap.is_file() and recap.stat().st_size > 100_000) and not force:
        log("  video: recap.mp4 exists -> skip")
        return True
    run([PY, "-m", "cli", "guided", "video",
        str(out_dir / "panels.json"), "--out", str(recap),
        *VIDEO_LOOK,
        "--tts", "edge", "--voice", VOICE, "--rate", RATE,
        "--tts-concurrency", "16"]
        + (["--force"] if force else []), "video")
    return recap.is_file() and recap.stat().st_size > 100_000


def concat(recaps: list[Path], final: Path) -> bool:
    if not recaps:
        log("  concat: no chapter recaps to join")
        return False
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - fall back to system ffmpeg
        ff = "ffmpeg"
    # Normalize timescales first: chunked renders historically produced 90k-tbn
    # tracks while direct renders use 15360; a -c copy concat of mixed
    # timebases silently inflates the video timeline (~7x) while the audio
    # stays correct. Remuxing is stream-copy, so this is cheap.
    norm_dir = OUT / "_concat_norm"
    norm_dir.mkdir(exist_ok=True)
    normed: list[Path] = []
    for p in recaps:
        n = norm_dir / (p.parent.name + ".mp4")
        rc = subprocess.run([ff, "-y", "-i", str(p), "-c", "copy",
                             "-video_track_timescale", "15360", str(n)],
                            cwd=str(REPO), capture_output=True).returncode
        normed.append(n if rc == 0 and n.is_file() else p)
    lst = OUT / "concat_list.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in normed),
                   "utf-8")
    log(f"  concat: {len(recaps)} chapter recaps -> {final.name}")
    rc = subprocess.run([ff, "-y", "-f", "concat", "-safe", "0",
                         "-i", str(lst), "-c", "copy", str(final)],
                        cwd=str(REPO)).returncode
    if rc != 0 or not final.is_file():
        log("  concat: stream-copy failed; re-encoding")
        rc = subprocess.run([ff, "-y", "-f", "concat", "-safe", "0",
                             "-i", str(lst), "-c:v", "libx264", "-preset",
                             "veryfast", "-c:a", "aac", str(final)],
                            cwd=str(REPO)).returncode
    for n in norm_dir.glob("*.mp4"):
        n.unlink(missing_ok=True)
    return final.is_file() and rc == 0


def series_title(src: Path, out: Path, select: str | None = None,
                 series: str = "") -> None:
    """One CTR title for the whole arc, from the concatenated chapter recaps."""
    try:
        from recap_script import generate_recap_title
        text = "\n".join(
            (out / d.name / "script.json").read_text("utf-8")
            for d in chapter_dirs(src, select)
            if (out / d.name / "script.json").is_file())[:6000]
        if not text.strip():
            return
        res = generate_recap_title(text, series_title=series)
        if res.get("title"):
            (out / "series_title.txt").write_text(res["title"] + "\n", "utf-8")
            log(f"  series title: {res['title']}")
    except Exception as exc:  # noqa: BLE001 - title is a bonus
        log(f"  ! series title skipped ({exc})")


def main() -> int:
    # --src/--out let one batch drive any downloaded series; they rebind the
    # module globals before any stage reads them, so every helper below (cut,
    # narrate, render, concat, series_title) follows along.
    global SRC, OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=None, metavar="DIR",
                    help="downloaded source folder holding chapter_* dirs "
                         "(default demo_duke)")
    ap.add_argument("--out", default=None, metavar="DIR",
                    help="output folder for the per-chapter + joined recap "
                         "(default demo_duke_out)")
    ap.add_argument("--series", default="",
                    help="series name, used only for the CTR title pass")
    ap.add_argument("--force", action="store_true",
                    help="re-cut AND re-narrate AND re-render everything "
                         "(spends a large batch of AI vision calls)")
    ap.add_argument("--force-video", action="store_true",
                    help="re-render the Phase-3 videos from the existing cut "
                         "+ narration (no AI calls)")
    ap.add_argument("--chapters", default=None, metavar="SPEC",
                    help='chapter numbers to build, e.g. "1-22", "0.5-6" or '
                         '"3,5,9-11" (default: every chapter in the source '
                         "folder)")
    args = ap.parse_args()
    if args.src:
        SRC = (REPO / args.src).resolve()
    if args.out:
        OUT = (REPO / args.out).resolve()
    load_dotenv(REPO / ".env")
    OUT.mkdir(exist_ok=True)
    chapters = chapter_dirs(SRC, args.chapters)
    if not chapters:
        log(f"no chapter_* folders under {SRC} -- nothing to do")
        return 1
    log(f"=== building {len(chapters)} chapters "
        f"({chapters[0].name}..{chapters[-1].name}) -> {OUT} "
        f"(look: {' '.join(VIDEO_LOOK)}, sfx: OFF, force={args.force} "
        f"force-video={args.force_video}) ===")
    recaps: list[Path] = []
    prev_out: Path | None = None
    failures: list[str] = []
    for i, src_dir in enumerate(chapters, 1):
        out_dir = OUT / src_dir.name
        log(f"[{i}/{len(chapters)}] {src_dir.name}")
        ok = cut(src_dir, out_dir, force=args.force)
        if not ok:
            failures.append(f"{src_dir.name}: cut")
            prev_out = out_dir if (out_dir / "story_context.json").is_file() else prev_out
            continue
        chain_memory(prev_out, out_dir, i)
        if not narrate(out_dir, force=args.force):
            failures.append(f"{src_dir.name}: narrate")
        if render(out_dir, force=args.force or args.force_video):
            recaps.append(out_dir / "recap.mp4")
        else:
            failures.append(f"{src_dir.name}: video")
        prev_out = out_dir
    first, last = chapters[0], chapters[-1]
    span = ("" if args.chapters is None else
            f"_ch{_tag(chapter_number(first.name))}-{_tag(chapter_number(last.name))}")
    final = OUT / f"recap_full_16x9{span}.mp4"
    good = concat(recaps, final)
    series_title(SRC, OUT, args.chapters, args.series)
    log("=" * 52)
    log(f"DONE. chapters joined: {len(recaps)}/{len(chapters)}")
    if failures:
        log(f"failed stages ({len(failures)}): " + "; ".join(failures))
    log(f"final video: {final if good else 'NOT PRODUCED'}")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
