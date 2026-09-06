# recap_video.py
"""Phase 3 — turn the cut panels (+ narration) into a recap VIDEO.

Input : panels.json  (CutArtifact written by `guided run` / `guided cut`)
Output: recap.mp4    1080x1920 (9:16), 30 fps, H.264 + AAC, loudness-normalised
        + sidecars:  narration.json, audio.json, timeline.json, recap.srt

Stages (all artifacts are written next to the mp4):

  1. build_narration()   CutPanel -> NarrationEntry.  Script text is the
                         panel narration, optionally followed by the dialogue
                         (skipped when the narration already quotes it).
  2. synthesize_audio()  one mp3 per panel via adapters.tts_edge; duration is
                         MEASURED with ffprobe.  `tts="none"` skips this and
                         produces a silent video timed by reading speed
                         (allowed: with no audio there is nothing to drift).
  3. build_timeline()    contiguous TimelineEntry list.  Each entry lasts
                         max(audio + gap, min_display, pan_travel / max_speed)
                         so narration always finishes and pans are never
                         faster than `max_pan_px_per_sec`.
  4. render_video()      adapters.render_ffmpeg.render (single ffmpeg)
  5. write_srt()         captions from the SAME timeline (+ edge-tts word
                         timings when available) so they cannot drift either.

Caching follows the repo rule: a stage is skipped iff its output exists and
its recorded config_hash / input_hashes match.  --force bypasses.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from adapters.schemas import (
    SCHEMA_VERSION,
    AudioArtifact,
    AudioEntry,
    BBox,
    Meta,
    NarrationArtifact,
    NarrationEntry,
    PanSpec,
    TimelineArtifact,
    TimelineEntry,
)
from guided_cutter import CutArtifact, CutPanel

log = logging.getLogger(__name__)

WIDTH, HEIGHT = 1080, 1920
GENERATOR = "recap_video.1"


class VideoError(RuntimeError):
    """Raised for any user-facing failure in the video stage."""


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class VideoConfig:
    tts: Literal["edge", "none"] = "edge"
    voice: str = "en-US-AriaNeural"
    rate: str = "+0%"            # edge-tts rate, e.g. "+10%"
    pitch: str = "+0Hz"
    include_dialogue: bool = True
    gap_seconds: float = 0.35    # trailing silence after each panel
    min_display_seconds: float = 2.0
    max_display_seconds: float = 12.0   # cap for SILENT panels only
    silent_wpm: int = 160        # reading speed used ONLY when tts == "none"
    max_pan_px_per_sec: int = 450  # slow, readable Ken-Burns pan
    fps: int = 30
    ffmpeg_exe: str = "ffmpeg"
    ffprobe_exe: str = "ffprobe"

    def hash(self) -> str:
        return _sha256_text(json.dumps(asdict(self), sort_keys=True))


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _meta(config_hash: str, input_hashes: dict[str, str]) -> Meta:
    return Meta(schema_version=SCHEMA_VERSION, generator=GENERATOR,
                config_hash=config_hash, input_hashes=input_hashes)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _normalise(text: str) -> str:
    """Whitespace-collapse + ensure terminal punctuation (TTS-friendly)."""
    s = " ".join(text.split())
    if s and s[-1] not in ".!?…\"'”’":
        s += "."
    return s


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def probe_duration(path: Path, ffprobe_exe: str = "ffprobe") -> float:
    """MEASURED media duration in seconds via ffprobe (never estimated)."""
    exe = shutil.which(ffprobe_exe)
    if exe is None:
        raise VideoError(f"{ffprobe_exe!r} not found on PATH; install FFmpeg "
                         "(https://ffmpeg.org/download.html)")
    cmd = [exe, "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False,
                          shell=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise VideoError(f"ffprobe could not read {path}: "
                         f"{proc.stderr.strip()[-300:]}")
    try:
        dur = float(proc.stdout.strip())
    except ValueError as exc:
        raise VideoError(f"ffprobe returned non-numeric duration for {path}: "
                         f"{proc.stdout!r}") from exc
    if dur <= 0:
        raise VideoError(f"{path} has zero duration")
    return dur


# --------------------------------------------------------------------------- #
# Stage 1 — narration entries
# --------------------------------------------------------------------------- #
def script_text(panel: CutPanel, *, include_dialogue: bool = True) -> str:
    """The exact text spoken for one panel.

    narration first; dialogue appended only when it adds information
    (i.e. the narration does not already contain the same words).
    """
    narration = _normalise(panel.narration) if panel.narration.strip() else ""
    dialogue = _normalise(panel.dialogue) if panel.dialogue.strip() else ""
    if not include_dialogue or not dialogue:
        return narration
    if narration and dialogue.strip(".!?\"'”’ ").lower() in narration.lower():
        return narration            # already quoted by the narrator
    return f"{narration} {dialogue}".strip()


def build_narration(artifact: CutArtifact, cfg: VideoConfig,
                    *, panels_hash: str) -> NarrationArtifact:
    panels = sorted(artifact.panels, key=lambda p: p.y_start)
    entries: list[NarrationEntry] = []
    for order, p in enumerate(panels, start=1):
        text = script_text(p, include_dialogue=cfg.include_dialogue)
        quotes = [q.strip() for q in re.findall(r"[\"“]([^\"”]+)[\"”]",
                                                p.dialogue or "")]
        entries.append(NarrationEntry(id=p.id, panel_id=p.id, order=order,
                                      speaker=None, text=text, quotes=quotes))
    return NarrationArtifact(
        meta=_meta(cfg.hash(), {"panels.json": panels_hash}),
        mode="narrator", entries=entries)


# --------------------------------------------------------------------------- #
# Stage 2 — TTS
# --------------------------------------------------------------------------- #
def synthesize_audio(narration: NarrationArtifact, audio_dir: Path,
                     cfg: VideoConfig, *, force: bool = False) -> AudioArtifact:
    """One mp3 per non-empty entry.  Reuses audio.json when hashes match."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    sidecar = audio_dir / "audio.json"
    narration_hash = _sha256_text(narration.model_dump_json())
    input_hashes = {"narration.json": narration_hash}

    if cfg.tts == "none":
        return AudioArtifact(meta=_meta(cfg.hash(), input_hashes),
                             voice="none", entries=[])

    if sidecar.exists() and not force:
        try:
            prev = AudioArtifact.model_validate_json(sidecar.read_text("utf-8"))
            if (prev.meta.config_hash == cfg.hash()
                    and prev.meta.input_hashes == input_hashes
                    and all((audio_dir / e.path).is_file()
                            for e in prev.entries)):
                log.info("audio cache hit (%d clips)", len(prev.entries))
                return prev
        except Exception as exc:  # noqa: BLE001 - stale/corrupt cache
            log.debug("ignoring unreadable audio.json: %s", exc)

    try:
        from adapters.tts_edge import synthesize_entry
    except ImportError as exc:
        raise VideoError("edge-tts is not installed: pip install edge-tts "
                         "(or use --tts none)") from exc

    entries: list[AudioEntry] = []
    total = sum(1 for e in narration.entries if e.text.strip())
    done = 0
    for e in narration.entries:
        out = synthesize_entry(
            e, audio_dir, voice=cfg.voice, rate=cfg.rate, pitch=cfg.pitch,
            probe_duration=lambda p: probe_duration(p, cfg.ffprobe_exe))
        if out is None:
            continue
        entries.append(out)
        done += 1
        log.info("tts %d/%d  %s  %.2fs", done, total, e.id, out.duration_seconds)

    artifact = AudioArtifact(meta=_meta(cfg.hash(), input_hashes),
                             voice=cfg.voice, entries=entries)
    _write_atomic(sidecar, artifact.model_dump_json(indent=2) + "\n")
    return artifact


# --------------------------------------------------------------------------- #
# Stage 3 — timeline
# --------------------------------------------------------------------------- #
def compute_pan(width: int, height: int) -> PanSpec:
    """Scale the panel so BOTH dims cover 1080x1920; pan along the overflow.

    Never centre-crops away content: a tall panel is read top->bottom
    (pan_down), a wide one left->right (pan_right).  Exact-fit is static.
    """
    if width <= 0 or height <= 0:
        raise ValueError("panel must have positive size")
    scale = max(WIDTH / width, HEIGHT / height)
    # -1e-6 guards against float noise (800 * 1.35 == 1080.0000000000002)
    scaled_w = max(WIDTH, math.ceil(width * scale - 1e-6))
    scaled_h = max(HEIGHT, math.ceil(height * scale - 1e-6))
    over_h, over_w = scaled_h - HEIGHT, scaled_w - WIDTH
    if over_h > 2 and over_h >= over_w:
        return PanSpec(kind="pan_down", scaled_w=scaled_w, scaled_h=scaled_h,
                       travel_px=over_h)
    if over_w > 2:
        return PanSpec(kind="pan_right", scaled_w=scaled_w, scaled_h=scaled_h,
                       travel_px=over_w)
    return PanSpec(kind="static", scaled_w=WIDTH, scaled_h=HEIGHT, travel_px=0)


def display_seconds(*, audio_seconds: float | None, words: int,
                    travel_px: int, cfg: VideoConfig) -> float:
    """How long a panel stays on screen (INCLUDING its trailing gap)."""
    pan_floor = travel_px / cfg.max_pan_px_per_sec if travel_px else 0.0
    if audio_seconds is not None:
        # spoken panel: narration must finish; never capped
        return round(max(audio_seconds + cfg.gap_seconds,
                         cfg.min_display_seconds, pan_floor), 3)
    # silent panel: reading-speed heuristic (no audio => no drift possible)
    read = (words / cfg.silent_wpm) * 60.0 if words else 0.0
    dur = min(max(read + cfg.gap_seconds, cfg.min_display_seconds),
              cfg.max_display_seconds)
    return round(max(dur, pan_floor), 3)


def build_timeline(artifact: CutArtifact, panels_dir: Path,
                   narration: NarrationArtifact, audio: AudioArtifact,
                   audio_dir: Path, cfg: VideoConfig,
                   *, panels_hash: str) -> TimelineArtifact:
    by_audio = {a.entry_id: a for a in audio.entries}
    by_text = {n.id: n for n in narration.entries}
    entries: list[TimelineEntry] = []
    t = 0.0
    for order, p in enumerate(sorted(artifact.panels,
                                     key=lambda p: p.y_start), start=1):
        h = p.y_end - p.y_start
        if h <= 0:
            log.warning("skipping zero-height panel %s", p.id)
            continue
        img = (panels_dir / p.image_file).resolve()
        if not img.is_file():
            raise VideoError(f"panel image missing: {img} "
                             "(re-run 'guided cut' or 'guided run')")
        pan = compute_pan(artifact.width, h)
        a = by_audio.get(p.id)
        text = by_text[p.id].text if p.id in by_text else ""
        dur = display_seconds(
            audio_seconds=a.duration_seconds if a else None,
            words=_word_count(text), travel_px=pan.travel_px, cfg=cfg)
        entries.append(TimelineEntry(
            panel_id=p.id, order=order, source_image=str(img),
            bbox=BBox(x=0, y=p.y_start, w=artifact.width, h=h),
            start_seconds=round(t, 3), duration_seconds=dur,
            audio_path=str((audio_dir / a.path).resolve()) if a else None,
            pan=pan))
        t += dur
    if not entries:
        raise VideoError("no usable panels in panels.json")
    return TimelineArtifact(
        meta=_meta(cfg.hash(), {
            "panels.json": panels_hash,
            "narration.json": _sha256_text(narration.model_dump_json()),
            "audio.json": _sha256_text(audio.model_dump_json())}),
        width=WIDTH, height=HEIGHT, fps=cfg.fps,
        gap_seconds=cfg.gap_seconds,
        min_display_seconds=cfg.min_display_seconds, entries=entries)


def total_seconds(timeline: TimelineArtifact) -> float:
    return round(sum(e.duration_seconds for e in timeline.entries), 3)


# --------------------------------------------------------------------------- #
# Stage 5 — captions
# --------------------------------------------------------------------------- #
def srt_time(seconds: float) -> str:
    ms = int(round(max(seconds, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _cues_for_entry(entry: TimelineEntry, text: str,
                    audio: AudioEntry | None,
                    max_words: int = 9) -> list[tuple[float, float, str]]:
    """(start, end, text) cues in ABSOLUTE seconds for one panel."""
    if not text.strip():
        return []
    base = entry.start_seconds
    words = audio.words if audio else []
    if len(words) >= 2:
        cues: list[tuple[float, float, str]] = []
        for i in range(0, len(words), max_words):
            chunk = words[i:i + max_words]
            cues.append((base + float(chunk[0]["start"]),
                         base + float(chunk[-1]["end"]) + 0.15,
                         " ".join(str(w["text"]) for w in chunk)))
        return cues
    # no word timings: one cue over the speech (or the whole silent panel)
    end = base + (audio.duration_seconds if audio else entry.duration_seconds)
    return [(base, end, text)]


def write_srt(timeline: TimelineArtifact, narration: NarrationArtifact,
              audio: AudioArtifact, out_path: Path) -> int:
    by_text = {n.id: n.text for n in narration.entries}
    by_audio = {a.entry_id: a for a in audio.entries}
    lines: list[str] = []
    n = 0
    for e in timeline.entries:
        cues = _cues_for_entry(e, by_text.get(e.panel_id, ""),
                               by_audio.get(e.panel_id))
        for start, end, text in cues:
            end = min(end, e.start_seconds + e.duration_seconds)
            if end <= start:
                continue
            n += 1
            lines += [str(n), f"{srt_time(start)} --> {srt_time(end)}",
                      " ".join(text.split()), ""]
    _write_atomic(out_path, "\n".join(lines) + ("\n" if lines else ""))
    return n


# --------------------------------------------------------------------------- #
# Stage 4 — render
# --------------------------------------------------------------------------- #
def render_video(timeline: TimelineArtifact, out_path: Path,
                 cfg: VideoConfig) -> None:
    exe = shutil.which(cfg.ffmpeg_exe)
    if exe is None:
        raise VideoError(f"{cfg.ffmpeg_exe!r} not found on PATH; install "
                         "FFmpeg (https://ffmpeg.org/download.html)")
    from adapters.render_ffmpeg import RenderError, render
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".partial.mp4")
    try:
        render(timeline, tmp, ffmpeg_exe=exe)
    except RenderError as exc:
        raise VideoError(str(exc)) from exc
    tmp.replace(out_path)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def make_recap_video(panels_json: Path, out_path: Path,
                     cfg: VideoConfig | None = None, *,
                     force: bool = False, dry_run: bool = False
                     ) -> dict[str, Any]:
    """panels.json -> recap.mp4 (+ sidecars).  Returns a summary dict."""
    cfg = cfg or VideoConfig()
    panels_json = Path(panels_json)
    if not panels_json.is_file():
        raise FileNotFoundError(f"panels.json not found: {panels_json}")
    panels_dir = panels_json.parent
    out_path = Path(out_path)
    work = out_path.parent
    work.mkdir(parents=True, exist_ok=True)
    audio_dir = work / "audio"

    artifact = CutArtifact.model_validate_json(panels_json.read_text("utf-8"))
    if not artifact.panels:
        raise VideoError("panels.json contains no panels")
    panels_hash = _sha256_file(panels_json)

    # 1. narration
    narration = build_narration(artifact, cfg, panels_hash=panels_hash)
    _write_atomic(work / "narration.json",
                  narration.model_dump_json(indent=2) + "\n")
    spoken = sum(1 for e in narration.entries if e.text.strip())
    if spoken == 0:
        log.warning("no narration in panels.json (fallback plan?) — the "
                    "video will be silent; consider re-running 'guided run' "
                    "with an AI backend")

    # 2. audio
    audio = synthesize_audio(narration, audio_dir, cfg, force=force)

    # 3. timeline
    timeline = build_timeline(artifact, panels_dir, narration, audio,
                              audio_dir, cfg, panels_hash=panels_hash)
    _write_atomic(work / "timeline.json",
                  timeline.model_dump_json(indent=2) + "\n")

    # 5. captions (before render so a render failure still leaves them)
    srt_path = out_path.with_suffix(".srt")
    cues = write_srt(timeline, narration, audio, srt_path)

    summary: dict[str, Any] = {
        "panels": len(timeline.entries),
        "spoken_panels": len(audio.entries),
        "total_seconds": total_seconds(timeline),
        "voice": audio.voice,
        "timeline": str(work / "timeline.json"),
        "srt": str(srt_path),
        "srt_cues": cues,
        "video": None,
    }
    if dry_run:
        return summary

    # 4. render (cache: skip when the timeline hash is unchanged)
    stamp = out_path.with_name(out_path.name + ".hash")
    tl_hash = _sha256_text(timeline.model_dump_json())
    if (out_path.is_file() and stamp.is_file() and not force
            and stamp.read_text("utf-8").strip() == tl_hash):
        log.info("video cache hit: %s", out_path)
    else:
        render_video(timeline, out_path, cfg)
        _write_atomic(stamp, tl_hash + "\n")
    summary["video"] = str(out_path)
    return summary
