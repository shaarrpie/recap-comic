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
import time
import wave
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

try:
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

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
    tts: Literal["edge", "kokoro", "none"] = "edge"
    voice: str = "en-US-AriaNeural"
    rate: str = "+0%"            # edge-tts rate, e.g. "+10%"
    pitch: str = "+0Hz"
    speed: float = 1.0           # kokoro speed multiplier
    include_dialogue: bool = True
    gap_seconds: float = 0.35    # trailing silence after each panel
    min_display_seconds: float = 2.0
    max_display_seconds: float = 12.0   # cap for SILENT panels only
    silent_wpm: int = 160        # reading speed used ONLY when tts == "none"
    max_pan_px_per_sec: int = 450  # slow, readable Ken-Burns pan
    fps: int = 30
    ffmpeg_exe: str = "ffmpeg"
    ffprobe_exe: str = "ffprobe"
    kokoro_model_path: Path | None = None
    kokoro_voices_path: Path | None = None

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
_CJK_RE = re.compile(r"[\u3040-\u30ff\uac00-\ud7af\u4e00-\u9fff]")


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text)) + len(_CJK_RE.findall(text))


def _resolve_ffmpeg(exe: str = "ffmpeg") -> str:
    resolved = shutil.which(exe)
    if resolved is not None:
        return resolved
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise VideoError(f"{exe!r} not found on PATH or via imageio-ffmpeg; "
                     "install FFmpeg (https://ffmpeg.org/download.html)")


def _resolve_ffprobe(exe: str = "ffprobe") -> str | None:
    resolved = shutil.which(exe)
    if resolved is not None:
        return resolved
    ffmpeg_path = _resolve_ffmpeg()
    parent = Path(ffmpeg_path).parent
    candidate = parent / exe
    if candidate.is_file():
        return str(candidate)
    return None


def _probe_with_ffmpeg(path: Path, ffmpeg_exe: str) -> float:
    cmd = [ffmpeg_exe, "-i", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False,
                          shell=False, timeout=30)
    stderr = proc.stderr or ""
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", stderr)
    if not m:
        raise VideoError(f"ffmpeg could not probe duration for {path}: "
                         f"{stderr.strip()[-300:]}")
    hours, minutes, seconds = m.groups()
    dur = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if dur <= 0:
        raise VideoError(f"{path} has zero duration")
    return dur


def probe_duration(path: Path, ffprobe_exe: str = "ffprobe") -> float:
    """MEASURED media duration in seconds via ffprobe (or ffmpeg fallback)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as f:
                n_frames = f.getnframes()
                rate = f.getframerate()
                if rate > 0:
                    dur = n_frames / rate
                    if dur > 0:
                        return dur
        except Exception:
            pass
    ffprobe = _resolve_ffprobe(ffprobe_exe)
    if ffprobe is not None:
        exe = ffprobe
        cmd = [exe, "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False,
                              shell=False, timeout=30)
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                dur = float(proc.stdout.strip())
                if dur > 0:
                    return dur
            except ValueError:
                pass
    ffmpeg = _resolve_ffmpeg()
    return _probe_with_ffmpeg(path, ffmpeg)


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
    if narration and _token_overlap_ratio(dialogue, narration) >= 0.8:
        return narration
    return f"{narration} {dialogue}".strip()


def _token_overlap_ratio(a: str, b: str) -> float:
    """Fraction of word tokens in `a` that also appear in `b`."""
    tokens_a = set(re.findall(r"[A-Za-z0-9']+|[\u3040-\u30ff\uac00-\ud7af\u4e00-\u9fff]", a.lower()))
    tokens_b = set(re.findall(r"[A-Za-z0-9']+|[\u3040-\u30ff\uac00-\ud7af\u4e00-\u9fff]", b.lower()))
    if not tokens_a:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


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
    result = NarrationArtifact(
        meta=_meta(cfg.hash(), {"panels.json": panels_hash}),
        mode="narrator", entries=entries)
    log.info("build_narration entries=%d", len(entries))
    return result


# --------------------------------------------------------------------------- #
# Stage 2 — TTS
# --------------------------------------------------------------------------- #
def synthesize_audio(narration: NarrationArtifact, audio_dir: Path,
                     cfg: VideoConfig, *, force: bool = False,
                     retries: int = 3) -> AudioArtifact:
    """One mp3/wav per non-empty entry. Reuses audio.json when hashes match."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    sidecar = audio_dir / "audio.json"
    # B4: cache key includes text content + voice + provider so changing
    # voice or provider invalidates the cache, but re-running with the same
    # inputs reuses clips.
    tts_input_parts = []
    for e in narration.entries:
        if e.text.strip():
            tts_input_parts.append(f"{e.id}:{e.text}:{cfg.voice}:{cfg.tts}")
    tts_input_hash = _sha256_text("\n".join(tts_input_parts))
    input_hashes = {
        "narration.json": _sha256_text(narration.model_dump_json()),
        "tts_input": tts_input_hash,
    }

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
        from adapters.tts import synthesize_entry as tts_synth
    except ImportError as exc:
        raise VideoError(f"TTS provider not available: {exc}") from exc

    entries: list[AudioEntry] = []
    prev_text = ""
    synth_count = 0
    for e in narration.entries:
        if not e.text.strip():
            continue
        if e.text.strip() == prev_text:
            log.info("TTS skipped %s: duplicate of previous narration", e.id)
            continue
        prev_text = e.text.strip()
        synth_count += 1
    total = synth_count
    done = 0
    if _HAS_RICH:
        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        )
        progress_ctx.start()
        task_id = progress_ctx.add_task(
            "[cyan]Synthesizing TTS audio...",
            total=total,
        )
    else:
        progress_ctx = None
        task_id = None
    try:
        prev_text = ""
        for e in narration.entries:
            text = e.text.strip()
            if not text:
                continue
            if text == prev_text:
                log.info("TTS skipped %s: duplicate of previous narration", e.id)
                continue
            prev_text = text
            out, err = tts_synth(
                e, audio_dir, provider=cfg.tts, voice=cfg.voice,
                rate=cfg.rate, pitch=cfg.pitch, speed=cfg.speed,
                probe_duration=lambda p: probe_duration(p, cfg.ffprobe_exe),
                kokoro_model_path=cfg.kokoro_model_path,
                kokoro_voices_path=cfg.kokoro_voices_path,
                retries=retries)
            if err is not None or out is None:
                log.warning("TTS skipped %s: %s", e.id, err or "empty text")
                continue
            entries.append(out)
            done += 1
            log.info("tts %d/%d  %s  %.2fs", done, total, e.id, out.duration_seconds)
            if progress_ctx is not None and task_id is not None:
                progress_ctx.update(task_id, advance=1)
    finally:
        if progress_ctx is not None:
            progress_ctx.stop()

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
    if scale > 4.0:
        scale = 4.0
    scaled_w = math.ceil(width * scale)
    scaled_h = math.ceil(height * scale)
    over_h, over_w = scaled_h - HEIGHT, scaled_w - WIDTH
    if over_h > 2 and over_h >= over_w:
        return PanSpec(kind="pan_down", scaled_w=scaled_w, scaled_h=scaled_h,
                       travel_px=over_h)
    if over_w > 2:
        return PanSpec(kind="pan_right", scaled_w=scaled_w, scaled_h=scaled_h,
                       travel_px=over_w)
    return PanSpec(kind="static", scaled_w=scaled_w, scaled_h=scaled_h, travel_px=0)


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
    skipped_missing = 0
    for order, p in enumerate(sorted(artifact.panels,
                                     key=lambda p: p.y_start), start=1):
        h = p.y_end - p.y_start
        if h <= 0:
            log.warning("skipping zero-height panel %s", p.id)
            continue
        img = (panels_dir / p.image_file).resolve()
        if not img.is_file():
            # panels.json can reference panels whose PNG was skipped during
            # the cut (too thin / zero-range / etc.) or that belong to a
            # previous run that was cleaned up. Skipping keeps the rest of
            # the timeline usable instead of failing the whole render.
            log.warning(
                "skipping panel %s: image file missing at %s "
                "(panels.json references it but the PNG was not produced; "
                "re-run 'guided cut' / 'guided run' to regenerate)",
                p.id, img)
            skipped_missing += 1
            continue
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
    result = TimelineArtifact(
        meta=_meta(cfg.hash(), {
            "panels.json": panels_hash,
            "narration.json": _sha256_text(narration.model_dump_json()),
            "audio.json": _sha256_text(audio.model_dump_json())}),
        width=WIDTH, height=HEIGHT, fps=cfg.fps,
        gap_seconds=cfg.gap_seconds,
        min_display_seconds=cfg.min_display_seconds, entries=entries)
    log.info("build_timeline entries=%d total_duration=%.2fs",
             len(entries), total_seconds(result))
    return result


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
    exe = _resolve_ffmpeg(cfg.ffmpeg_exe)
    from adapters.render_ffmpeg import RenderError, render
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".partial.mp4")
    log.info("render_video start out=%s timeline_entries=%d",
             out_path, len(timeline.entries))
    t0 = time.time()
    if _HAS_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress_ctx:
            progress_ctx.add_task("Rendering video with ffmpeg...")
            try:
                render(timeline, tmp, ffmpeg_exe=exe)
            except RenderError as exc:
                raise VideoError(str(exc)) from exc
    else:
        try:
            render(timeline, tmp, ffmpeg_exe=exe)
        except RenderError as exc:
            raise VideoError(str(exc)) from exc
    elapsed = time.time() - t0
    tmp.replace(out_path)
    _apply_faststart(out_path, exe)
    log.info("render_video complete out=%s duration=%.2fs", out_path, elapsed)


def _apply_faststart(path: Path, ffmpeg_exe: str) -> None:
    tmp = path.with_name(path.stem + ".faststart.mp4")
    cmd = [ffmpeg_exe, "-y", "-nostdin", "-i", str(path),
           "-c", "copy", "-movflags", "+faststart", str(tmp)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-10:])
        log.warning("faststart post-process failed for %s: %s", path, tail)
        if tmp.is_file():
            tmp.unlink()
    else:
        tmp.replace(path)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def make_recap_video(panels_json: Path, out_path: Path,
                     cfg: VideoConfig | None = None,
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
    log.info("make_recap_video start panels=%d tts=%s voice=%s",
             len(artifact.panels), cfg.tts, cfg.voice)

    # 1. narration
    narration = build_narration(artifact, cfg, panels_hash=panels_hash)
    _write_atomic(work / "narration.json",
                  narration.model_dump_json(indent=2) + "\n")
    spoken = sum(1 for e in narration.entries if e.text.strip())
    if spoken == 0:
        # Distinguish "AI was never asked" from "AI was rate-limited" so the
        # user knows whether re-running with --backend <other> would help.
        # We detect a fallback plan by looking for provenance == 'fallback'
        # in the most recent plan.json next to panels.json.
        prov = "unknown"
        plan_json = work / "plan.json"
        if plan_json.is_file():
            try:
                import json as _json
                prov = _json.loads(plan_json.read_text("utf-8")).get(
                    "provenance", "unknown")
            except Exception:  # noqa: BLE001 - best-effort provenance probe
                pass
        if prov == "fallback":
            log.warning(
                "no narration in panels.json (provenance=fallback — the AI "
                "pre-read was skipped, returned no plan, or hit a quota / "
                "rate-limit). The video will be silent. To fix: re-run "
                "'guided run' once your API quota resets, OR pass "
                "--tts none to skip TTS entirely, OR populate "
                "narration.json by hand.")
        else:
            log.warning(
                "no narration in panels.json (provenance=%s). The video will "
                "be silent; consider re-running 'guided run' with an AI "
                "backend (gemini/openai/anthropic/ollama).", prov)

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
        log.info("make_recap_video dry_run summary=%s", summary)
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
    log.info("make_recap_video complete summary=%s", summary)
    return summary


# --------------------------------------------------------------------------- #
# Editor render — reuse existing assets, apply user overrides
# --------------------------------------------------------------------------- #
def render_edited_project(editor_path: Path, out_path: Path,
                          cfg: VideoConfig | None = None) -> dict[str, Any]:
    """Render a video from an editor.json without regenerating narration/audio.

    Reads the edited timeline, applies user overrides (duration, effect), rebuilds
    captions from edited captions, and renders via the existing FFmpeg pipeline.
    """
    from adapters.editor import Editor
    cfg = cfg or VideoConfig()
    editor = Editor.load(editor_path)
    session_dir = editor_path.parent
    panels_dir = session_dir
    audio_dir = session_dir / "audio"

    # Reconstruct timeline from edited entries
    tl_entries = []
    for e in editor.project.edited_timeline:
        effect = next((fx for fx in editor.project.effects if fx["panel_id"] == e["panel_id"]), None)
        pan_kind = effect["kind"] if effect else e.get("pan", {}).get("kind", "static")
        scaled_w = e.get("pan", {}).get("scaled_w", WIDTH)
        scaled_h = e.get("pan", {}).get("scaled_h", HEIGHT)
        travel_px = e.get("pan", {}).get("travel_px", 0)
        if pan_kind in ("zoom_in", "zoom_out"):
            scaled_w = WIDTH
            scaled_h = HEIGHT
            travel_px = 0
        audio_path = e.get("audio_path")
        if audio_path and not Path(audio_path).is_file():
            audio_path = None
        tl_entries.append(TimelineEntry(
            panel_id=e["panel_id"],
            order=e["order"],
            source_image=e["source_image"],
            bbox=BBox(**e["bbox"]),
            start_seconds=e["start_seconds"],
            duration_seconds=e["duration_seconds"],
            audio_path=audio_path,
            pan=PanSpec(kind=pan_kind, scaled_w=scaled_w, scaled_h=scaled_h, travel_px=travel_px),
        ))

    timeline = TimelineArtifact(
        meta=_meta(cfg.hash(), {"editor.json": _sha256_text(editor_path.read_text("utf-8"))}),
        width=editor.project.project.get("width", WIDTH),
        height=editor.project.project.get("height", HEIGHT),
        fps=editor.project.project.get("fps", cfg.fps),
        gap_seconds=cfg.gap_seconds,
        min_display_seconds=cfg.min_display_seconds,
        entries=tl_entries,
    )

    # Build SRT from edited captions
    srt_path = out_path.with_suffix(".srt")
    _write_srt_from_editor(timeline, editor.project.captions, srt_path)

    # Render
    render_video(timeline, out_path, cfg)

    summary: dict[str, Any] = {
        "panels": len(timeline.entries),
        "total_seconds": total_seconds(timeline),
        "timeline": str(session_dir / "timeline.json"),
        "srt": str(srt_path),
        "srt_cues": len(editor.project.captions),
        "video": str(out_path),
    }
    editor.project.last_rendered_at = time.time()
    editor.project.needs_render = False
    editor.save(editor_path)
    log.info("render_edited_project complete out=%s", out_path)
    return summary


def _write_srt_from_editor(timeline: TimelineArtifact,
                           captions: list[dict[str, Any]], out_path: Path) -> int:
    lines: list[str] = []
    n = 0
    for cap in captions:
        start = cap["start_seconds"]
        end = cap["end_seconds"]
        text = cap["text"].strip()
        if end <= start or not text:
            continue
        n += 1
        lines += [str(n), f"{srt_time(start)} --> {srt_time(end)}",
                  " ".join(text.split()), ""]
    _write_atomic(out_path, "\n".join(lines) + ("\n" if lines else ""))
    return n
