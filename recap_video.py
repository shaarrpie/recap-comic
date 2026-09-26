# recap_video.py
"""Phase 3 — turn the cut panels (+ narration) into a recap VIDEO.

Input : panels.json  (CutArtifact written by `guided run` / `guided cut`)
Output: recap.mp4    1080x1920 (9:16), 30 fps, H.264 + AAC, loudness-normalised
        + sidecars:  narration.json, audio.json, timeline.json, recap.srt
        (+ sfx.json and an audio mixdown when --sfx-dir is set)

Stages (all artifacts are written next to the mp4):

  1. build_narration()   CutPanel -> NarrationEntry.  Script text is the
                         panel narration, optionally followed by the dialogue
                         (skipped when the narration already quotes it).
  2. synthesize_audio()  one audio clip per panel via adapters.tts (kokoro); duration is
                         MEASURED with ffprobe.  `tts="none"` skips this and
                         produces a silent video timed by reading speed
                         (allowed: with no audio there is nothing to drift).
  3. build_timeline()    contiguous TimelineEntry list.  Each entry lasts
                         max(audio + gap, min_display, pan_travel / max_speed)
                         so narration always finishes and pans are never
                         faster than `max_pan_px_per_sec`.
  4. render_video()      adapters.render_ffmpeg.render (single ffmpeg)
  5. write_srt()         captions from the SAME timeline (+ word
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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from adapters.schemas import (
    SCHEMA_VERSION,
    AudioArtifact,
    AudioEntry,
    BBox,
    Meta,
    NarrationArtifact,
    NarrationEntry,
    PanSpec,
    SfxArtifact,
    SfxEvent,
    TimelineArtifact,
    TimelineEntry,
)
from guided_cutter import CutArtifact, CutPanel
from text_clean import is_promo_text, strip_non_latin

try:
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

# Set to True by the webapp (webapp.pipeline) before running jobs: the
# rich.Progress bars are a CLI affordance and garble the shared uvicorn
# console when 10+ jobs run concurrently. The frontend tracks progress
# through job stages instead.
EMBEDDED_MODE = False

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
    tts: Literal["edge", "kokoro", "none"] = "kokoro"  # edge=cloud, kokoro=local CPU, none=silent
    voice: str = "af_heart"
    rate: str = "+0%"            # legacy edge-tts knob: accepted, ignored
    pitch: str = "+0Hz"          # legacy edge-tts knob: accepted, ignored
    speed: float = 1.0           # kokoro speed multiplier
    include_dialogue: bool = True
    gap_seconds: float = 0.0     # trailing silence after each panel; 0 so the
    # narrator runs straight into the next line with no pause to catch breath
    min_display_seconds: float = 2.0
    max_display_seconds: float = 12.0   # cap for SILENT panels only
    min_silent: float = 1.0      # seconds an un-narrated panel shows between
    # its narrated neighbours. The chapter script speaks fewer lines than
    # there are panels; the skipped panels stay on screen as short silent
    # beats so the video shows the whole chapter instead of jumping. Set to
    # 0 to drop them instead: the timeline then holds only spoken panels, so
    # the narrator never pauses and no shot is rushed through a filler hold.
    silent_wpm: int = 160        # reading speed used ONLY when tts == "none"
    # ── Speech window (natural-flow recap beats) ────────────────────────────
    # The narrator is the master: it reads each line at its own pace and the
    # panel lasts exactly as long as that line (gap_seconds is 0, so lines run
    # back-to-back with no pause). The cap only stops a frame from LINGERING
    # past short narration (min-display / pan / class-mult inflation) — it
    # never cuts or speeds up speech. Because per-panel text is trimmed to
    # whole sentences inside speech_target_seconds and there is no trailing
    # gap, a normal line already lands under 5s with no tempo change. Only a
    # pathologically long sentence runs past 5s, and we let it: clean sentence
    # flow beats an artificial hard cap. Disable with speech_window=False for
    # the legacy (lingering, uncapped) behaviour.
    speech_window: bool = True
    speech_target_seconds: float = 4.0  # trim budget aim (sub-5s window)
    speech_max_seconds: float = 4.6     # trim/cap budget; normal lines fit under
    # 5s with no gap and no tempo change (4.6 + 0.0 gap < 5s)
    speech_wpm: int = 140          # conservative budgeting rate (TTS varies;
                                   # slow estimate keeps real audio in-window)
    max_pan_px_per_sec: int = 450  # slow, readable Ken-Burns pan
    # When a panel's pan would outlast its narration, speed the pan up to
    # fit inside the speech window (bounded: never more than this multiple
    # of max_pan_px_per_sec) instead of holding silent frames after the
    # voice stops. Default off keeps the classic "pan floor always wins"
    # pacing; opt in when panels visibly linger past the narration.
    pan_fit_speech: bool = False
    pan_fit_speech_speedup: float = 3.0
    # Per-class pacing (Fix: flat TTS-length pacing reads as monotone).
    # Multipliers apply AFTER floors; action may also drop below
    # min_display_seconds down to action_floor_seconds.
    action_floor_seconds: float = 0.8
    class_duration_multiplier: dict[str, float] | None = None  # set in __post_init__
    fps: int = 30
    # Output canvas. Defaults to 1080x1920 (9:16 portrait). Pass 1920x1080
    # for a 16:9 landscape edit. Drives pan geometry, the TimelineArtifact's
    # declared dimensions, and the ffmpeg crop; participates in hash() so a
    # canvas change invalidates the video cache.
    canvas_w: int = WIDTH
    canvas_h: int = HEIGHT
    ffmpeg_exe: str = "ffmpeg"
    ffprobe_exe: str = "ffprobe"
    kokoro_model_path: Path | None = None
    kokoro_voices_path: Path | None = None
    # ── Visual style (manhwa-recap look) ────────────────────────────────────
    # The panel floats on a blurred full-frame copy of itself with a strong
    # dark vignette; colour grade is OFF by default. See adapters.render_ffmpeg
    # StyleConfig. blur_background makes panels contain-fitted (whole panel
    # visible, no Ken-Burns pan); the timeline geometry follows automatically.
    blur_background: bool = True
    color_grade: bool = False
    vignette: bool = True
    vignette_angle: str = "PI/2.5"   # ffmpeg angle expr; smaller = stronger
    blur_sigma: float = 40.0         # gblur sigma of the background branch
    # Ken-Burns push-in strength (fraction of fitted size reached by the
    # last frame). 0.25 = the panel grows to 1.25x. Applies to the blur
    # foreground and to the zoom_in/zoom_out pan kinds. 0 disables motion.
    # Kept small so zoom animations stay slow and cinematic.
    zoom_strength: float = 0.25
    # ── Reference-motion preset (editing style) ─────────────────────────────
    # "reference" (default) reproduces the strict camera cycle of
    # reference_motion_preset.json — a repeating 4-beat sequence (zoom in,
    # pan down, pan up, zoom out) normalized to each panel/canvas. "none"
    # falls back to the geometry-driven automation. motion_strength scales
    # pan travel (1.0 = as measured); motion_preset_path overrides the
    # bundled JSON (custom templates).
    motion_preset: str = "reference"
    motion_preset_path: Path | None = None
    motion_strength: float = 1.0
    # Kokoro clips are independent, so they are synthesized in a bounded
    # thread pool. 6 keeps laptop CPU busy without thrashing; lower to 2-3
    # on weak machines, set 1 to serialize.
    tts_concurrency: int = 6
    # ── Narration fit (opt-in tempo backstop) ───────────────────────────────
    # OFF by default: the narrator is NEVER sped up or slowed down — a rushed
    # read is worse than a panel that runs a hair past 5s, so a line keeps its
    # natural pacing and the panel simply catches up to it. When explicitly
    # enabled, every MEASURED clip over speech_max_seconds is tempo-adjusted
    # with ffmpeg atempo (pitch preserved, bounded by speech_fit_max_speedup);
    # only when even max speed cannot fit is the tail trimmed, with a warning.
    # Both fields participate in hash_essentials: toggling them changes the
    # audio bytes, so caches invalidate (one refit/re-synth, then stable).
    speech_fit_audio: bool = False
    speech_fit_max_speedup: float = 2.0
    # ── SFX (automatic sound effects) ────────────────────────────────────────
    # sfx_dir points at a sound bank with transition/ action/ reveal/
    # subfolders (.wav/.mp3/.ogg/.m4a/.flac; an optional default/ fills any
    # empty category). When set, a deterministic plan (sfx.json) tags panel
    # cuts and action/reveal narration beats, and the mixdown overlays the
    # sounds onto the finished video. Render-only: never changes narration
    # or TTS bytes, so these fields are excluded from hash_essentials.
    sfx_dir: Path | None = None
    sfx_volume: float = 0.9         # global loudness multiplier
    sfx_volumes: dict[str, float] | None = None  # per-kind, set in __post_init__

    def __post_init__(self) -> None:
        if not self.class_duration_multiplier:
            self.class_duration_multiplier = {
                "action": 1.0,
                "reveal": 1.35,   # hold the beat so the moment lands
                "dialogue": 1.0,
                "calm": 1.0,
            }
        if not self.sfx_volumes:
            self.sfx_volumes = {"transition": 0.5, "action": 0.9, "reveal": 0.6}
        mp = (self.motion_preset or "none").lower()
        if mp not in ("none", "reference"):
            raise ValueError(
                f"unknown motion_preset {self.motion_preset!r}; "
                "choose 'none' or 'reference'")
        self.motion_preset = mp
        if self.motion_strength < 0:
            raise ValueError("motion_strength must be >= 0")

    def hash(self) -> str:
        # Path objects are not JSON serializable; convert to strings
        def _default(o):
            if isinstance(o, Path):
                return str(o)
            raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
        return _sha256_text(json.dumps(asdict(self), sort_keys=True, default=_default))

    # Visual-only style fields. Narration and audio depend solely on
    # text/voice/provider/pacing, so these are excluded from the TTS cache
    # key: toggling the vignette must not re-synthesize identical clips.
    # The full hash() still covers the timeline + video cache, where the
    # style genuinely changes the output. Motion-preset fields are likewise
    # timeline/render-only (they never change narration text or TTS audio,
    # only camera geometry and pacing distribution).
    _STYLE_FIELDS = ("blur_background", "color_grade", "vignette",
                     "vignette_angle", "blur_sigma", "zoom_strength",
                     "motion_preset", "motion_preset_path", "motion_strength",
                     "min_silent",
                     "sfx_dir", "sfx_volume", "sfx_volumes")

    def hash_essentials(self) -> str:
        """hash() minus the visual-only style flags (TTS/narration cache key)."""
        def _default(o):
            if isinstance(o, Path):
                return str(o)
            raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
        d = {k: v for k, v in asdict(self).items() if k not in self._STYLE_FIELDS}
        return _sha256_text(json.dumps(d, sort_keys=True, default=_default))


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


_NON_LEXICAL_RE = re.compile(r"^[\s.·•—–\-_*~…!?]*$")


def is_non_lexical(text: str | None) -> bool:
    """True when text carries no speakable words ("...", "—", "", "*").

    The vision model emits these for blank/silent panels; they must never
    reach TTS (a literal "..." was being spoken in narration.txt).
    """
    return bool(_NON_LEXICAL_RE.match(text or ""))


_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_CJK_RE = re.compile(r"[\u3040-\u30ff\uac00-\ud7af\u4e00-\u9fff]")


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text)) + len(_CJK_RE.findall(text))


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…\"'”’])\s+")


def estimate_speech_seconds(text: str, cfg: VideoConfig) -> float:
    """Rough speech length for budgeting (conservative: errs slow so real
    TTS audio lands inside the sub-5s window, never over it)."""
    return (_word_count(text) / cfg.speech_wpm) * 60.0 if text.strip() else 0.0


def fit_text_to_speech_window(text: str, cfg: VideoConfig) -> str:
    """Trim one panel's spoken text to whole sentences inside the window.

    Lines already under speech_max_seconds pass through untouched. Longer
    lines keep leading whole sentences up to speech_target_seconds. A single
    sentence that alone overruns the budget is NOT chopped mid-clause: per the
    natural-flow design we let the narrator finish the thought and the panel
    run to its absolute display ceiling (max_display_seconds). Only a runaway,
    sentence-boundary-free run-on longer than that ceiling is hard-cut to the
    max budget with terminal punctuation restored (TTS-friendly), so one
    pathological line can never pin a frame forever. Never returns non-lexical
    text.
    """
    if not cfg.speech_window or not text.strip():
        return text
    max_words = max(1, int(cfg.speech_max_seconds * cfg.speech_wpm / 60))
    # Whole-panel ceiling: a line short enough to be delivered inside
    # max_display_seconds is spoken COMPLETE -- we never trim or chop it, so
    # normal recap prose keeps every clause (the panel simply lasts as long as
    # the voice, with no lingering because there is no trailing gap). Only a
    # line past this ceiling is a candidate for trimming.
    hold_words = max(max_words,
                     int(cfg.max_display_seconds * cfg.speech_wpm / 60))
    if _word_count(text) <= hold_words:
        return text
    target_words = max(1, int(cfg.speech_target_seconds * cfg.speech_wpm / 60))
    sentences = [s for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    kept: list[str] = []
    kept_words = 0
    for s in sentences:
        w = _word_count(s)
        if kept and kept_words + w > target_words:
            break
        kept.append(s)
        kept_words += w
        if kept_words >= target_words:
            break
    if not kept:
        # No sentence boundary at all: treat the whole line as one run-on and
        # let the hold ceiling below decide whether it is shortenable.
        kept = [text.strip()]
        kept_words = _word_count(text.strip())
    if kept_words > hold_words:
        # A single run-on past the whole-panel ceiling (no usable sentence
        # break): hard-cut to the budget rather than hold one frame forever.
        kept = [" ".join(re.findall(r"\S+",
                                    " ".join(kept))[:max_words])]
    short = _normalise(" ".join(kept))
    return short if not is_non_lexical(short) else ""


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
    Non-lexical strings ("...", "—") are dropped entirely.
    """
    raw_narration = panel.narration if panel.narration else ""
    narration = _normalise(raw_narration) if not is_non_lexical(raw_narration) else ""
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


def _load_chapter_script(work_dir: Path) -> dict | None:
    """Load script.json (Phase 2.5 whole-chapter pass) when it holds lines."""
    path = work_dir / "script.json"
    if not path.is_file():
        return None
    try:
        import json as _json
        data = _json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and isinstance(data.get("lines"), list) \
            and data["lines"]:
        return data
    return None


def build_narration(artifact: CutArtifact, cfg: VideoConfig,
                    *, panels_hash: str, work_dir: Path | None = None,
                    panels_dir: Path | None = None) -> NarrationArtifact:
    """Build narration entries for the video.

    Text source priority:
      1. script.json (Phase 2.5 whole-chapter script pass) — real recap
         lines mapped onto panels; consecutive lines may skip panels.
      2. Per-panel captions (script_text) — legacy/offline path.

    The chapter script is looked for in ``work_dir`` (the output folder) and
    then beside ``panels.json`` (``panels_dir``): the script is a Phase 2.5
    sibling of the cut, so `guided video DIR/panels.json --out elsewhere/x.mp4`
    must still honour the persona pass rather than silently reverting to flat
    per-panel captions.

    Either way:
      * blank panels never get an entry;
      * a context_only (demoted bubble) panel never gets an entry of its OWN
        and never a frame, but its dialogue is folded onto the nearest scene
        panel so the story keeps hearing that line (voice-over carry-over);
      * non-lexical text ("...") never gets an entry;
      * a panel whose text is identical to the PREVIOUS spoken line gets
        no entry (the voice repeats nothing; the panel still appears in
        the video as a silent beat).
    Unspoken panels are NOT dropped from the timeline: the chapter script
    deliberately speaks fewer lines than there are panels, and removing
    the unspoken frames would visibly skip panels in the video. They stay
    as silent beats (min_display_seconds) between their narrated
    neighbours.
    """
    # Order by panel_index: panels_confirmed.json renumbers panel_index to
    # the user's confirmed order (Panel Review reordering must survive).
    # y_start is only a tiebreak for legacy artifacts with duplicate indices.
    panels = sorted(artifact.panels, key=lambda p: (p.panel_index, p.y_start))
    script = _load_chapter_script(work_dir) if work_dir is not None else None
    if script is None and panels_dir is not None:
        script = _load_chapter_script(panels_dir)
    by_id_line: dict[str, dict] = {}
    if script is not None:
        for ln in script.get("lines", []):
            pid = ln.get("panel_id")
            if isinstance(pid, str) and pid not in by_id_line:
                by_id_line[pid] = ln
        log.info("build_narration using script.json (%d lines over %d panels)",
                 len(script.get("lines", [])), len(panels))

    entries: list[NarrationEntry] = []
    prev_text = ""
    pending_lead = ""  # bubble speech seen before any scene frame to carry it on
    for order, p in enumerate(panels, start=1):
        if getattr(p, "blank_flag", "normal") == "blank":
            # blank crops are never narrated or spoken
            continue
        if getattr(p, "context_only", False):
            # Demoted bubble panel: it never gets its own frame (build_timeline
            # still skips it), but dropping it outright would silently cut a
            # line out of the story. Fold its own words onto the nearest scene
            # panel so the narrator still speaks them as a voice-over.
            # Speak the bubble's OWN words (its dialogue) — not the vision
            # model's descriptive caption ("two speech bubbles float in white
            # space"), which is meaningless as narration; fall back to the
            # narration line only when there is no dialogue. "/"-separated
            # bubbles are read as one continuous line.
            ctext = ""
            ln = by_id_line.get(p.id)
            if ln and (ln.get("text") or "").strip():
                ctext = _normalise(ln["text"])
            else:
                raw = (p.dialogue or "").strip() or (p.narration or "").strip()
                ctext = _normalise(re.sub(r"\s*/\s*", " ", raw))
            if any(ord(ch) > 127 for ch in ctext):
                # This folded text is appended to a NEIGHBOURING entry below,
                # so it never passes the spoken-line gate further down -- an
                # untranslated bubble here would be spoken verbatim by TTS.
                ctext = _normalise(strip_non_latin(ctext))
            if is_non_lexical(ctext) or not ctext.strip() or ctext == prev_text:
                continue
            if is_promo_text(ctext, p.dialogue, p.narration):
                # A demoted scanlation ad must stay SILENT, not be folded onto
                # the next scene panel: the voice-over carry-over exists for
                # real story dialogue, and "Read at SITE.COM" is not story.
                log.info("panel %s: promo card text dropped (never voiced "
                         "over)", p.id)
                continue
            if entries:
                last = entries[-1]
                entries[-1] = last.model_copy(
                    update={"text": f"{last.text} {ctext}".strip()})
            else:
                pending_lead = f"{pending_lead} {ctext}".strip()
            continue
        if script is not None:
            # Phase 2.5 mapping: only panels the scriptwriter assigned a
            # line to get narration. Panels without a line stay silent
            # visuals — or get dropped by build_timeline when silent.
            ln = by_id_line.get(p.id)
            text = (ln or {}).get("text", "") or ""
            text = _normalise(text) if text.strip() else ""
            text = fit_text_to_speech_window(text, cfg)
            if is_non_lexical(text):
                text = ""
            if text and text == prev_text:
                text = ""            # never speak the same line twice
            if text:
                prev_text = text
            quotes = [((ln or {}).get("quote") or "").strip()] if \
                (ln or {}).get("quote") else []
        else:
            text = script_text(p, include_dialogue=cfg.include_dialogue)
            text = fit_text_to_speech_window(text, cfg)
            if text and text == prev_text:
                continue             # duplicate caption: no entry at all
            if text:
                prev_text = text
            quotes = [q.strip() for q in re.findall(r"[\"“]([^\"”]+)[\"”]",
                                                    p.dialogue or "")]
        # English-only at the LAST gate before TTS. Prompt-time cleaning only
        # guards what goes INTO the model; a script written before that contract
        # can still carry raw Hangul syllables or even an emoji (U+1F3B5) inside
        # a spoken line, and edge-tts reads those literally. Stripping here makes
        # already-narrated chapters safe without re-spending a vision call.
        if text and any(ord(c) > 127 for c in text):
            stripped = _normalise(strip_non_latin(text))
            log.info("panel %s: stripped %d non-Latin char(s) from the spoken "
                     "line", p.id, sum(1 for c in text if ord(c) > 127))
            text = "" if is_non_lexical(stripped) else stripped
        if quotes:
            quotes = [q for q in (strip_non_latin(q).strip() for q in quotes)
                      if q and not is_non_lexical(q)]
        if text and is_promo_text(text, *quotes):
            # The script pass can turn an ad panel into an ad LINE ("Read the
            # full story at asurascans.com"), and a legacy caption can carry the
            # URL itself. Panel demotion removes the frame; this removes the
            # voice, otherwise the promo is still spoken over real art.
            log.info("panel %s: narration line advertises a scan site; left "
                     "unspoken", p.id)
            text = ""
            quotes = []
        if pending_lead:
            # a bubble panel that opened the chapter (no preceding scene):
            # voice it over the first real scene frame we reach.
            text = f"{pending_lead} {text}".strip()
            pending_lead = ""
        entries.append(NarrationEntry(id=p.id, panel_id=p.id, order=order,
                                       speaker=None, text=text, quotes=quotes))
    result = NarrationArtifact(
        meta=_meta(cfg.hash_essentials(), {"panels.json": panels_hash}),
        mode="narrator", entries=entries)
    log.info("build_narration entries=%d spoken=%d",
             len(entries), sum(1 for e in entries if e.text.strip()))
    return result


# --------------------------------------------------------------------------- #
# Stage 2 — TTS
# --------------------------------------------------------------------------- #
def _clips_to_synthesize(narration: NarrationArtifact) -> list[str]:
    """Ids of narration entries that get their own TTS clip.

    The ONE source of truth for which clips are expected (duplicates of the
    previous line and empty texts are skipped exactly as in the synth loop),
    so the cache can tell a COMPLETE artifact from a partially failed one.
    """
    ids: list[str] = []
    prev_text = ""
    for e in narration.entries:
        text = e.text.strip()
        if not text or text == prev_text:
            continue
        prev_text = text
        ids.append(e.id)
    return ids


def fit_narration_clip(entry: AudioEntry, audio_dir: Path,
                       cfg: VideoConfig, probe) -> AudioEntry:
    """Enforce the speech budget on one MEASURED TTS clip.

    The text trim keeps normal narration inside speech_max_seconds, but the
    budget is an estimate: a slow voice (or a stale cached clip) can measure
    over it. The clip file is tempo-adjusted with ffmpeg atempo (pitch
    preserved) by exactly the overrun factor, bounded by
    speech_fit_max_speedup; word timestamps are scaled to stay in sync so
    SRT cues do not drift. If even max speed cannot fit, the tail is
    trimmed at the budget with a loud warning (pathological audio only).
    Returns the entry unchanged when it already fits.
    """
    if not cfg.speech_window or not cfg.speech_fit_audio:
        return entry
    budget = cfg.speech_max_seconds
    dur = entry.duration_seconds
    if dur <= budget or dur <= 0:
        return entry
    path = audio_dir / entry.path
    if not path.is_file():
        log.warning("narration fit skipped %s: clip file missing",
                    entry.entry_id)
        return entry
    ffmpeg_exe = _resolve_ffmpeg(cfg.ffmpeg_exe)
    factor = min(dur / budget, cfg.speech_fit_max_speedup)
    tmp = path.with_name(f"{path.stem}.fit{path.suffix}")
    cmd = [ffmpeg_exe, "-y", "-i", str(path),
           "-filter:a", f"atempo={factor:.6f}", str(tmp)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              check=False, shell=False, timeout=120)
        stderr = proc.stderr or ""
        if proc.returncode != 0 or not tmp.is_file():
            tmp.unlink(missing_ok=True)
            log.warning("narration fit FAILED %s (%.2fs > %.2fs budget): %s",
                        entry.entry_id, dur, budget, stderr.strip()[-200:])
            return entry
        new_dur = probe(tmp)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        tmp.unlink(missing_ok=True)
        log.warning("narration fit FAILED %s (%.2fs > %.2fs budget): %s",
                    entry.entry_id, dur, budget, exc)
        return entry
    trimmed = False
    if new_dur > budget:
        # even max speedup cannot fit (pathological audio): trim the tail
        trim_tmp = path.with_name(f"{path.stem}.trim{path.suffix}")
        cmd = [ffmpeg_exe, "-y", "-i", str(tmp), "-t", f"{budget:.3f}",
               str(trim_tmp)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  check=False, shell=False, timeout=120)
            if proc.returncode == 0 and trim_tmp.is_file():
                trimmed = True
                new_dur = probe(trim_tmp)
                log.warning(
                    "narration %s still %.2fs at max speedup %.2fx — trimmed "
                    "to %.2fs (audio was pathological; check the voice or "
                    "the speech_wpm estimate)", entry.entry_id, dur / factor,
                    factor, new_dur)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            log.warning("narration fit trim FAILED %s: %s",
                        entry.entry_id, exc)
        finally:
            if trimmed:
                trim_tmp.replace(path)
            else:
                trim_tmp.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
        if not trimmed:
            return entry
    else:
        tmp.replace(path)
    scale = 1.0 / factor
    scaled_words = [
        {**w, "start": round(float(w.get("start", 0.0)) * scale, 3),
         "end": round(float(w.get("end", 0.0)) * scale, 3)}
        for w in entry.words]
    if trimmed:
        scaled_words = [w for w in scaled_words
                        if float(w.get("start", 0.0)) < new_dur]
    log.info("narration fit %s: %.2fs -> %.2fs (atempo %.2fx%s)",
             entry.entry_id, dur, new_dur, factor,
             ", trimmed" if trimmed else "")
    return AudioEntry(entry_id=entry.entry_id, path=entry.path,
                      duration_seconds=new_dur, words=scaled_words)


def synthesize_audio(narration: NarrationArtifact, audio_dir: Path,
                     cfg: VideoConfig, *, force: bool = False,
                     retries: int = 3) -> AudioArtifact:
    """One mp3/wav per non-empty entry.

    audio.json is only a cache hit when the hashes match AND a usable clip
    file exists for every expected entry: an artifact left behind by a
    transient TTS failure is repaired (missing clips re-synthesized, the rest
    reused) instead of being reused forever. --force re-synthesizes all.
    """
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
        return AudioArtifact(meta=_meta(cfg.hash_essentials(), input_hashes),
                             voice="none", entries=[])

    expected_ids = _clips_to_synthesize(narration)

    prev: AudioArtifact | None = None
    if sidecar.exists() and not force:
        try:
            prev = AudioArtifact.model_validate_json(sidecar.read_text("utf-8"))
        except Exception as exc:  # noqa: BLE001 - stale/corrupt cache
            log.debug("ignoring unreadable audio.json: %s", exc)
            prev = None
    # A cache hit needs the SAME inputs AND a clip file for EVERY expected
    # entry. Writing audio.json after a transient TTS failure used to poison
    # the cache forever (the degraded artifact was reused until --force).
    same_inputs = (
        prev is not None
        and prev.meta.config_hash == cfg.hash_essentials()
        and prev.meta.input_hashes == input_hashes
    )
    reusable: dict[str, AudioEntry] = {}
    if same_inputs and prev is not None:
        reusable = {e.entry_id: e for e in prev.entries
                    if (audio_dir / e.path).is_file()}
        missing = [i for i in expected_ids if i not in reusable]
        if not missing:
            # Re-apply the speech-budget fit even on a full cache hit. A clip
            # cached by an earlier run — or one whose fit was skipped because
            # ffmpeg was momentarily unavailable — can still measure over
            # speech_max_seconds, which would push its panel past the sub-5s
            # window. fit_narration_clip is a no-op for clips already inside
            # the budget (no subprocess), so this only touches stragglers.
            def _cache_probe(p: Path) -> float:
                return probe_duration(p, cfg.ffprobe_exe)
            fitted = [fit_narration_clip(e, audio_dir, cfg, _cache_probe)
                      for e in prev.entries]
            log.info("audio cache hit (%d clips)", len(prev.entries))
            return prev.model_copy(update={"entries": fitted})
        # Per-clip resume: only the missing clips are re-synthesized.
        log.warning(
            "audio.json is INCOMPLETE (%d of %d clips usable, missing: %s); "
            "synthesizing the missing clips instead of reusing the degraded "
            "artifact", len(expected_ids) - len(missing), len(expected_ids),
            ", ".join(missing[:8]) or "?")

    try:
        from adapters.tts import synthesize_entry as tts_synth
    except ImportError as exc:
        raise VideoError(f"TTS provider not available: {exc}") from exc

    entries: list[AudioEntry] = []
    total = len(expected_ids)
    done = 0
    if _HAS_RICH and not EMBEDDED_MODE:
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
        def probe_fn(p: Path) -> float:
            return probe_duration(p, cfg.ffprobe_exe)

        # Serial planning pass (cheap): drop empty/duplicate text, reuse
        # cached clips. The remaining entries are independent — each writes
        # its own file — so the network-bound synthesis can run concurrently.
        # `order` keeps the final entries sequential regardless of the order
        # in which the concurrent results arrive.
        to_synth: list[NarrationEntry] = []
        order: list[str] = []
        by_id: dict[str, AudioEntry] = {}
        prev_text = ""
        for e in narration.entries:
            text = e.text.strip()
            if not text:
                continue
            if text == prev_text:
                log.info("TTS skipped %s: duplicate of previous narration", e.id)
                continue
            prev_text = text
            order.append(e.id)
            hit = reusable.pop(e.id, None)
            if hit is not None:
                by_id[e.id] = fit_narration_clip(hit, audio_dir, cfg,
                                                 probe_fn)
                done += 1
                log.info("tts %d/%d  %s  %.2fs (reused cached clip)",
                         done, total, e.id, hit.duration_seconds)
                if progress_ctx is not None and task_id is not None:
                    progress_ctx.update(task_id, advance=1)
                continue
            to_synth.append(e)

        if to_synth:
            conc = max(1, getattr(cfg, "tts_concurrency", 6))
            log.info("tts %d clips via thread pool (concurrency=%d)",
                     len(to_synth), conc)
            with ThreadPoolExecutor(max_workers=conc,
                                    thread_name_prefix="tts") as pool:
                results = list(pool.map(
                    lambda ent: (ent, tts_synth(
                        ent, audio_dir, provider=cfg.tts, voice=cfg.voice,
                        rate=cfg.rate, pitch=cfg.pitch, speed=cfg.speed,
                        probe_duration=lambda p: probe_duration(p, cfg.ffprobe_exe),
                        kokoro_model_path=cfg.kokoro_model_path,
                        kokoro_voices_path=cfg.kokoro_voices_path,
                        retries=retries)),
                    to_synth))
            for ent, (out, err) in results:
                if err is not None or out is None:
                    log.warning("TTS skipped %s: %s", ent.id, err or "empty text")
                    continue
                by_id[ent.id] = fit_narration_clip(out, audio_dir, cfg,
                                                   probe_fn)
                done += 1
                log.info("tts %d/%d  %s  %.2fs", done, total, ent.id,
                         out.duration_seconds)
                if progress_ctx is not None and task_id is not None:
                    progress_ctx.update(task_id, advance=1)

        # Emit in narration order, not completion order.
        entries = [by_id[pid] for pid in order if pid in by_id]
    finally:
        if progress_ctx is not None:
            progress_ctx.stop()

    artifact = AudioArtifact(meta=_meta(cfg.hash_essentials(), input_hashes),
                             voice=cfg.voice, entries=entries)
    _write_atomic(sidecar, artifact.model_dump_json(indent=2) + "\n")
    missing_after = [i for i in expected_ids if i not in {a.entry_id for a in entries}]
    if missing_after:
        log.warning(
            "audio.json written INCOMPLETE (%d of %d clips): %s failed TTS. "
            "The next run retries exactly those clips (no --force needed).",
            len(entries), total, ", ".join(missing_after[:8]) or "?")
    return artifact


# --------------------------------------------------------------------------- #
# Stage 3 — timeline
# --------------------------------------------------------------------------- #
def compute_pan(width: int, height: int,
                canvas_w: int = WIDTH, canvas_h: int = HEIGHT,
                *, blur_background: bool = False) -> PanSpec:
    """Scale the panel so BOTH dims cover the canvas; pan along the overflow.

    Canvas defaults to 1080x1920 (9:16). Pass landscape dimensions (e.g.
    1920x1080) for a 16:9 edit -- the overflow axis flips, so wide panels
    become pan_down candidates. Never centre-crops away content: a tall panel
    is read top->bottom (pan_down), a wide one left->right (pan_right).
    Exact-fit is static.

    With ``blur_background`` the panel is contain-fitted instead (the whole
    panel stays visible floating on the blurred background), so there is no
    overflow to pan through: the spec is static and travel_px is 0. That
    also keeps display_seconds from padding duration for a pan that would
    never be rendered.
    """
    if width <= 0 or height <= 0:
        raise ValueError("panel must have positive size")
    if blur_background:
        scale = min(canvas_w / width, canvas_h / height)
        if scale > 4.0:
            scale = 4.0
        return PanSpec(kind="static",
                       scaled_w=max(1, math.ceil(width * scale)),
                       scaled_h=max(1, math.ceil(height * scale)),
                       travel_px=0)
    scale = max(canvas_w / width, canvas_h / height)
    if scale > 4.0:
        scale = 4.0
    scaled_w = math.ceil(width * scale)
    scaled_h = math.ceil(height * scale)
    over_h, over_w = scaled_h - canvas_h, scaled_w - canvas_w
    if over_h > 2 and over_h >= over_w:
        return PanSpec(kind="pan_down", scaled_w=scaled_w, scaled_h=scaled_h,
                       travel_px=over_h)
    if over_w > 2:
        return PanSpec(kind="pan_right", scaled_w=scaled_w, scaled_h=scaled_h,
                       travel_px=over_w)
    return PanSpec(kind="static", scaled_w=scaled_w, scaled_h=scaled_h, travel_px=0)


def _round_ms_up(x: float) -> float:
    """Round up to millisecond precision.

    Panel durations are LOWER bounds: narration must finish and a pan must
    stay readable at max_pan_px_per_sec. Truncating with round() can shave a
    sub-millisecond off an exact pan floor and silently violate that
    invariant (observed: 45000/1350 = 33.3333 rendered as 33.333, dropping
    below the readability floor). The inner round(..., 6) strips binary
    float noise so genuinely exact ms values are left untouched.
    """
    return math.ceil(round(x * 1000.0, 6)) / 1000.0


def _speech_cap_seconds(cfg: VideoConfig,
                        audio_seconds: float | None) -> float | None:
    """Backstop for the natural-flow speech window (None when disabled).

    The cap sits at speech_max_seconds + gap but never below the measured
    audio + gap: narration must always finish; only trailing silence, pan
    padding and class-mult inflation get trimmed. The renderer plays each
    pan inside the (possibly shortened) window via t/dur, so the camera move
    stays in sync with the narration. Narration is never truncated or tempo-
    shifted by default (gap_seconds is 0 and speech_fit_audio is off), so a
    spoken panel lasts exactly its audio; the sub-5s result comes from
    trimming the TEXT up front, not from touching the measured voice.
    """
    if not cfg.speech_window:
        return None
    cap = cfg.speech_max_seconds + cfg.gap_seconds
    if audio_seconds is not None:
        cap = max(cap, audio_seconds + cfg.gap_seconds)
    return cap


def display_seconds(*, audio_seconds: float | None, words: int,
                    travel_px: int, cfg: VideoConfig,
                    panel_class: str = "calm") -> float:
    """How long a panel stays on screen (INCLUDING its trailing gap).

    panel_class (action/reveal/dialogue/calm from cinematic_effects)
    modulates pacing the way recap channels do:
      * action   — fast cuts; may go BELOW min_display (0.8s floor), so
                   rapid panels keep energy even when narration is short;
      * reveal   — held beats: +35% over the audio floor for the moment
                   to land;
      * dialogue — neutral (the voice already sets the pace);
      * calm     — the base behaviour (audio/gap/min floor).
    Pan floor always applies (a pan must stay readable at any class).
    """
    pan_floor = travel_px / cfg.max_pan_px_per_sec if travel_px else 0.0
    mult = cast(dict[str, float], cfg.class_duration_multiplier).get(
        panel_class, cast(dict[str, float], cfg.class_duration_multiplier).get("calm", 1.0))
    if audio_seconds is not None:
        # spoken panel: narration must finish; never capped below the audio
        base = audio_seconds + cfg.gap_seconds
        if cfg.pan_fit_speech and travel_px and pan_floor > base:
            # The pan would still be running after the narrator stops.
            # Speed it up (bounded) so it lands inside the speech window
            # rather than holding silent frames on a static tail.
            max_speed = cfg.max_pan_px_per_sec * cfg.pan_fit_speech_speedup
            pan_floor = max(base, travel_px / max_speed)
        if panel_class == "action":
            # action floor is lower: fast cuts read as energy, not as
            # truncation, once the voice has finished
            floor = min(cfg.min_display_seconds,
                        cfg.action_floor_seconds)
        else:
            floor = cfg.min_display_seconds
        dur = max(base, floor, pan_floor) * mult
        cap = _speech_cap_seconds(cfg, audio_seconds)
        if cap is not None:
            dur = min(dur, cap)
        return _round_ms_up(dur)
    # silent panel: reading-speed heuristic (no audio => no drift possible);
    # the floor is min_silent so un-narrated montage beats stay short even
    # when --min-display is generous
    read = (words / cfg.silent_wpm) * 60.0 if words else 0.0
    dur = min(max(read + cfg.gap_seconds, cfg.min_silent),
              cfg.max_display_seconds)
    cap = _speech_cap_seconds(cfg, None)
    if cap is not None:
        # speech window: same rule as the spoken branch — the pan plays
        # inside the (possibly shortened) window via t/dur, so neither the
        # pan floor nor the class multiplier may outlast the cap
        dur = min(max(dur, pan_floor) * mult, cap)
    else:
        # legacy behaviour: pan floor always wins, applied after the cap
        dur = max(dur, pan_floor) * mult
    return _round_ms_up(dur)


def _cap_filler_travel(motion: dict, cfg: VideoConfig) -> None:
    """Keep an un-narrated beat short AND its pan readable.

    display_seconds() floors a silent panel at min_silent, but the pan floor
    (travel / max_pan_px_per_sec) can outlast it. With the closer 2.0/2.2x
    framing the reveal is ~670px == a 1.5s floor, so a filler crop would either
    inflate past min_silent (the dead air we removed) or, held at 1.0s, sweep
    the panel at 670px/s (unreadable). Trimming the travel to what fits the
    hold honours both contracts. Narrated panels are never touched: their audio
    sets the duration.
    """
    travel = int(motion.get("travel_px") or 0)
    allowed = int(cfg.min_silent * cfg.max_pan_px_per_sec)
    if travel <= allowed or allowed <= 0:
        return
    factor = allowed / travel
    motion["travel_px"] = allowed
    # The renderer reads pan_x_px/pan_y_px as the final travel, so they must be
    # rescaled with the cap or the pan floor and the picture would disagree.
    motion["pan_x_px"] = round(float(motion.get("pan_x_px", 0.0) or 0.0)
                               * factor, 3)
    motion["pan_y_px"] = round(float(motion.get("pan_y_px", 0.0) or 0.0)
                               * factor, 3)


def _load_motion_preset(path: Path | str | None):  # type: ignore[no-untyped-def]
    """Import motion_presets robustly (script-dir vs installed-package CWD).

    pytest and `recap-comic` run with the repo root on sys.path, but a
    `python /tmp/script.py` invocation puts the script dir first instead —
    fall back to the sibling file next to this module so the preset layer
    never depends on the caller's CWD.
    """
    try:
        import motion_presets as _mp  # type: ignore[import-not-found]
        return _mp.load_preset(path)
    except ModuleNotFoundError:
        import importlib.util as _ilu
        import sys as _sys
        sibling = Path(__file__).resolve().parent / "motion_presets.py"
        spec = _ilu.spec_from_file_location("motion_presets", sibling)
        if spec is None or spec.loader is None:
            raise
        mod = _ilu.module_from_spec(spec)
        _sys.modules.setdefault("motion_presets", mod)
        spec.loader.exec_module(mod)
        return mod.load_preset(path)


def _motion_module():  # type: ignore[no-untyped-def]
    try:
        import motion_presets as _mp  # type: ignore[import-not-found]
        return _mp
    except ModuleNotFoundError:
        import importlib.util as _ilu
        import sys as _sys
        sibling = Path(__file__).resolve().parent / "motion_presets.py"
        spec = _ilu.spec_from_file_location("motion_presets", sibling)
        if spec is None or spec.loader is None:
            raise
        mod = _ilu.module_from_spec(spec)
        _sys.modules.setdefault("motion_presets", mod)
        spec.loader.exec_module(mod)
        return mod


def _classify(p: CutPanel) -> str:
    """Panel class for pacing (action/reveal/dialogue/calm).

    cinematic_effects.classify_panel reads narration+dialogue text with
    pure regex — deterministic, no AI, no new dependency in the render.
    """
    try:
        from cinematic_effects import classify_panel
        return classify_panel({"narration": p.narration or "",
                               "dialogue": p.dialogue or ""})
    except Exception:  # noqa: BLE001 - pacing hint must never kill render
        return "calm"


def build_timeline(artifact: CutArtifact, panels_dir: Path,
                   narration: NarrationArtifact, audio: AudioArtifact,
                   audio_dir: Path, cfg: VideoConfig,
                   *, panels_hash: str,
                   canvas_w: int = WIDTH, canvas_h: int = HEIGHT) -> TimelineArtifact:
    """Assemble the timeline.

    canvas_w/canvas_h select the output canvas (default 1080x1920 portrait;
    pass 1920x1080 for a 16:9 landscape edit). The canvas drives both pan
    geometry and the TimelineArtifact's declared dimensions, which the
    renderer honours.

    With ``motion_preset="reference"`` the camera plan comes from the
    normalized reference template (motion_presets.py) instead of the
    heuristic compute_pan: segment order, relative timing weights, zoom
    targets and pan direction/speed are reproduced per shot, clamped to
    each panel's safe range. Durations scale proportionally to the measured
    narration total so a short reference shot stays short relative to a
    long one while narration never truncates.
    """
    by_audio = {a.entry_id: a for a in audio.entries}
    by_text = {n.id: n for n in narration.entries}
    # First pass: collect usable panels (same skip contract as default).
    usable: list[tuple[Any, Path, int, int, str, Any]] = []
    skipped: list[dict] = []
    for _order, p in enumerate(sorted(artifact.panels,
                                      key=lambda p: (p.panel_index, p.y_start)), start=1):
        h = p.y_end - p.y_start
        if h <= 0:
            log.warning("skipping zero-height panel %s", p.id)
            skipped.append({"panel_id": p.id, "reason": "zero_height"})
            continue
        if getattr(p, "blank_flag", "normal") == "blank":
            # deterministic blank detector marked this crop empty; it must
            # not reach narration/TTS/render unless the user kept it.
            log.info("skipping panel %s in timeline: blank_flag=blank "
                     "(score %.2f)", p.id, getattr(p, "blank_score", 0.0))
            skipped.append({"panel_id": p.id, "reason": "blank"})
            continue
        if getattr(p, "context_only", False):
            # panel_filter demoted this text-only panel: no video frame.
            # Its dialogue is not lost — build_narration already folded it
            # onto the nearest scene panel's line (voice-over carry-over).
            log.info("skipping panel %s in timeline: context_only "
                     "(dialogue voiced over a neighbouring scene)", p.id)
            skipped.append({"panel_id": p.id, "reason": "context_only"})
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
            skipped.append({"panel_id": p.id, "reason": "image_missing"})
            continue
        a = by_audio.get(p.id)
        text = by_text[p.id].text if p.id in by_text else ""
        if not text.strip() and a is None:
            # Two contracts meet here. The chapter script deliberately speaks
            # fewer lines than there are panels ("skip panels that add
            # nothing"), so an unspoken panel may either stay as a short silent
            # beat (min_silent > 0: the whole chapter is visible but the voice
            # pauses) or be dropped (min_silent <= 0: the timeline is
            # speech-driven, so the narrator never stops and no shot is rushed
            # through a filler hold). The filler choice is only sane for a beat
            # or two; a chapter whose script covers 11 of 68 panels turns it
            # into minute-long silences and 1s pans.
            if cfg.min_silent > 0:
                log.info("panel %s: no narration/audio -> silent filler beat "
                         "(kept on the timeline)", p.id)
                skipped.append({"panel_id": p.id,
                                "reason": "no_text_no_audio_silent_beat"})
                # fall through: the panel stays usable with no audio
            else:
                log.info("skipping panel %s in timeline: no narration, no "
                         "audio (dead-air drop)", p.id)
                skipped.append({"panel_id": p.id,
                                "reason": "no_text_no_audio_dead_air_drop"})
                continue
        # Pan geometry must match the PNG on disk, NOT the source-strip
        # geometry: the cutter may normalize panel PNGs (390x[760,800]
        # center-crop/pad) while y_start/y_end/strip width remain source
        # coordinates. Scaling a 390x800 PNG to source-derived scaled_h
        # would stretch/crop wrongly and invent pan travel_px that the
        # image does not have (inflating display_seconds via pan_floor).
        png_w = p.output_width
        png_h = p.output_height
        if png_w is None or png_h is None:
            # Legacy full-resolution crops (normalize_output=False): the
            # PNG dimensions are the source crop dimensions.
            png_w = p.strip_width or artifact.width
            png_h = h
        usable.append((p, img, png_w, png_h, text, a))

    if not usable and not skipped:
        raise VideoError("no usable panels in panels.json")

    preset = None
    seg_indices: list[int] = []
    _mp = None
    if (cfg.motion_preset or "none") != "none":
        try:
            _mp = _motion_module()
            preset = _load_motion_preset(cfg.motion_preset_path)
            seg_indices = _mp.map_segments_to_panels(len(usable), preset)
        except Exception as exc:
            raise VideoError(
                f"motion_preset {cfg.motion_preset!r} failed to load: {exc}"
            ) from exc

    entries: list[TimelineEntry] = []
    t = 0.0
    if preset is None:
        for order, (p, img, png_w, png_h, text, a) in enumerate(usable, start=1):
            pan = compute_pan(png_w, png_h, canvas_w, canvas_h,
                              blur_background=cfg.blur_background)
            dur = display_seconds(
                audio_seconds=a.duration_seconds if a else None,
                words=_word_count(text), travel_px=pan.travel_px, cfg=cfg,
                panel_class=_classify(p))
            entries.append(TimelineEntry(
                panel_id=p.id, order=order, source_image=str(img),
                bbox=BBox(x=0, y=p.y_start, w=png_w, h=png_h),
                start_seconds=round(t, 3), duration_seconds=dur,
                audio_path=str((audio_dir / a.path).resolve()) if a else None,
                pan=pan))
            t += dur
    else:
        assert preset is not None and preset.segments is not None
        assert _mp is not None
        # Base durations first (narration must finish; pan floor applies).
        # panel_class is forced to calm so the per-class multiplier does not
        # distort the reference rhythm (the template is the pacing source).
        bases: list[float] = []
        resolved: list[dict] = []
        for (_p, _img, png_w, png_h, text, a), si in zip(
                usable, seg_indices, strict=True):
            seg = (preset.segments or [])[si]
            r = _mp.resolve_for_panel(
                png_w=png_w, png_h=png_h, canvas_w=canvas_w,
                canvas_h=canvas_h, seg=seg, preset=preset,
                motion_strength=cfg.motion_strength,
                blur_background=cfg.blur_background)
            if not text.strip() and a is None:
                # Un-narrated filler beat: shorten the reveal to fit the hold
                # instead of stretching the hold to fit the reveal.
                _cap_filler_travel(r, cfg)
            resolved.append(r)
            pan_probe = PanSpec(kind=r["kind"], scaled_w=r["scaled_w"],
                                scaled_h=r["scaled_h"],
                                travel_px=r["travel_px"])
            base = display_seconds(
                audio_seconds=a.duration_seconds if a else None,
                words=_word_count(text), travel_px=pan_probe.travel_px,
                cfg=cfg, panel_class="calm")
            bases.append(base)
        total_base = sum(bases) or 1.0
        total_ref = sum((preset.segments or [])[si].dur
                        for si in seg_indices) or 1.0
        scale = total_base / total_ref
        split_seen = 0  # ordinal of split panels, to alternate band-pan dirs
        for order, ((p, img, png_w, png_h, text, a), si, r, base) in enumerate(
                zip(usable, seg_indices, resolved, bases, strict=True), start=1):
            seg = (preset.segments or [])[si]
            scaled_ref = seg.dur * scale
            if a is not None and a.duration_seconds:
                # Narration is the clock: a spoken panel must advance to the
                # next one the instant its line's audio ends. The reference
                # template still shapes the camera MOVE inside the window (via
                # t/dur), but it must never STRETCH a spoken panel past its
                # measured audio — that is the audible "long pause" where the
                # voice has stopped but the shot holds to hit the template
                # beat. base is already audio-driven (audio + gap, with only
                # the small min-display/pan floors as a lower bound). Silent
                # beats (no audio) keep following the preset pace.
                dur = base
            elif not text.strip():
                # Narration-less montage beat (the chapter script skipped
                # this panel): fixed short hold from min_silent, never
                # stretched by the reference template pacing — otherwise
                # filler crops would bloat the runtime.
                dur = _round_ms_up(base)
            else:
                dur = _round_ms_up(max(base, scaled_ref))
            cap = _speech_cap_seconds(
                cfg, a.duration_seconds if a else None)
            if cap is not None:
                dur = min(dur, _round_ms_up(cap))
            pan = PanSpec(kind=r["kind"], scaled_w=r["scaled_w"],
                          scaled_h=r["scaled_h"], travel_px=r["travel_px"])
            # Split panels pan their two bands in opposite directions and the
            # pair flips on every successive split panel (+1, -1, +1, ...).
            scols = int(r.get("split_columns", 0) or 0)
            if scols >= 2:
                split_dir = 1 if (split_seen % 2 == 0) else -1
                split_seen += 1
            else:
                split_dir = int(r.get("split_dir", 1) or 1)
            motion = {
                "preset": preset.name,
                "seg": seg.seg,
                "duration": seg.dur,
                "zoom": seg.zoom,
                "zoom_strength": r["zoom_strength"],
                "dx": seg.dx,
                "dy": seg.dy,
                "dxps": seg.dxps,
                "dyps": seg.dyps,
                "ndx": r["ndx"],
                "ndy": r["ndy"],
                "ndxps": r["ndxps"],
                "ndyps": r["ndyps"],
                "matchScore": seg.matchScore,
                "confidence": seg.matchScore,
                "damping": r["damping"],
                "static": r["static"],
                "pan_x_px": r["pan_x_px"],
                "pan_y_px": r["pan_y_px"],
                "tall_panel": r.get("tall_panel", False),
                "panel_scale": r.get("panel_scale", 1.0),
                "split_columns": scols,
                "split_dir": split_dir,
                "split_gap_frac": r.get("split_gap_frac", 0.05),
                "split_pan_frac": r.get("split_pan_frac", 0.15),
                "split_ss": r.get("split_ss", 3.0),
                # How much of the closer crop this shot actually sweeps: the
                # renderer multiplies its pan travel by it, which decouples
                # framing tightness from camera speed.
                "pan_travel_frac": r.get("pan_travel_frac", 1.0),
                "rhythm_weight": seg.dur / total_ref,
                "scaled_ref_seconds": round(scaled_ref, 3),
                "source_panel": p.id,
            }
            entries.append(TimelineEntry(
                panel_id=p.id, order=order, source_image=str(img),
                bbox=BBox(x=0, y=p.y_start, w=png_w, h=png_h),
                start_seconds=round(t, 3), duration_seconds=dur,
                audio_path=str((audio_dir / a.path).resolve()) if a else None,
                pan=pan, motion=motion))
            t += dur
        try:
            log.info("motion_preset=%s segments=%s\n%s",
                     preset.name, seg_indices,
                     _mp.preview_table(preset, seg_indices))
        except Exception:
            log.debug("motion preview table failed", exc_info=True)
    if not entries:
        raise VideoError("no usable panels in panels.json")
    result = TimelineArtifact(
        meta=_meta(cfg.hash(), {
            "panels.json": panels_hash,
            "narration.json": _sha256_text(narration.model_dump_json()),
            "audio.json": _sha256_text(audio.model_dump_json())}),
        width=canvas_w, height=canvas_h, fps=cfg.fps,
        gap_seconds=cfg.gap_seconds,
        min_display_seconds=cfg.min_display_seconds, entries=entries,
        skipped_panels=skipped)
    log.info("build_timeline entries=%d skipped=%d total_duration=%.2fs",
             len(entries), len(skipped), total_seconds(result))
    return result


def total_seconds(timeline: TimelineArtifact) -> float:
    return round(sum(e.duration_seconds for e in timeline.entries), 3)


def build_motion_report(timeline: TimelineArtifact) -> list[dict[str, Any]]:
    """Debug/preview rows for the reference-motion preset.

    One row per timeline entry: selected preset, segment number, duration,
    zoom, dx/dy, speed, normalized movement, source panel, confidence.
    Empty list when the default automation path was used.
    """
    rows: list[dict[str, Any]] = []
    for e in timeline.entries:
        m = getattr(e, "motion", None) or {}
        if not m:
            continue
        rows.append({
            "panel_id": e.panel_id,
            "order": e.order,
            "preset": m.get("preset"),
            "seg": m.get("seg"),
            "duration_seconds": e.duration_seconds,
            "reference_dur": m.get("duration"),
            "zoom": m.get("zoom"),
            "zoom_strength": m.get("zoom_strength"),
            "dx": m.get("dx"),
            "dy": m.get("dy"),
            "dxps": m.get("dxps"),
            "dyps": m.get("dyps"),
            "ndx": m.get("ndx"),
            "ndy": m.get("ndy"),
            "pan_kind": e.pan.kind,
            "travel_px": e.pan.travel_px,
            "pan_x_px": m.get("pan_x_px"),
            "pan_y_px": m.get("pan_y_px"),
            "confidence": m.get("confidence", m.get("matchScore")),
            "static": m.get("static"),
        })
    return rows


# --------------------------------------------------------------------------- #
# Stage 3c — SFX plan (opt-in: cfg.sfx_dir)
# --------------------------------------------------------------------------- #
_SFX_AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".m4a", ".flac"}
SFX_KINDS = ("transition", "action", "reveal")

# Action keywords that earn a precise, word-timed impact sound. Deliberately
# regex (not LLM): deterministic, offline, cache-stable, and panel classes
# from cinematic_effects.classify_panel already gate when it even applies.
_SFX_ACTION_RE = re.compile(
    r"\b(swung|swing|swings|drew|draws|unsheathe|unsheathed|unsheathes|"
    r"slash|slashes|slashed|stab|stabs|stabbed|punch|punches|punched|"
    r"slam|slams|slammed|smash|smashes|smashed|crash|crashes|crashed|"
    r"shatter|shatters|shattered|exploded|explosion|blast|blasted|"
    r"boom|thunder|roar|roared|scream|screamed|shriek|shrieked|"
    r"clang|clanged|struck|strike|impact)\b", re.IGNORECASE)


def _sfx_pick(pool: list[Path], seed: str) -> Path:
    """Deterministic bank pick: same panel+kind+ordinal -> same file on every
    run (the mixdown must be reproducible for the video cache)."""
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return pool[int(h[:8], 16) % len(pool)]


def _sfx_source_ref(sound: Path, bank_dir: Path) -> str:
    """Store the plan's source RELATIVE to the bank root, keeping the category
    subfolder (e.g. ``transition/whoosh.wav``), so ``mix_sfx`` can resolve
    ``bank_dir / source``. A bare basename would drop the subfolder and miss
    the file entirely — ``load_sfx_bank`` keeps sounds under ``<bank>/<kind>/``
    — which silently drops every SFX event at mixdown."""
    try:
        return sound.resolve().relative_to(bank_dir.resolve()).as_posix()
    except ValueError:
        return sound.name


def _sfx_volume_for(cfg: VideoConfig, kind: str) -> float:
    mults = cast(dict[str, float], cfg.sfx_volumes)
    return round(min(max(mults.get(kind, 0.7) * cfg.sfx_volume, 0.0), 2.0), 3)


def _sfx_bank_hash(bank: dict[str, list[Path]], bank_root: Path) -> str:
    parts = [f"{p.relative_to(bank_root).as_posix()}:{_sha256_file(p)}"
             for files in bank.values() for p in files]
    return _sha256_text("\n".join(sorted(parts)))


def load_sfx_bank(sfx_dir: Path | str) -> dict[str, list[Path]]:
    """Discover the sound bank: <dir>/<kind>/* (default/ fills any empty
    category). Deterministic order; raises on a missing/empty bank."""
    bank_root = Path(sfx_dir)
    if not bank_root.is_dir():
        raise VideoError(f"--sfx-dir not found: {bank_root}")

    def _files(cat: Path) -> list[Path]:
        if not cat.is_dir():
            return []
        return sorted(p for p in cat.iterdir()
                      if p.is_file() and p.suffix.lower() in _SFX_AUDIO_EXTS)

    bank: dict[str, list[Path]] = {k: _files(bank_root / k)
                                   for k in SFX_KINDS}
    fallback = _files(bank_root / "default")
    for kind in SFX_KINDS:
        if not bank[kind] and fallback:
            bank[kind] = fallback
    if not any(bank.values()):
        raise VideoError(
            f"--sfx-dir has no audio files: {bank_root} (expected "
            "transition/ action/ reveal/ subfolders with .wav/.mp3/.ogg/"
            ".m4a/.flac)")
    for kind in SFX_KINDS:
        if not bank[kind]:
            log.warning("sfx bank: no %r sounds and no default/ fallback; "
                        "%s events will be skipped", kind, kind)
    return bank


def _panel_class_for_sfx(text: str, quotes: list[str]) -> str:
    try:
        from cinematic_effects import classify_panel
        return classify_panel({"narration": text or "",
                               "dialogue": " ".join(q for q in quotes if q)})
    except Exception:  # noqa: BLE001 - pacing must survive a broken helper
        return "calm"


def _sfx_action_time(entry: TimelineEntry, text: str,
                     words: list[dict]) -> tuple[float, str]:
    """When to fire the action hit: at the MEASURED timestamp of the matching
    keyword word when word timings exist, else just after the panel cut."""
    m = _SFX_ACTION_RE.search(text or "")
    if m:
        kw = m.group(1).lower()
        for w in words:
            wt = str(w.get("text", "")).strip(".,!?…\"'“”’ ").lower()
            if wt == kw:
                return entry.start_seconds + float(w.get("start", 0.0)), \
                    f"keyword:{kw}"
    return entry.start_seconds + 0.15, "panel_class:action"


def build_sfx_plan(timeline: TimelineArtifact, narration: NarrationArtifact,
                   audio: AudioArtifact, cfg: VideoConfig) -> SfxArtifact:
    """Tag panel cuts and narration beats with sound effects.

    Rules (at most 3 events per panel, so dense scenes stay tasteful):
      * transition — a whoosh/page-flick at every panel cut except the video's
        first start;
      * action     — an impact hit on action-class panels, at the keyword
        word's measured timestamp when word timings are available;
      * reveal     — a riser/rumble right after a reveal-class panel's cut.
    Silent/dialogue/calm panels get no beat sounds. All times are absolute
    in the finished video and clamped to its bounds.
    """
    if not cfg.sfx_dir:
        raise VideoError("build_sfx_plan requires cfg.sfx_dir")
    bank = load_sfx_bank(cfg.sfx_dir)
    bank_dir = Path(cfg.sfx_dir)
    by_text = {n.id: n for n in narration.entries}
    by_audio = {a.entry_id: a for a in audio.entries}
    total = total_seconds(timeline)
    events: list[SfxEvent] = []
    for i, e in enumerate(timeline.entries):
        n = by_text.get(e.panel_id)
        text = n.text if n else ""
        quotes = list(n.quotes) if n else []
        words = by_audio[e.panel_id].words if e.panel_id in by_audio else []
        if i > 0 and bank["transition"] and e.audio_path:
            # Transition whoosh at cuts that OPEN a spoken beat only: with
            # un-narrated filler beats in the timeline the cut density would
            # otherwise turn the whoosh bank into machine-gun sfx.
            events.append(SfxEvent(
                id=f"sfx_{len(events) + 1:03d}", kind="transition",
                panel_id=e.panel_id, at_seconds=round(e.start_seconds, 3),
                source=_sfx_source_ref(
                    _sfx_pick(bank["transition"],
                              f"{e.panel_id}:transition"), bank_dir),
                volume=_sfx_volume_for(cfg, "transition"),
                trigger="panel_cut"))
        cls = _panel_class_for_sfx(text, quotes)
        if cls == "action" and bank["action"]:
            at, trigger = _sfx_action_time(e, text, words)
            events.append(SfxEvent(
                id=f"sfx_{len(events) + 1:03d}", kind="action",
                panel_id=e.panel_id, at_seconds=round(at, 3),
                source=_sfx_source_ref(
                    _sfx_pick(bank["action"],
                              f"{e.panel_id}:action"), bank_dir),
                volume=_sfx_volume_for(cfg, "action"), trigger=trigger,
                text=text[:120]))
        elif cls == "reveal" and bank["reveal"]:
            events.append(SfxEvent(
                id=f"sfx_{len(events) + 1:03d}", kind="reveal",
                panel_id=e.panel_id,
                at_seconds=round(e.start_seconds + 0.1, 3),
                source=_sfx_source_ref(
                    _sfx_pick(bank["reveal"],
                              f"{e.panel_id}:reveal"), bank_dir),
                volume=_sfx_volume_for(cfg, "reveal"),
                trigger="panel_class:reveal", text=text[:120]))
    for ev in events:
        ev.at_seconds = round(min(max(ev.at_seconds, 0.0),
                                  max(total - 0.2, 0.0)), 3)
    plan = SfxArtifact(
        meta=_meta(cfg.hash(), {
            "timeline.json": _sha256_text(timeline.model_dump_json()),
            "sfx_bank": _sfx_bank_hash(bank, bank_dir)}),
        bank_dir=str(bank_dir), events=events)
    log.info("build_sfx_plan events=%d (transition=%d action=%d reveal=%d) "
             "total=%.2fs", len(events),
             sum(1 for x in events if x.kind == "transition"),
             sum(1 for x in events if x.kind == "action"),
             sum(1 for x in events if x.kind == "reveal"), total)
    return plan


def mix_sfx(video_path: Path, plan: SfxArtifact,
            cfg: VideoConfig) -> bool:
    """Post-render audio mixdown: overlay the plan's SFX onto the finished
    video. The video stream is stream-copied (bytes untouched), so this is
    cheap and only runs right after a fresh render — never on a video cache
    hit, where the mp4 already contains the mix. Returns True when mixed.
    """
    if not cfg.sfx_dir or not plan.events:
        return False
    bank_dir = Path(cfg.sfx_dir)
    events: list[SfxEvent] = []
    for ev in plan.events:
        if (bank_dir / ev.source).is_file():
            events.append(ev)
        else:
            log.warning("mix_sfx: skipping %s — bank file missing: %s",
                        ev.id, ev.source)
    if not events:
        return False
    exe = _resolve_ffmpeg(cfg.ffmpeg_exe)

    def _mix_cmd(event_list: list[SfxEvent],
                 normalize: bool) -> tuple[list[str], list[str]]:
        chains, labels = [], []
        for i, ev in enumerate(event_list):
            delay_ms = int(round(max(ev.at_seconds, 0.0) * 1000))
            chains.append(
                f"[{i + 1}:a]aresample=48000,aformat=channel_layouts=stereo,"
                f"volume={ev.volume:.3f},adelay={delay_ms}:all=1[s{i}]")
            labels.append(f"[s{i}]")
        mix = (f"amix=inputs={len(event_list) + 1}:duration=first"
               + (":normalize=0" if normalize else ""))
        # [0:a] (the rendered narration track) MUST be the first amix input:
        # with duration=first the mix ends when the FIRST input ends. If a
        # delayed SFX clip came first instead, everything past that clip —
        # i.e. all narration after the first few seconds — would be silently
        # truncated from the output.
        filt = (";".join(chains) + ";[0:a]" + "".join(labels)
                + f"{mix}[aout]")
        cmd = [exe, "-y", "-i", str(video_path)]
        for ev in event_list:
            cmd += ["-i", str(bank_dir / ev.source)]
        cmd += ["-filter_complex", filt, "-map", "0:v", "-map", "[aout]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart"]
        return filt, cmd

    tmp = video_path.with_name(video_path.stem + ".sfx.mp4")
    filt, cmd = _mix_cmd(events, normalize=True)
    ok = False
    try:
        proc = subprocess.run(cmd + [str(tmp)], capture_output=True,
                              text=True, check=False, shell=False,
                              timeout=600)
        ok = proc.returncode == 0 and tmp.is_file()
        if not ok:
            log.warning("mix_sfx failed (%.300s) — retrying with the legacy "
                        "amix volume-scaling fallback", (proc.stderr or ""))
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("mix_sfx failed: %s — retrying with the legacy amix "
                    "volume-scaling fallback", exc)
    if not ok:
        # Legacy ffmpeg without amix normalize= : emulate it by pre-scaling
        # every event down (normalize divides each input by N).
        scaled = [ev.model_copy(update={"volume": ev.volume / len(events)})
                  for ev in events]
        filt, cmd = _mix_cmd(scaled, normalize=False)
        try:
            proc = subprocess.run(cmd + [str(tmp)], capture_output=True,
                                  text=True, check=False, shell=False,
                                  timeout=600)
            ok = proc.returncode == 0 and tmp.is_file()
            if not ok:
                log.warning("mix_sfx legacy fallback failed: %.300s",
                            (proc.stderr or ""))
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("mix_sfx legacy fallback failed: %s", exc)
    if not ok:
        tmp.unlink(missing_ok=True)
        log.warning("mix_sfx: keeping the un-mixed video (SFX skipped); "
                    "re-run with --force after fixing the bank/ffmpeg")
        return False
    tmp.replace(video_path)
    log.info("mix_sfx complete: %d events over %s", len(events), video_path)
    return True


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
    from adapters.render_ffmpeg import (
        RenderError,
        StyleConfig,
        pick_render_strategy,
        render,
        render_chunked,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".partial.mp4")
    style = StyleConfig(
        blur_background=cfg.blur_background,
        color_grade=cfg.color_grade,
        vignette=cfg.vignette,
        vignette_angle=cfg.vignette_angle,
        blur_sigma=cfg.blur_sigma,
        zoom_strength=cfg.zoom_strength)
    log.info("render_video start out=%s timeline_entries=%d style=blur=%s/"
             "grade=%s/vignette=%s",
             out_path, len(timeline.entries), style.blur_background,
             style.color_grade, style.vignette)
    t0 = time.time()
    strategy = pick_render_strategy(len(timeline.entries), total_seconds(timeline))
    log.info("render strategy=%s", strategy)
    # The render is the long silent phase of a CLI run: show a live rich bar
    # (interactive) or a throttled INFO line every 10% (embedded/webapp, who
    # must not paint on the uvicorn console and track progress via job
    # records instead). ffmpeg's own time= output drives the fraction, so it
    # stays honest even for the chunked strategy (seconds across all segs).
    bar_ctx: Progress | None = None
    task_id = None
    _last_pct = [-10]

    def _report(frac: float, msg: str) -> None:
        if bar_ctx is not None:
            desc = f"[cyan]Rendering video ({strategy})"
            if msg:
                desc += f" · {msg}"
            bar_ctx.update(task_id, completed=frac, description=desc)
            return
        pct = int(frac * 100) // 10 * 10
        if pct >= _last_pct[0] + 10:
            _last_pct[0] = pct
            el = time.time() - t0
            eta = (el / frac - el) if 0 < frac < 1 else 0.0
            log.info("render %3d%% (%.0fs elapsed, ~%.0fs left)%s",
                     min(pct, 100), el, eta, f" [{msg}]" if msg else "")

    if _HAS_RICH and not EMBEDDED_MODE:
        bar_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        )
        bar_ctx.start()
        task_id = bar_ctx.add_task(f"[cyan]Rendering video ({strategy})",
                                   total=1.0)
    try:
        if strategy == "chunked":
            render_chunked(
                timeline, tmp, ffmpeg_exe=exe, chunk_size=12,
                profile={"preset": "veryfast", "crf": "23", "threads": "4"},
                style=style, progress_cb=_report)
        else:
            render(timeline, tmp, ffmpeg_exe=exe, style=style,
                   progress_cb=_report)
    except RenderError as exc:
        raise VideoError(str(exc)) from exc
    finally:
        if bar_ctx is not None:
            bar_ctx.stop()
    elapsed = time.time() - t0
    tmp.replace(out_path)
    # +faststart is folded into the encode itself (build_command /
    # build_command_chunked) — no separate full-file copy pass here.
    log.info("render_video complete out=%s duration=%.2fs", out_path, elapsed)


def _apply_faststart(path: Path, ffmpeg_exe: str) -> None:
    """Deprecated: +faststart is now folded into the encode itself
    (adapters.render_ffmpeg.build_command*), so no second copy pass runs.
    Kept as a no-op shim for any external callers/tests."""
    return None


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
    narration = build_narration(artifact, cfg, panels_hash=panels_hash,
                                work_dir=work, panels_dir=panels_dir)
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
                "backend (agnes).", prov)

    # 2. audio
    audio = synthesize_audio(narration, audio_dir, cfg, force=force)

    # 3. timeline
    timeline = build_timeline(artifact, panels_dir, narration, audio,
                              audio_dir, cfg, panels_hash=panels_hash,
                              canvas_w=cfg.canvas_w, canvas_h=cfg.canvas_h)
    _write_atomic(work / "timeline.json",
                  timeline.model_dump_json(indent=2) + "\n")
    # 3b. motion debug sidecar (reference preset only; preview the camera
    # plan without rendering the full video).
    motion_report = build_motion_report(timeline)
    if motion_report:
        _write_atomic(work / "motion_report.json",
                      json.dumps(motion_report, indent=2) + "\n")

    # 3c. sfx plan (opt-in: cfg.sfx_dir) — deterministic, before the render
    # so a render failure still leaves the auditable plan sidecar.
    sfx: SfxArtifact | None = None
    if cfg.sfx_dir:
        sfx = build_sfx_plan(timeline, narration, audio, cfg)
        _write_atomic(work / "sfx.json",
                      sfx.model_dump_json(indent=2) + "\n")

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
        "motion_preset": cfg.motion_preset,
        "motion_report": (str(work / "motion_report.json")
                          if motion_report else None),
        "sfx": (str(work / "sfx.json") if sfx is not None else None),
        "sfx_events": (len(sfx.events) if sfx is not None else 0),
    }
    if dry_run:
        log.info("make_recap_video dry_run summary=%s", summary)
        return summary

    # 4. render (cache: skip when the timeline hash AND sfx plan are unchanged;
    # a cache hit means the mp4 already contains the SFX mixdown)
    stamp = out_path.with_name(out_path.name + ".hash")
    tl_hash = _sha256_text(timeline.model_dump_json())
    sfx_hash = (_sha256_text(sfx.model_dump_json())
                if sfx is not None else "off")
    render_key = f"{tl_hash}:{sfx_hash}"
    if (out_path.is_file() and stamp.is_file() and not force
            and stamp.read_text("utf-8").strip() == render_key):
        log.info("video cache hit: %s", out_path)
    else:
        render_video(timeline, out_path, cfg)
        if sfx is not None:
            mix_sfx(out_path, sfx, cfg)
        _write_atomic(stamp, render_key + "\n")
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

    # Reconstruct timeline from edited entries
    tl_entries = []
    for e in editor.project.edited_timeline:
        effect = next((fx for fx in editor.project.effects if fx["panel_id"] == e["panel_id"]), None)
        pan_kind = effect["kind"] if effect else e.get("pan", {}).get("kind", "static")
        scaled_w = e.get("pan", {}).get("scaled_w", WIDTH)
        scaled_h = e.get("pan", {}).get("scaled_h", HEIGHT)
        travel_px = e.get("pan", {}).get("travel_px", 0)
        if pan_kind in ("zoom_in", "zoom_out"):
            # zoom covers the canvas; the renderer derives the crop from the
            # timeline's declared dimensions, so match those here.
            scaled_w = editor.project.project.get("width", WIDTH)
            scaled_h = editor.project.project.get("height", HEIGHT)
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
            motion=e.get("motion"),
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
