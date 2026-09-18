# manhwa-recap

Turn tall manhwa/webtoon strips into per-panel images with AI-generated narration.

## What it does

```
sample_strip.png ──► panel_001.png + panel_002.png + ... + panels.json
                                     (each panel's narration, dialogue,
                                      Y-range, type, confidence)
```

A two-phase pipeline:
1. **Phase 1** (`strip_analyzer.py`): the strip is sliced into overlapping
    2000px chunks and sent to a vision LLM. The backend is **agnes**
    (Agnes AI gateway, OpenAI-compatible `https://apihub.agnes-ai.com/v1`):
    primary model `agnes-2.5-flash` with automatic fallback to
    `agnes-2.0-flash` on any failure — same prompt, same image, no
    fabricated results. The model returns a strict JSON panel plan:
    boundaries, narration, dialogue, and per-panel confidence.
2. **Phase 2** (`guided_cutter.py`): AI boundaries are *refined* by snapping to
   real gutters (row-variance + Sobel edge density), continuous art is merged,
   oversized panels are split, and cuts never go through speech bubbles. Output:
   one PNG per panel plus a `panels.json` sidecar.

If the AI call fails or most panels are low-confidence, the pipeline falls back
to a pure pixel gutter detector (no narration).

## Install

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
```

Optional extras:
```bash
pip install openai                # Agnes backend client (OpenAI-compatible)
```

Set API keys in a `.env` file (see `.env.example`).

## Quick start

```bash
# Phase 1 only (dry run) — prints the plan, cuts nothing
python cli.py guided plan sample_strip.png --backend agnes

# Phase 1 + 2 — full pipeline
python cli.py guided run sample_strip.png --out-dir guided_out --backend agnes

# Offline (gutter-detector fallback, no API key)
python cli.py guided run sample_strip.png --backend none

# Narration script for TTS
python cli.py guided narrate plan.json --style recap --out narration.txt

# Phase 3 — narrated 9:16 recap video (needs ffmpeg on PATH)
python cli.py guided video guided_out/panels.json --out guided_out/recap.mp4

# Silent, captioned video without any network/TTS
python cli.py guided video guided_out/panels.json --tts none

# Preview the timing without rendering
python cli.py guided video guided_out/panels.json --dry-run
```

Outputs next to the mp4: recap.srt (captions), timeline.json, audio clips,
narration.json. Re-running is incremental: unchanged narration reuses the
cached audio, an unchanged timeline skips the ffmpeg render (`--force` resets).

### Offline speech (Kokoro)

Spoken videos use local Kokoro TTS (`--tts kokoro`, the default) — no
network calls. Fetch the weights once (~300MB, or ~80MB quantized;
see the kokoro-onnx README for the download links) and either place
`kokoro-v1.0.onnx` + `voices-v1.0.bin` in `./models/` or point
`KOKORO_MODEL_PATH` / `KOKORO_VOICES_PATH` at them:

```bash
# Narrated video with a Kokoro voice (default)
python cli.py guided video guided_out/panels.json --voice af_heart --speed 1.0
```

### Manhwa-recap visual style (blur + vignette)

`guided video` renders with the manhwa-recap look by default: each panel
floats on a blurred, slightly darkened full-frame copy of itself, with a
strong dark vignette around all four edges. The colour grade is OFF by
default.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--blur-background` / `--no-blur-background` | on | panel contain-fitted over a blurred full-frame background (disables the Ken-Burns pan, since the whole panel is already visible) |
| `--vignette` / `--no-vignette` | on | dark vignette on all 4 edges |
| `--vignette-angle` | `PI/2.5` | ffmpeg angle expression; **smaller = stronger** (`PI/3.5` mild, `PI/2.5` default, `PI/2.1` aggressive) |
| `--color-grade` / `--no-color-grade` | off | darken + desaturate + cool blue-gray tint |
| `--blur-sigma` | `40` | gblur sigma of the background branch |

```bash
# The shipped default look
python cli.py guided video guided_out/panels.json

# Classic Ken-Burns edit, no styling at all
python cli.py guided video guided_out/panels.json --no-blur-background --no-vignette

# Mild vignette + the moody colour grade
python cli.py guided video guided_out/panels.json --vignette-angle PI/3.5 --color-grade
```

Style flags are part of the timeline cache key, so changing the look re-renders
the video — but NOT the TTS audio: narration and speech depend only on text,
voice and pacing, so toggling a visual flag never re-synthesizes clips.

### Cinematic pass (optional Phase 3 variant)

`guided cinematic` renders the same panels.json into a manhwa-recap-style
video: punch zoom on action panels, Ken-Burns zoom+pan, screen shake, glitch
transitions, vignette, teal/orange color grade, speed lines, optional
letterbox and background music. Panels are auto-classified (action / reveal /
dialogue / calm) from their narration text; see `cinematic_effects.py`.

```bash
# Full dynamic style on top of an existing guided_out/
python cli.py guided cinematic guided_out/panels.json

# Subtle effects, with background music
python cli.py guided cinematic guided_out/panels.json --style subtle \
    --bgm bgm.mp3 --bgm-volume 0.15

# Letterbox bars, no glitch transitions
python cli.py guided cinematic guided_out/panels.json --letterbox --no-glitch
```

The webapp also exposes this as the Cinematic Studio view (`#/cinematic`):
one-click full pipeline, stage-by-stage semi-auto runs, per-panel effect
overrides with 3.5s previews, BGM upload, color-grade swatches, and an
export panel.

## How the pipeline works

```
strip.png
   │
   ▼  Phase 1: AI pre-read
plan.json  ──►  panel boundaries (normalized 0–1000 coords in each chunk,
                converted back to absolute strip pixels), narration, dialogue
   │
   ▼  Phase 2: physical dissection
panel_001.png + panel_002.png + … + panels.json
```

**Gutter detection** uses two signals: row variance (low = uniform gutter) AND
Sobel edge density (low = no screentone/gradient). A gutter must be low on
**both**. This prevents mis-firing on heavy screentone backgrounds.

## Webapp

```bash
pip install -e .[web]
uvicorn webapp.main:app --port 8000
```

Open `http://localhost:8000`. Nothing runs automatically on upload —
press Run to start a job. Set `AGNES_API_KEY` in
`.env` for AI features; without a key the app runs in offline mode
(deterministic cropping; no AI narration).

## Tests

```bash
pytest tests webapp -q
```

All tests are offline and synthetic — they prove plumbing, not real-world
vision accuracy.

## Live smoke test

```bash
python scripts/smoke_test_live.py samples/real_strip_01.png --backend agnes
```

Prints panel count, snap-distance stats, % below confidence 0.5, token usage,
and wall time. Requires `samples/real_strip_01.png` and a valid API key.

## License

MIT
