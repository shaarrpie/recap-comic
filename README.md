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
   2000px chunks and sent to a vision LLM (Gemini / OpenAI / Anthropic / local
   Ollama). The model returns a strict JSON panel plan: boundaries, narration,
   dialogue, and per-panel confidence.
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
pip install google-genai          # Gemini backend
pip install openai                # OpenAI backend
pip install anthropic             # Anthropic backend
```

Set API keys in a `.env` file (see `.env.example`).

## Quick start

```bash
# Phase 1 only (dry run) — prints the plan, cuts nothing
python cli.py guided plan sample_strip.png --backend gemini

# Phase 1 + 2 — full pipeline
python cli.py guided run sample_strip.png --out-dir guided_out --backend gemini

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

Outputs next to the mp4: recap.srt (captions), timeline.json, audio/*.mp3,
narration.json. Re-running is incremental: unchanged narration reuses the
cached audio, an unchanged timeline skips the ffmpeg render (`--force` resets).

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

## Tests

```bash
pytest tests -q
```

All tests are offline and synthetic — they prove plumbing, not real-world
vision accuracy.

## Live smoke test

```bash
python scripts/smoke_test_live.py samples/real_strip_01.png --backend gemini
```

Prints panel count, snap-distance stats, % below confidence 0.5, token usage,
and wall time. Requires `samples/real_strip_01.png` and a valid API key.

## License

MIT
