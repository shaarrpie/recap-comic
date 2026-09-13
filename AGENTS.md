# Agent instructions

## Pipeline contract

The guided path is a strict artifact chain:

1. **Phase 1:** `strip_analyzer.py` writes `plan.json` from the source strip.
   - Schema: [schemas/plan.schema.json](schemas/plan.schema.json)
   - Runtime model: `strip_analyzer.PanelPlan`
2. **Phase 2:** `guided_cutter.py` reads that plan and writes `panels.json` plus
   one PNG per panel.
   - Schema: [schemas/panels.schema.json](schemas/panels.schema.json)
   - Runtime model: `guided_cutter.CutArtifact`
3. **Phase 3:** the video pipeline reads `panels.json`, writes `narration.json`,
   then writes `timeline.json`.
   - `narration.json`: [NarrationArtifact](adapters/schemas.py)
   - `timeline.json`: [TimelineArtifact](adapters/schemas.py)

Treat each artifact as immutable input to the next stage. Validate JSON before
editing or passing it downstream; the JSON Schema files describe the wire
shape, while the Pydantic models enforce coordinate, ordering, and provenance
rules that JSON Schema cannot express.

## Golden rule

**Phase 2 must never trust Phase 1 blindly.** Boundary snapping, continuous-art
merging, oversized-panel splitting, and speech-bubble protection belong in
`guided_cutter.py`. Agents may tune cutter parameters and tests, but must not
bypass those safety checks or move them into generated/agent code.

## Agent ownership

Use pipeline stages as the default subagent boundaries:

- **analyzer-agent:** `strip_analyzer.py` and prompts — chunking strategy, prompt
  engineering, confidence calibration, and multi-backend parity.
- **cutter-agent:** `guided_cutter.py` — gutter detection tuning, merge/split
  heuristics, and speech-bubble avoidance.
- **narrator-agent:** narration and TTS — recap styles, caption timing, and SRT
  generation.
- **video-agent:** Phase 3 and FFmpeg — timeline logic, 9:16 layout, and
  incremental cache correctness.
- **qa-agent:** `tests/` and `scripts/smoke_test_live.py` — synthetic fixtures,
  regression samples, and benchmark statistics.

Keep a reviewer/orchestrator agent (or the parent agent) responsible for
cross-cutting decisions and final integration. A change in one stage must be
checked against its downstream contract: for example, analyzer chunk-size or
coordinate changes require matching cutter coordinate math and regression tests.
No stage agent should silently alter another stage's artifact contract.

## Ground truth

Run the offline suite before and after changes:

```bash
pytest tests -q
```

The live accuracy check is:

```bash
python scripts/smoke_test_live.py samples/real_strip_01.png --backend gemini
```

The smoke test needs the sample and an API key. Do not tune thresholds to hide
a large snap distance; inspect the overlay and report the result.

## API key hygiene and CI

Never commit `.env` or real credentials. Copy `.env.example` to a local `.env`
only when running a live backend. Use `--backend none` for CI, offline work, and
deterministic regression tests. Never print or log key values.

## Agent workflow

1. Read the relevant schema and the existing implementation before editing.
2. Keep changes scoped to the requested artifact or parameter.
3. Validate generated JSON with the matching Pydantic model and schema.
4. Run `pytest tests -q`; run the live smoke test only when credentials and the
   sample are available.
