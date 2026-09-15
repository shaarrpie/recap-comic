# tests/test_ai_narration_memory.py
"""Offline tests for the story_context wiring in adapters/ai_narration.py.

Fake transports only — no network, no keys. Enforces the INTEGRATION
contract from story_context.py:
  * seed pass runs BEFORE the panel loop (one text call)
  * per-panel vision prompt carries the [STORY MEMORY] block
  * the cache key includes the memory-augmented prompt (different memory
    state -> different cache file -> no stale-caption reuse)
  * optional "entities" in a response update the context + save it
  * geometry lock still holds; scrub_fences applied to narration
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from adapters import ai_narration as ain
from guided_cutter import CutArtifact, CutPanel

SEED_JSON = json.dumps({
    "series_title": "Tower of God",
    "characters": [
        {"name": "Bam", "role": "protagonist",
         "description": "black hair, determined", "aliases": []},
    ],
    "locations": [],
    "story_threads": [
        {"summary": "Bam is climbing the Tower to find Rachel"}],
})


def _panel(i: int, narration: str = "", dialogue: str = "") -> CutPanel:
    return CutPanel(id=f"panel_{i:03d}", panel_index=i, y_start=(i - 1) * 800,
                    y_end=i * 800, narration=narration, dialogue=dialogue,
                    panel_type="panel", confidence=0.9,
                    image_file=f"panel_{i:03d}.png")


@pytest.fixture()
def session(tmp_path: Path) -> Path:
    panels = [_panel(1, "", '"Where am I?"'),
              _panel(2, "", '"Bam, hurry!"')]
    art = CutArtifact(source="strip.png", width=800, height=1600,
                      plan_hash="x", config={}, panels=panels)
    (tmp_path / "panels.json").write_text(art.model_dump_json(), "utf-8")
    for p in panels:
        Image.new("RGB", (800, 800), "white").save(tmp_path / p.image_file)
    return tmp_path


def _vision_response(narration: str, dialogue: str,
                     entities: dict | None = None) -> str:
    obj = {"narration": narration, "dialogue": dialogue}
    if entities is not None:
        obj["entities"] = entities
    return json.dumps(obj)


class Harness:
    """Fake transport + recorder for narrate_cropped_panels runs.

    Routing is by CONTENT, not position: ai_models passes
    (model, prompt, b64_png) for vision calls and (model, system, user)
    for text calls — positionally identical, so the second argument's
    text decides: vision prompts narrate a panel, the seed system
    pre-reads the chapter. The user body distinguishes seed vs the
    Phase 2.5 script pass (both use _SEED_SYSTEM as system).
    """

    VISION_MARK = "You are narrating ONE cropped comic/manhwa panel"
    SEED_USER_MARK = "Panel text in order:"
    SCRIPT_USER_MARK = "Write the recap narration for this chapter"

    def __init__(self, vision_texts: list[str], seed_text: str = SEED_JSON,
                 script_text: str = "not-a-lines-response"):
        self.vision_texts = list(vision_texts)
        self.seed_text = seed_text
        self.script_text = script_text
        self.vision_calls: list[dict] = []
        self.seed_calls: list[str] = []
        self.script_calls: list[str] = []

    def request_fn(self, model: str, prompt: str, third: str = "") -> str:
        if self.VISION_MARK in prompt:
            self.vision_calls.append({"model": model, "prompt": prompt,
                                      "b64": third[:16]})
            if not self.vision_texts:
                raise RuntimeError("no scripted vision response")
            return self.vision_texts.pop(0)
        if self.SEED_USER_MARK in third:
            self.seed_calls.append(third)
            return self.seed_text
        if self.SCRIPT_USER_MARK in third:
            self.script_calls.append(third)
            return self.script_text
        raise RuntimeError(f"unrouted call: {prompt[:60]!r} / {third[:60]!r}")


def test_seed_runs_before_panel_prompts(session):
    h = Harness([_vision_response("Bam stands at the gate.", '"Bam!"'),
                 _vision_response("Bam runs.", "")])
    summary = ain.narrate_cropped_panels(
        session, api_key="k", request_fn=h.request_fn, gap_s=0)

    assert summary["narrated"] == 2
    # exactly ONE seed text call for two panels
    assert len(h.seed_calls) == 1
    # seed prompt contains BOTH panels' dialogue
    seed_prompt = h.seed_calls[0]
    assert "Where am I?" in seed_prompt and "Bam, hurry!" in seed_prompt
    # the vision prompt on the SECOND panel carries the seeded memory
    assert len(h.vision_calls) == 2
    assert "[STORY MEMORY" in h.vision_calls[1]["prompt"]
    assert "Bam" in h.vision_calls[1]["prompt"]


def test_memory_changes_cache_key(session, tmp_path):
    """The cache key must include the memory-augmented prompt: after the
    roster grows, the same image must NOT reuse the cached caption."""
    cache = tmp_path / "cache"

    h1 = Harness([_vision_response("A person stands.", "")])
    ain.narrate_cropped_panels(session, api_key="k",
                               request_fn=h1.request_fn, gap_s=0,
                               cache_dir=cache)
    cache_files_1 = list(cache.glob("panel_001_*.json"))

    # memory evolves (Bam now last_seen=1): the seed is skipped (already
    # built) but inject_into_prompt now has panel-1 freshness info ->
    # different final prompt -> different cache key -> fresh call.
    # Simulate by giving the vision model a response that updates Bam.
    h2 = Harness([
        _vision_response("Bam stands at the gate.", "",
                         entities={"characters_seen": ["Bam"]}),
    ])
    ain.narrate_cropped_panels(session, api_key="k",
                               request_fn=h2.request_fn, gap_s=0,
                               cache_dir=cache, force=True)
    assert (session / "story_context.json").is_file()
    ctx = json.loads((session / "story_context.json").read_text("utf-8"))
    assert ctx["characters"]["Bam"]["last_seen"] == 1

    # a THIRD run with the grown memory must miss panel_001's old cache
    h3 = Harness([
        _vision_response("Bam stands at the gate, determined.", ""),
        _vision_response("Bam runs.", ""),
    ])
    s3 = ain.narrate_cropped_panels(session, api_key="k",
                                    request_fn=h3.request_fn, gap_s=0,
                                    cache_dir=cache, force=True)
    # both panels re-narrated: memory state changed the cache keys
    assert s3["narrated"] == 2
    cache_files_3 = list(cache.glob("panel_001_*.json"))
    assert len(cache_files_3) == 2      # two distinct cache entries
    assert cache_files_3[0].name != cache_files_3[1].name


def test_entities_update_persists_and_grows_roster(session):
    h = Harness([
        _vision_response("A girl with pink hair appears.", '"Who?"',
                         entities={"new_characters": [
                             {"name": "Endorsi", "role": "antagonist",
                              "description": "pink hair"}]}),
        _vision_response("Endorsi smiles.", ""),
    ])
    ain.narrate_cropped_panels(session, api_key="k",
                               request_fn=h.request_fn, gap_s=0)
    ctx = json.loads((session / "story_context.json").read_text("utf-8"))
    assert "Endorsi" in ctx["characters"]
    # second panel's prompt saw her in memory
    assert "Endorsi" in h.vision_calls[1]["prompt"]


def test_seed_failure_is_non_fatal(session):
    class Boom(Harness):
        def request_fn(self, model, prompt, b64="", system=""):
            if not b64:
                raise RuntimeError("seed unavailable")
            self.vision_calls.append({"model": model, "prompt": prompt,
                                      "b64": ""})
            return self.vision_texts.pop(0)

    h = Boom([_vision_response("No memory narration.", ""),
              _vision_response("Still narrates.", "")])
    summary = ain.narrate_cropped_panels(
        session, api_key="k", request_fn=h.request_fn, gap_s=0)
    assert summary["narrated"] == 2
    assert summary["story_memory"] in (True, False)   # field present


def test_scrub_fences_applied_to_narration(session):
    fenced = ("```json\nBam leaps.\n```")
    h = Harness([_vision_response(fenced, ""),
                 _vision_response("Bam runs.", "")])
    ain.narrate_cropped_panels(session, api_key="k",
                              request_fn=h.request_fn, gap_s=0)
    art = CutArtifact.model_validate_json(
        (session / "panels.json").read_text("utf-8"))
    assert art.panels[0].narration == "Bam leaps."   # fences scrubbed


def test_geometry_lock_holds(session):
    before = CutArtifact.model_validate_json(
        (session / "panels.json").read_text("utf-8"))
    geo = {p.id: (p.y_start, p.y_end, p.image_file) for p in before.panels}
    h = Harness([_vision_response("x", ""), _vision_response("y", "")])
    ain.narrate_cropped_panels(session, api_key="k",
                               request_fn=h.request_fn, gap_s=0)
    after = CutArtifact.model_validate_json(
        (session / "panels.json").read_text("utf-8"))
    for p in after.panels:
        assert (p.y_start, p.y_end, p.image_file) == geo[p.id]


def test_chapter_script_pass_runs_after_narration(session, monkeypatch):
    """The Phase 2.5 script pass fires after the panels are narrated and
    writes script.json (with a fake text model)."""
    import recap_script as rsc
    calls = {"n": 0}

    def fake_build(session_dir, **kw):
        calls["n"] += 1
        return {"status": "built", "used_fallback": False,
                "lines": [{"panel_id": "panel_001", "panel_index": 1,
                           "text": "Hook.", "part": "hook", "quote": None}],
                "text": "Hook.", "model_used": "fake",
                "structure": rsc.STRUCTURE, "input_hash": "x"}

    monkeypatch.setattr(rsc, "build_chapter_script", fake_build)
    h = Harness([_vision_response("Bam stands.", ""),
                 _vision_response("Bam runs.", "")])
    summary = ain.narrate_cropped_panels(session, api_key="k",
                                         request_fn=h.request_fn, gap_s=0)
    assert calls["n"] == 1
    assert summary["script_pass"]["status"] == "built"
    # narration.txt comes from the script pass
    assert (session / "narration.txt").read_text("utf-8") == "Hook."
