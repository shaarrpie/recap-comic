# tests/test_ai_narration_memory.py
"""Offline tests for the story_context wiring in adapters/ai_narration.py.

Fake transports only — no network, no keys. Enforces the INTEGRATION
contract from story_context.py:
  * seed pass runs BEFORE the panel loop (one text call)
  * per-panel vision prompt carries the frozen [STORY MEMORY] seed block
  * panels are narrated INDEPENDENTLY in a thread pool (no per-panel memory
    mutation), so the cache key is stable and re-runs HIT cache; continuity is
    restored by the single whole-chapter script pass, not by chaining panels
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


def _narrate(session: Path, harness, **kw):
    """narrate_cropped_panels with an ISOLATED cache dir per call site.

    Tests share identical white panel PNGs; without isolation the repo
    .cache/ai-narration leaks captions between tests (and between runs).
    """
    cache = session / "_test_cache" / f"call_{kw.pop('call_id', id(harness))}"
    cache.mkdir(parents=True, exist_ok=True)
    return ain.narrate_cropped_panels(
        session, api_key="k", request_fn=harness.request_fn, gap_s=0,
        cache_dir=cache, **kw)


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
    IMAGE_SEED_MARK = "contact-sheet"
    SEED_USER_MARK = "Panel text in order:"
    SCRIPT_USER_MARK = "Write the recap narration for this chapter"

    def __init__(self, vision_texts: list[str], seed_text: str = SEED_JSON,
                 script_text: str = "not-a-lines-response",
                 image_seed_text: str = SEED_JSON):
        self.vision_texts = list(vision_texts)
        self.seed_text = seed_text
        self.script_text = script_text
        self.image_seed_text = image_seed_text
        self.vision_calls: list[dict] = []
        self.seed_calls: list[str] = []
        self.image_seed_calls: list[dict] = []
        self.script_calls: list[str] = []

    def request_fn(self, model: str, prompt: str, third: str = "") -> str:
        if self.VISION_MARK in prompt:
            self.vision_calls.append({"model": model, "prompt": prompt,
                                      "b64": third[:16]})
            if not self.vision_texts:
                raise RuntimeError("no scripted vision response")
            return self.vision_texts.pop(0)
        if self.IMAGE_SEED_MARK in prompt:
            self.image_seed_calls.append({"model": model, "b64": third[:16]})
            return self.image_seed_text
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
    summary = _narrate(session, h, call_id="seed_first")

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


def test_cache_key_is_stable_across_runs(session):
    """Per-panel vision no longer mutates memory mid-loop (the seed ctx is
    frozen), so the cache key is STABLE and a re-run HITS cache instead of
    re-spending every caption -- the opposite of the old chaining design, and
    exactly what makes re-runs cheap."""
    cache = session / "_test_cache" / "stable_key"
    cache.mkdir(parents=True, exist_ok=True)

    h1 = Harness([_vision_response("Bam stands at the gate.", '"Hi"'),
                  _vision_response("Bam runs.", "")])
    s1 = ain.narrate_cropped_panels(session, api_key="k",
                                    request_fn=h1.request_fn, gap_s=0,
                                    cache_dir=cache)
    assert s1["narrated"] == 2 and s1["cached"] == 0
    assert len(list(cache.glob("panel_001_*.json"))) == 1

    # second run, same frozen seed -> both panels served from cache, no new
    # vision calls (the scripted response below must never be consumed)
    h2 = Harness([_vision_response("MUST NOT BE USED", "")])
    s2 = ain.narrate_cropped_panels(session, api_key="k",
                                    request_fn=h2.request_fn, gap_s=0,
                                    cache_dir=cache)
    assert s2["cached"] == 2 and s2["narrated"] == 0
    assert len(list(cache.glob("panel_001_*.json"))) == 1


def test_panels_narrate_independently_no_chaining(session):
    """A panel's entities no longer thread into the NEXT panel's prompt (the
    loop is independent + parallel); continuity is the whole-chapter script
    pass's job. So panel prompts carry only the SEED roster, never a peer's
    newly-emitted entity, and the roster does not grow during narration."""
    h = Harness([
        _vision_response("A girl with pink hair appears.", '"Who?"',
                         entities={"new_characters": [
                             {"name": "Endorsi", "role": "antagonist",
                              "description": "pink hair"}]}),
        _vision_response("Endorsi smiles.", ""),
    ])
    summary = _narrate(session, h, call_id="independent")
    assert summary["narrated"] == 2
    # no per-panel chaining: Endorsi never reaches any vision prompt
    assert all("Endorsi" not in c["prompt"] for c in h.vision_calls)
    # the frozen seed roster (Bam) is present in the prompts
    assert any("Bam" in c["prompt"] for c in h.vision_calls)
    # and the saved context is the seed, not grown by panel entities
    ctx = json.loads((session / "story_context.json").read_text("utf-8"))
    assert "Endorsi" not in ctx["characters"]


def test_seed_failure_is_non_fatal(session):
    class Boom(Harness):
        def request_fn(self, model, prompt, third=""):
            if self.SEED_USER_MARK in third:
                raise RuntimeError("seed unavailable")
            return super().request_fn(model, prompt, third)

    h = Boom([_vision_response("No memory narration.", ""),
              _vision_response("Still narrates.", "")])
    summary = _narrate(session, h, call_id="boom", seed_cast_from_images=False)
    assert summary["narrated"] == 2
    # memory field present (seed failed -> False)
    assert "story_memory" in summary


def test_scrub_fences_applied_to_narration(session):
    fenced = ("```json\nBam leaps.\n```")
    h = Harness([_vision_response(fenced, ""),
                 _vision_response("Bam runs.", "")])
    _narrate(session, h, call_id="scrub")
    art = CutArtifact.model_validate_json(
        (session / "panels.json").read_text("utf-8"))
    # panels narrate in a thread pool, so which panel receives the fenced
    # response is non-deterministic; assert the fences were scrubbed wherever
    # it landed and that no code-fence markup survives on any panel.
    narrations = [p.narration for p in art.panels]
    assert "Bam leaps." in narrations
    assert all("```" not in (n or "") and "json" not in (n or "")
               for n in narrations)


def test_geometry_lock_holds(session):
    before = CutArtifact.model_validate_json(
        (session / "panels.json").read_text("utf-8"))
    geo = {p.id: (p.y_start, p.y_end, p.image_file) for p in before.panels}
    h = Harness([_vision_response("x", ""), _vision_response("y", "")])
    _narrate(session, h, call_id="geom")
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
    summary = _narrate(session, h, call_id="scriptpass")
    assert calls["n"] == 1
    assert summary["script_pass"]["status"] == "built"
    # narration.txt comes from the script pass
    assert (session / "narration.txt").read_text("utf-8") == "Hook."


# --------------------------------------------------- vision cast survey (seed)
@pytest.fixture()
def empty_session(tmp_path: Path) -> Path:
    """A fresh cut: no dialogue/narration yet, so the text seed is empty."""
    panels = [_panel(1), _panel(2)]
    art = CutArtifact(source="strip.png", width=800, height=1600,
                      plan_hash="x", config={}, panels=panels)
    (tmp_path / "panels.json").write_text(art.model_dump_json(), "utf-8")
    for p in panels:
        Image.new("RGB", (800, 800), "white").save(tmp_path / p.image_file)
    return tmp_path


def test_image_seed_fires_when_panels_have_no_text(empty_session):
    h = Harness([_vision_response("A boy stands at a gate.", ""),
                 _vision_response("The boy runs.", "")],
                image_seed_text=SEED_JSON)
    summary = _narrate(empty_session, h, call_id="imgseed")
    assert summary["narrated"] == 2
    # no panel text -> the text seed made NO model call at all
    assert h.seed_calls == []
    # exactly ONE vision cast-survey over the contact sheet
    assert len(h.image_seed_calls) == 1
    ctx = json.loads(
        (empty_session / "story_context.json").read_text("utf-8"))
    assert "Bam" in ctx["characters"]
    assert ctx["_meta"]["image_seed_built"] is True
    # the surveyed cast reached the 2nd panel's memory-augmented prompt
    assert "[STORY MEMORY" in h.vision_calls[1]["prompt"]
    assert "Bam" in h.vision_calls[1]["prompt"]


def test_image_seed_skipped_when_text_present(session):
    h = Harness([_vision_response("Bam stands.", ""),
                 _vision_response("Bam runs.", "")])
    _narrate(session, h, call_id="imgskip")
    # panels carry dialogue -> text seed succeeds -> no extra vision call
    assert len(h.seed_calls) == 1
    assert h.image_seed_calls == []


def test_image_seed_failure_is_non_fatal(empty_session):
    h = Harness([_vision_response("No memory.", ""),
                 _vision_response("Narrates anyway.", "")],
                image_seed_text="this is not json")
    summary = _narrate(empty_session, h, call_id="imgboom")
    assert summary["narrated"] == 2
    assert len(h.image_seed_calls) == 1
    ctx = json.loads(
        (empty_session / "story_context.json").read_text("utf-8"))
    # unparseable survey leaves the cast empty (retryable) but never crashes
    assert ctx["characters"] == {}
    assert ctx["_meta"]["image_seed_built"] is not True
