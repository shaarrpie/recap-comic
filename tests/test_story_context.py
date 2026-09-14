# tests/test_story_context.py
"""Offline tests for story_context.py (persistent narrative memory).

All model calls are fakes -- no network, no API keys, no AI. These tests
enforce the review-driven contract:

  * seed: success marks seed_built, transient failure retries next run
  * entities ride INSIDE the strict-JSON response (no fences), extracted
    from either the parsed dict or the raw string
  * alias/case-insensitive matching grows the alias list post-seed
  * thread lifecycle: token-overlap resolve/dedupe; a one-word hint never
    closes a thread
  * injection: ladder-trimmed, footer never cut, threads outlast
    characters, never empty while context is non-empty, seeded-but-unseen
    characters are ranked fresh (last_seen==0 sentinel)
  * shape tolerance: every malformed field degrades, never crashes; a
    corrupt meta does not discard the rest of the update
  * carry_forward resets per-chapter counters but keeps the roster
  * a future-version context file is backed up, not destroyed
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import story_context as sc


# --------------------------------------------------------------------- fakes --
class FakeModel:
    """(prompt -> str) callable with a scriptable queue."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("no scripted response left")
        return self.responses.pop(0)


SEED_JSON = json.dumps({
    "series_title": "Tower of God",
    "characters": [
        {"name": "Bam", "role": "protagonist",
         "description": "black hair, determined", "aliases": ["Twenty-Fifth Bam"]},
        {"name": "Khun", "role": "ally", "description": "blue hair, strategist",
         "aliases": []},
    ],
    "locations": [{"name": "Tower entrance",
                   "description": "massive stone gate"}],
    "story_threads": [
        {"summary": "Bam is climbing the Tower to find Rachel"}],
})


def seeded_ctx(tmp_path: Path) -> sc._empty_context:
    model = FakeModel([SEED_JSON])
    panels = [{"panel_index": 1, "dialogue": "Bam: I must find Rachel",
               "narration": "", "text": ""}]
    return sc.build_seed_context(panels, tmp_path, model_call=model)


# ------------------------------------------------------------------- load/save --
def test_load_missing_file_returns_scaffold(tmp_path):
    ctx = sc.load_context(tmp_path)
    assert ctx["characters"] == {} and ctx["locations"] == {}
    assert ctx["_meta"]["seed_built"] is False
    assert ctx["_meta"]["version"] == sc._VERSION


def test_load_deep_merges_partial_files(tmp_path):
    # hand-edited file missing most keys: no KeyError anywhere
    partial = {"characters": {"Bam": {"last_seen": 3}},
               "_meta": {"seed_built": True}}
    (tmp_path / sc.CONTEXT_FILE).write_text(json.dumps(partial), "utf-8")
    ctx = sc.load_context(tmp_path)
    assert ctx["characters"]["Bam"]["last_seen"] == 3
    assert ctx["locations"] == {}            # filled by scaffold
    assert ctx["story_threads"] == []
    assert ctx["_meta"]["version"] == sc._VERSION  # stamped


def test_load_future_version_backed_up_not_destroyed(tmp_path):
    future = sc._empty_context()
    future["_meta"]["version"] = sc._VERSION + 5
    future["characters"]["Bam"] = {"role": "protagonist"}
    path = tmp_path / sc.CONTEXT_FILE
    path.write_text(json.dumps(future), "utf-8")

    ctx = sc.load_context(tmp_path)
    assert ctx["characters"] == {}            # reset scaffold returned
    bak = tmp_path / "story_context.json.bak"
    assert bak.is_file()                      # but the file survives
    assert json.loads(bak.read_text("utf-8"))["characters"]["Bam"]["role"] \
        == "protagonist"


def test_save_context_atomic_roundtrip(tmp_path):
    ctx = sc._empty_context()
    ctx["characters"]["Bam"] = {"last_seen": 1}
    sc.save_context(ctx, tmp_path)
    assert not (tmp_path / "story_context.tmp").exists()  # tmp replaced
    assert sc.load_context(tmp_path)["characters"]["Bam"]["last_seen"] == 1


# ------------------------------------------------------------------------- seed --
def test_seed_success_populates_roster(tmp_path):
    ctx = seeded_ctx(tmp_path)
    assert ctx["_meta"]["seed_built"] is True
    assert set(ctx["characters"]) == {"Bam", "Khun"}
    assert "Tower entrance" in ctx["locations"]
    assert ctx["story_threads"][0]["status"] == "active"
    assert ctx["characters"]["Bam"]["aliases"] == ["Twenty-Fifth Bam"]


def test_seed_failure_retries_next_run(tmp_path):
    """H1: a transient failure leaves seed_built=False; the next call
    retries instead of skipping."""
    panels = [{"panel_index": 1, "dialogue": "Bam: ..."}]
    boom = FakeModel([])  # raises immediately
    ctx = sc.build_seed_context(panels, tmp_path, model_call=boom)
    assert ctx["_meta"]["seed_built"] is False
    assert ctx["characters"] == {}

    model = FakeModel([SEED_JSON])
    ctx = sc.build_seed_context(panels, tmp_path, model_call=model)
    assert ctx["_meta"]["seed_built"] is True
    assert set(ctx["characters"]) == {"Bam", "Khun"}


def test_seed_skips_when_already_built(tmp_path):
    ctx = seeded_ctx(tmp_path)
    model = FakeModel([])  # would raise if called: proves the skip
    ctx2 = sc.build_seed_context(
        [{"panel_index": 1, "dialogue": "x"}], tmp_path, model_call=model)
    assert ctx2["_meta"]["seed_built"] is True
    assert model.prompts == []


def test_seed_includes_context_only_panels(tmp_path):
    """context_only panels contribute dialogue to the seed text block."""
    panels = [
        {"panel_index": 1, "dialogue": "", "narration": "", "text": ""},
        {"panel_index": 2, "dialogue": "Rachel: hurry, Bam!",
         "narration": "", "text": "", "context_only": True},
    ]
    model = FakeModel([SEED_JSON])
    sc.build_seed_context(panels, tmp_path, model_call=model)
    assert "Rachel: hurry, Bam!" in model.prompts[0]


def test_seed_no_text_marks_built_nothing_to_retry(tmp_path):
    ctx = sc.build_seed_context([{"panel_index": 1}], tmp_path,
                               model_call=FakeModel([SEED_JSON]))
    assert ctx["_meta"]["seed_built"] is True
    assert ctx["characters"] == {}


def test_seed_text_block_cap_checked_before_overflow(tmp_path):
    big = [{"panel_index": i, "dialogue": "x" * 400}
           for i in range(1, 100)]
    model = FakeModel([SEED_JSON])
    sc.build_seed_context(big, tmp_path, model_call=model)
    assert len(model.prompts[0]) <= sc._MAX_TEXT_BLOCK_CHARS + 600


def test_seed_tolerates_fenced_response(tmp_path):
    fenced = "```json\n" + SEED_JSON + "\n```"
    ctx = sc.build_seed_context(
        [{"panel_index": 1, "dialogue": "hi"}], tmp_path,
        model_call=FakeModel([fenced]))
    assert set(ctx["characters"]) == {"Bam", "Khun"}


# -------------------------------------------------------------------- entities --
def test_extract_entities_from_parsed_dict():
    parsed = {"narration": "Bam runs.",
              "entities": {"characters_seen": ["Bam"]}}
    assert sc.extract_entities_from_response(parsed) == \
        {"characters_seen": ["Bam"]}


def test_extract_entities_from_raw_string():
    raw = ('{"narration": "Bam runs.", "dialogue": "go!", '
           '"entities": {"new_events": ["the test begins"]}}')
    assert sc.extract_entities_from_response(raw) == \
        {"new_events": ["the test begins"]}


@pytest.mark.parametrize("bad", [
    {"narration": "no entities key"},
    {"narration": "x", "entities": "not-a-dict"},
    "no json here at all",
    {"narration": "x", "entities": ["list-not-dict"]},
])
def test_extract_entities_absent_or_malformed_returns_none(bad):
    assert sc.extract_entities_from_response(bad) is None


# -------------------------------------------------------------------- matching --
def test_alias_and_case_matching(tmp_path):
    ctx = seeded_ctx(tmp_path)
    store = ctx["characters"]
    assert sc.find_entity_key("twenty-fifth bam", store) == "Bam"
    assert sc.find_entity_key("BAM", store) == "Bam"
    assert sc.find_entity_key("khun", store) == "Khun"
    assert sc.find_entity_key("Endorsi", store) is None
    assert sc.find_entity_key("", store) is None
    assert sc.find_entity_key(None, store) is None


def test_update_grows_alias_list_and_seen_via_alias(tmp_path):
    ctx = seeded_ctx(tmp_path)
    sc.update_context(ctx, {
        "characters_seen": ["Twenty-Fifth Bam"],   # alias form
        "new_characters": [{"name": "BAM", "aliases": ["Jyu Viole Grace"]}],
    }, panel_index=9)
    assert ctx["characters"]["Bam"]["last_seen"] == 9      # no new entry
    assert len(ctx["characters"]) == 2
    assert "Jyu Viole Grace" in ctx["characters"]["Bam"]["aliases"]


def test_new_character_defaults_role_survives_sanitisation(tmp_path):
    ctx = sc._empty_context()
    sc.update_context(ctx, {
        "new_characters": [{"name": "Mute", "role": 42}],  # non-string role
    }, panel_index=1)
    assert ctx["characters"]["Mute"]["role"] == "unknown"


# -------------------------------------------------------------------- threads --
def test_thread_resolved_by_word_overlap(tmp_path):
    ctx = seeded_ctx(tmp_path)
    sc.update_context(ctx, {"threads_resolved": [
        "Bam finds Rachel in the Tower"]}, panel_index=8)
    assert ctx["story_threads"][0]["status"] == "resolved"


def test_one_word_hint_never_resolves_a_thread(tmp_path):
    """R6: substring/one-word hints must not close threads."""
    ctx = seeded_ctx(tmp_path)
    sc.update_context(ctx, {"threads_resolved": ["rachel"]}, panel_index=8)
    sc.update_context(ctx, {"threads_resolved": ["tower"]}, panel_index=8)
    assert ctx["story_threads"][0]["status"] == "active"


def test_thread_dedupe_near_duplicate(tmp_path):
    ctx = sc._empty_context()
    sc.update_context(ctx, {"threads_new": [
        "Bam is climbing the Tower to find Rachel"]}, panel_index=1)
    sc.update_context(ctx, {"threads_new": [
        "climbing the Tower to find Rachel"]}, panel_index=2)
    assert len(ctx["story_threads"]) == 1


def test_thread_dedupe_only_against_active(tmp_path):
    """A re-opened thread may duplicate a RESOLVED one."""
    ctx = sc._empty_context()
    sc._add_thread(ctx["story_threads"], "Khun schemes against the rankers")
    ctx["story_threads"][0]["status"] = "resolved"
    sc._add_thread(ctx["story_threads"], "Khun schemes against the rankers")
    assert len(ctx["story_threads"]) == 2
    assert ctx["story_threads"][1]["status"] == "active"


# -------------------------------------------------------------------- injection --
def test_inject_empty_context_returns_empty():
    assert sc.inject_into_prompt(sc._empty_context(), 5) == ""


def test_inject_includes_all_sections_and_footer(tmp_path):
    ctx = seeded_ctx(tmp_path)
    sc.update_context(ctx, {"new_events": ["Headon sets the first test"]},
                      panel_index=4)
    block = sc.inject_into_prompt(ctx, panel_index=5)
    assert block.startswith("[STORY MEMORY -- panel 5]")
    assert "[/STORY MEMORY]" in block
    assert "maintain continuity" in block          # footer instruction
    assert "Bam" in block and "Khun" in block       # characters
    assert "Tower entrance" in block                # location
    assert "climbing the Tower" in block            # thread
    assert "first test" in block                    # events digest (R5)


def test_inject_seeded_unseen_characters_are_fresh(tmp_path):
    """R4: last_seen==0 (pre-read cast) must NOT be demoted to stale."""
    ctx = seeded_ctx(tmp_path)  # Bam/Khun have last_seen=0
    block = sc.inject_into_prompt(ctx, panel_index=1)
    assert "pre-read cast" in block                # rendered, not [last seen]
    # and ranked fresh: Bam appears in the block at all
    assert "Bam" in block


def test_inject_stale_characters_demoted_not_deleted(tmp_path):
    ctx = seeded_ctx(tmp_path)
    ctx["characters"]["Bam"]["last_seen"] = 1
    ctx["characters"]["Khun"]["last_seen"] = 100
    block = sc.inject_into_prompt(ctx, panel_index=200)
    # Khun (seen 100 panels ago) is marked stale but still listed
    assert "[last seen panel 100]" in block


def test_inject_big_cast_never_returns_empty(tmp_path):
    """R1: a 40-character cast must still produce a usable block -- the
    ladder shrinks item counts instead of dropping whole sections, and
    the result is never empty while the context holds anything."""
    ctx = sc._empty_context()
    for i in range(40):
        name = f"Character{i:02d}"
        ctx["characters"][name] = {
            "role": "minor", "description": "d" * 200,  # max-length desc
            "first_seen": 1, "last_seen": i,
            "aliases": [f"Alias{i}"], "status": "active",
            "associated_locations": [], "relationships": {},
        }
    ctx["story_threads"].append(
        {"id": "t1", "summary": "the long war continues", "status": "active"})
    block = sc.inject_into_prompt(ctx, panel_index=30)
    assert block != ""
    assert "[/STORY MEMORY]" in block                # footer never cut
    assert "maintain continuity" in block
    assert len(block) <= 1200                         # default cap honoured
    # threads outlast characters: the thread survives the ladder
    assert "the long war continues" in block


def test_inject_tiny_cap_keeps_thread_and_footer():
    """Even with a tiny max_chars the ladder degrades gracefully; the
    block keeps at least one thread line and the footer (it may exceed a
    pathological cap rather than drop the continuity instruction)."""
    ctx = sc._empty_context()
    ctx["characters"]["Bam"] = {
        "role": "protagonist", "description": "x" * 200, "first_seen": 1,
        "last_seen": 2, "aliases": [], "status": "active",
        "associated_locations": [], "relationships": {}}
    ctx["story_threads"].append(
        {"id": "t1", "summary": "the search continues", "status": "active"})
    block = sc.inject_into_prompt(ctx, panel_index=3, max_chars=300)
    assert "the search continues" in block
    assert "[/STORY MEMORY]" in block


# --------------------------------------------------------------- shape tolerance --
@pytest.mark.parametrize("entities,panel", [
    ({"characters_seen": "Bam"}, 7),                       # string not list
    ({"new_characters": {"name": "Solo"}}, 7),             # dict not list
    ({"new_characters": [{"name": 42}]}, 7),               # non-string name
    ({"locations_seen": None}, 7),                         # null
    ({"threads_new": {"not": "a list"}}, 7),               # dict coerced
    ({"new_events": ["ok", 42, None]}, 7),                 # mixed junk
    ("entirely not a dict", 7),                            # wrong type
])
def test_update_context_never_crashes_on_bad_shapes(tmp_path, entities, panel):
    ctx = seeded_ctx(tmp_path)
    before = json.dumps(ctx, sort_keys=True, default=str)
    sc.update_context(ctx, entities, panel)   # must not raise
    # nothing structurally destroyed
    assert set(ctx["characters"]) >= {"Bam", "Khun"}


def test_corrupt_meta_does_not_discard_update(tmp_path):
    """R10: per-field isolation -- a broken _meta must not lose the new
    character added in the same update."""
    ctx = seeded_ctx(tmp_path)
    ctx["_meta"]["last_panel"] = "garbage"    # int() would raise
    sc.update_context(ctx, {
        "new_characters": [{"name": "Endorsi", "role": "antagonist"}],
    }, panel_index=5)
    assert "Endorsi" in ctx["characters"]      # field survived
    assert ctx["_meta"]["last_panel"] == "garbage"  # meta untouched


def test_key_events_capped_newest_kept():
    ctx = sc._empty_context()
    for i in range(60):
        sc.update_context(ctx, {"new_events": [f"event {i}"]}, panel_index=i)
    assert len(ctx["key_events"]) == sc._MAX_KEY_EVENTS
    assert ctx["key_events"][-1]["summary"] == "event 59"
    assert ctx["key_events"][0]["summary"] == f"event {60 - sc._MAX_KEY_EVENTS}"


def test_scrub_fences_strips_fences():
    assert sc.scrub_fences("line\n```json\n{...}\n```\nend") == "line\nend"
    assert sc.scrub_fences("clean text") == "clean text"
    assert sc.scrub_fences(None) == ""


def test_sanitize_truncates_and_flattens():
    assert sc._sanitize("a\nb\nc", 3) == "a b"
    assert sc._sanitize(42) == ""
    assert sc._sanitize("x" * 500, 200) == "x" * 200


# ---------------------------------------------------------------- carry forward --
def test_carry_forward_resets_chapter_state_keeps_roster(tmp_path):
    prev = tmp_path / "ch46"
    new = tmp_path / "ch47"
    prev.mkdir()
    ctx = seeded_ctx(prev)
    sc.update_context(ctx, {
        "characters_seen": ["Bam"], "new_events": ["cliffhanger"],
    }, panel_index=50)
    ctx["chapter"] = 46

    carried = sc.carry_forward(prev, new, chapter=47)

    assert carried["characters"]["Bam"]["last_seen"] == 0   # reset
    assert carried["characters"]["Bam"]["last_seen"] == 0
    assert carried["key_events"] == []                     # chapter-specific
    assert carried["_meta"]["seed_built"] is False          # re-seeds
    assert carried["chapter"] == 47
    # roster + aliases + threads survive
    assert set(carried["characters"]) == {"Bam", "Khun"}
    assert "Twenty-Fifth Bam" in carried["characters"]["Bam"]["aliases"]
    assert carried["story_threads"][0]["status"] == "active"


# ------------------------------------------------------------------ cache key --
def test_prompt_sha_differentiates_memory_states():
    a = sc.prompt_sha("prompt\n[STORY MEMORY -- panel 1]\nBam")
    b = sc.prompt_sha("prompt\n[STORY MEMORY -- panel 2]\nBam, Khun")
    assert a != b and len(a) == 16


# ---------------------------------------------------------- make_text_model_call --
def test_make_text_model_call_uses_generate_text_with_fallback(monkeypatch):
    """R2: the adapter must call generate_text_with_fallback(system, user)
    with TWO arguments and return outcome.result."""
    import adapters.ai_models as ai
    calls: dict = {}

    class FakeOutcome:
        result = '{"characters": []}'
        model_used = "fake"
        fallback_used = False

    def fake_generate(system, user, **kw):
        calls["system"], calls["user"], calls["kw"] = system, user, kw
        return FakeOutcome()

    monkeypatch.setattr(ai, "generate_text_with_fallback", fake_generate)
    call = sc.make_text_model_call(api_key="k", model="m")
    out = call("SEED PROMPT")
    assert out == '{"characters": []}'
    assert calls["system"] == sc._SEED_SYSTEM
    assert calls["user"] == "SEED PROMPT"
    assert calls["kw"]["operation"] == "story-context-seed"
    assert calls["kw"]["primary_model"] == "m"


# ------------------------------------------------------------------- e2e flow --
def test_full_per_panel_flow(tmp_path):
    """Seed -> inject -> respond -> extract -> update -> save -> reload."""
    ctx = seeded_ctx(tmp_path)

    memory = sc.inject_into_prompt(ctx, panel_index=2)
    assert "Bam" in memory

    raw_response = json.dumps({
        "narration": "Bam takes his first steps inside the Tower.",
        "dialogue": "Headon: Welcome.",
        "entities": {
            "characters_seen": ["Bam", "twenty-fifth bam"],  # alias repeat ok
            "new_characters": [{"name": "Headon", "role": "antagonist",
                                 "description": "white fluffy guardian"}],
            "locations_seen": ["tower entrance"],
            "new_locations": [],
            "threads_new": ["Headon tests every new arrival"],
            "new_events": ["Headon greets Bam"],
        },
    })
    entities = sc.extract_entities_from_response(raw_response)
    assert entities is not None
    sc.update_context(ctx, entities, panel_index=2)
    sc.save_context(ctx, tmp_path)

    reloaded = sc.load_context(tmp_path)
    assert reloaded["characters"]["Bam"]["last_seen"] == 2
    assert reloaded["characters"]["Headon"]["role"] == "antagonist"
    assert reloaded["locations"]["Tower entrance"]["last_seen"] == 2
    assert len(reloaded["story_threads"]) == 2
    assert reloaded["key_events"][0]["panel"] == 2
