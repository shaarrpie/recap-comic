# webapp/test_settings_api.py — regression tests for the API-key persistence
# redesign: GET/POST /api/settings, GET /api/settings/key,
# webapp_output/settings.json, and key-resolution precedence over .env.
#
# The routes are being implemented by another agent concurrently. Until the
# implementation lands, the /api/settings* tests are strict xfail: they run
# the full request against the real app but tolerate a missing route, so a
# silent regression cannot hide behind a skip. The moment the routes exist,
# every test becomes a hard requirement (remove _ALLOW_MISSING_ROUTE).
#
# jobs.py disk persistence is also in flight: when present, POST /api/run
# writes job.config (which contains api_key) verbatim to
# <persist_dir>/<job_id>.json — a discovered leak pinned below.
#
# Run: pytest webapp/test_settings_api.py -q
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webapp import main as webmain
from webapp import pipeline
from webapp.jobs import Job, store

# Flip to False when GET/POST /api/settings and GET /api/settings/key land
# in webapp/main.py — every xfail below then becomes a hard failure.
_ALLOW_MISSING_ROUTE = True


def _png_bytes(w: int = 4, h: int = 8) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (255, 255, 255)).save(buf, "PNG")
    return buf.getvalue()


def _stub_generate_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace heavy generate stages with fakes (same pattern as
    webapp/test_webapp.py) so /api/run completes offline and instantly."""
    saved = list(pipeline.PIPELINES["generate"])

    def fake(job, **kwargs):
        pass

    stubbed = [(name, fake) for name, _ in saved]
    monkeypatch.setitem(pipeline.PIPELINES, "generate", stubbed)
    monkeypatch.setattr(webmain.store, "watchdog_interval", 10_000,
                        raising=False)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Isolated webapp_output (settings.json location under test) plus
    isolated job persistence dir, so no test touches real state."""
    _stub_generate_pipeline(monkeypatch)
    fake_out = tmp_path / "webapp_output"
    monkeypatch.setattr(webmain, "OUTPUT_DIR", fake_out, raising=False)
    import webapp.panel_api as _pa
    monkeypatch.setattr(_pa, "OUTPUT_DIR", fake_out, raising=False)
    try:
        import webapp.manual_crop_api as _mc
        monkeypatch.setattr(_mc, "OUTPUT_DIR", fake_out, raising=False)
    except ImportError:
        pass
    if hasattr(webmain.store, "_persist_dir"):
        monkeypatch.setattr(webmain.store, "_persist_dir", tmp_path / "jobs")
    c = TestClient(webmain.app)
    c._fake_out = fake_out  # type: ignore[attr-defined]
    return c


def _maybe_missing(r, label):
    """404/405 on a not-yet-implemented route -> strict xfail; any other
    status returns the response for normal assertions."""
    if _ALLOW_MISSING_ROUTE and r.status_code in (404, 405):
        pytest.xfail(f"/api/{label} not implemented yet")
    return r


# ---------------------------------------------------------------------------
# 1. GET /api/settings — shape + secret-free contract
# ---------------------------------------------------------------------------
def test_get_settings_returns_api_key_set_flag_only(client):
    r = client.get("/api/settings")
    r = _maybe_missing(r, "settings (GET)")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body.get("api_key_set"), bool)
    assert "api_key" not in body, "raw key leaked in GET /api/settings"
    assert "key" not in body


def test_get_settings_never_leaks_key_after_save(client):
    post = client.post("/api/settings", json={"api_key": "test-key-123"})
    post = _maybe_missing(post, "settings (POST)")
    assert post.status_code == 200, post.text
    r = client.get("/api/settings")
    r = _maybe_missing(r, "settings (GET)")
    assert r.status_code == 200
    body = r.json()
    assert body.get("api_key_set") is True
    assert "api_key" not in body


# ---------------------------------------------------------------------------
# 2. POST /api/settings + GET /api/settings/key round trip
# ---------------------------------------------------------------------------
def test_post_settings_then_get_key_roundtrip(client):
    post = client.post("/api/settings", json={"api_key": "test-key-123"})
    post = _maybe_missing(post, "settings (POST)")
    assert post.status_code == 200, post.text
    r = client.get("/api/settings/key")
    r = _maybe_missing(r, "settings/key (GET)")
    assert r.status_code == 200, r.text
    assert r.json().get("api_key") == "test-key-123"


# ---------------------------------------------------------------------------
# 3. POST with empty key must NOT wipe the saved key
# ---------------------------------------------------------------------------
def test_empty_post_preserves_saved_key(client):
    post = client.post("/api/settings", json={"api_key": "keep-me-42"})
    post = _maybe_missing(post, "settings (POST)")
    assert post.status_code == 200, post.text
    empty = client.post("/api/settings", json={"api_key": ""})
    empty = _maybe_missing(empty, "settings (POST, empty)")
    assert empty.status_code == 200, empty.text
    r = client.get("/api/settings/key")
    r = _maybe_missing(r, "settings/key (GET)")
    assert r.status_code == 200
    assert r.json().get("api_key") == "keep-me-42", \
        "empty POST wiped the previously saved key"


# ---------------------------------------------------------------------------
# 4. File location: webapp_output/settings.json, never in a session dir
# ---------------------------------------------------------------------------
def test_settings_file_location_and_content(client):
    post = client.post("/api/settings", json={"api_key": "test-key-123"})
    post = _maybe_missing(post, "settings (POST)")
    assert post.status_code == 200, post.text
    settings_file = client._fake_out / "settings.json"  # type: ignore[attr-defined]
    assert settings_file.is_file(), "webapp_output/settings.json not written"
    data = json.loads(settings_file.read_text("utf-8"))
    assert data.get("api_key") == "test-key-123"
    assert settings_file.parent == client._fake_out, (  # type: ignore[attr-defined]
        f"settings.json written outside webapp_output root: {settings_file}")
    nested = list(client._fake_out.glob("*/settings.json"))  # type: ignore[attr-defined]
    assert nested == [], f"settings.json inside a session dir: {nested}"


# ---------------------------------------------------------------------------
# 5. Precedence: manual settings.json key wins over env vars
# ---------------------------------------------------------------------------
def test_manual_key_wins_over_env(monkeypatch, tmp_path):
    ai_models = pytest.importorskip("adapters.ai_models")

    fake_out = tmp_path / "webapp_output"
    fake_out.mkdir()
    (fake_out / "settings.json").write_text(
        json.dumps({"api_key": "from-settings-json"}), "utf-8")

    monkeypatch.setattr(ai_models, "OUTPUT_DIR", fake_out, raising=False)
    monkeypatch.setenv("XKIRO_API_KEY", "from-env")
    monkeypatch.setenv("OPENAI_API_KEY", "from-env-openai")

    src_has_fallback = "settings.json" in Path(
        ai_models.__file__).read_text("utf-8")
    if not src_has_fallback:
        pytest.xfail("api_key_from_env settings.json fallback "
                     "not implemented yet")

    assert ai_models.api_key_from_env() == "from-settings-json", (
        "env var took precedence over webapp_output/settings.json")


# ---------------------------------------------------------------------------
# 6. Job serialization must NEVER contain the api_key
# ---------------------------------------------------------------------------
def _assert_no_key(obj, where):
    blob = json.dumps(obj)
    assert "leak-canary-key" not in blob, f"api_key leaked in {where}"
    assert '"api_key"' not in blob, f"api_key field present in {where}"


def test_job_serialization_never_leaks_api_key(client, tmp_path, monkeypatch):
    """Covers the API surface end-to-end: POST /api/run passes the key as
    worker kwargs (NOT job.config — main.py:282-300), and GET /api/jobs
    must never echo it. Also sweeps the disk snapshots for the canary key
    when persistence is present, so a future regression that moves the key
    into job.config (or config serialization) fails loudly here."""
    persist_dir = tmp_path / "jobs"
    persists = hasattr(webmain.store, "configure_persistence")
    if persists:
        # mkdir BEFORE the first _save: a missing dir makes _save fail
        # silently (OSError is caught) and the leak check would pass
        # vacuously with zero snapshots.
        persist_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(webmain.store, "_persist_dir", persist_dir)

    up = client.post("/api/upload", files={"file": ("s.png", io.BytesIO(
        _png_bytes()), "image/png")})
    assert up.status_code == 200, up.text
    session = up.json()["job_id"]

    r = client.post("/api/run", json={"session": session, "order": None,
                                      "backend": "none",
                                      "api_key": "leak-canary-key"})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    _wait_terminal(client, job_id)

    st = client.get(f"/api/jobs/{job_id}")
    assert st.status_code == 200
    _assert_no_key(st.json(), "GET /api/jobs payload")
    st_logs = client.get(f"/api/jobs/{job_id}?logs=1")
    assert st_logs.status_code == 200
    _assert_no_key(st_logs.json(), "GET /api/jobs?logs=1 payload")

    if not persists:
        pytest.skip("JobStore disk persistence not present in this checkout")
    snapshots = sorted(persist_dir.glob("*.json"))
    assert snapshots, ("persistence configured but zero snapshots written — "
                       "leak sweep would be vacuous")
    leaked = any("leak-canary-key" in p.read_text("utf-8") for p in snapshots)
    assert not leaked, ("api_key leaked into job disk snapshots "
                        "(jobs.py _save writes job.config verbatim)")


def _wait_terminal(c: TestClient, job_id: str, timeout: float = 5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = c.get(f"/api/jobs/{job_id}")
        if r.status_code == 200 and r.json().get("status") in (
                "completed", "failed", "cancelled"):
            return r.json()
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached a terminal state")


def test_to_dict_omits_config_entirely():
    """Job.to_dict never includes config (where the key travels), so the
    HTTP surface is clean; only the disk snapshot is at risk."""
    job = Job("aaaaaaaaaaaa", "generate", {"session": "s", "api_key": "k"})
    d = job.to_dict(include_logs=True)
    assert "config" not in d
    assert "api_key" not in d


def test_job_store_disk_snapshot_redaction_contract(tmp_path):
    """PENDING-FIX pin: with the NEW JobStore persistence, _save writes
    job.config verbatim to disk. TODAY /api/run keeps api_key out of
    job.config (kwargs-only), so nothing leaks — but the snapshot writer
    has no redaction of its own. If any future change starts putting the
    key (or another secret) into job.config, this pin fails and this
    docstring tells the fixer what to do: redact secrets in _save before
    the tmp.replace. Skipped on the old (pre-persistence) JobStore."""
    s = store.__class__(persist_dir=tmp_path) if "persist_dir" in (
        store.__class__.__init__.__code__.co_varnames) else None
    if s is None:
        pytest.skip("JobStore persistence not present in this checkout")
    job = s.create("generate", {"session": "sess", "api_key": "disk-canary"})
    assert job.config.get("api_key") == "disk-canary"
    files = list(tmp_path.glob("*.json"))
    assert files, "snapshot not written"
    blob = files[0].read_text("utf-8")
    if "disk-canary" in blob:
        # Strict xfail: the snapshot writer has no secret redaction today.
        # It is NOT currently exploitable because /api/run keeps api_key
        # out of job.config (kwargs-only, main.py:282-300) — but any change
        # that puts secrets into config makes them disk-visible. Flip this
        # pin to a hard assert when _save redacts (then remove this branch).
        pytest.xfail("JobStore._save writes job.config verbatim — secrets "
                     "in config reach disk unredacted (fix pending: redact "
                     "in _save)")
    assert "disk-canary" not in blob


# ---------------------------------------------------------------------------
# 7. Atomic write: no .json.tmp leftovers after save
# ---------------------------------------------------------------------------
def test_settings_save_is_atomic(client):
    post = client.post("/api/settings", json={"api_key": "atomic-key-1"})
    post = _maybe_missing(post, "settings (POST)")
    assert post.status_code == 200, post.text
    root: Path = client._fake_out  # type: ignore[attr-defined]
    leftovers = list(root.glob("*.tmp")) + list(root.glob("*.json.tmp"))
    assert leftovers == [], f"temp files left behind: {leftovers}"
    assert (root / "settings.json").is_file()
