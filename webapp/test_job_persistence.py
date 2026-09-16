# webapp/test_job_persistence.py
"""Job records + log buffers must survive a server restart (disk persistence)."""
from __future__ import annotations

from pathlib import Path

from webapp.jobs import JobStatus, JobStore


def _store(tmp_path: Path) -> JobStore:
    return JobStore(persist_dir=tmp_path / "jobs")


def test_logs_survive_restart(tmp_path: Path) -> None:
    s1 = _store(tmp_path)
    job = s1.create("generate", {"session": "sess-1"})
    job.log("INFO", "stage=segment_panels started", "segment_panels")
    job.log("WARNING", "low confidence", "segment_panels")
    job.progress = 30

    # Simulate a restart: brand-new store over the same dir.
    s2 = _store(tmp_path)
    got = s2.get(job.id)
    assert got is not None
    assert got.config["session"] == "sess-1"
    assert got.progress == 30
    msgs = [ln["msg"] for ln in got.to_dict(include_logs=True)["logs"]]
    assert "stage=segment_panels started" in msgs
    assert "low confidence" in msgs


def test_running_job_marked_interrupted_after_restart(tmp_path: Path) -> None:
    s1 = _store(tmp_path)
    job = s1.create("generate", {"session": "sess-2"})
    job.status = JobStatus.RUNNING
    job.log("INFO", "working", "tts_audio")

    s2 = _store(tmp_path)
    got = s2.get(job.id)
    assert got is not None
    assert got.status == JobStatus.FAILED
    assert "restarted" in (got.error or "")
    assert any("restarted" in ln["msg"] for ln in got.to_dict(include_logs=True)["logs"])


def test_completed_job_keeps_status(tmp_path: Path) -> None:
    s1 = _store(tmp_path)
    job = s1.create("segment", {"session": "sess-3"})
    job.status = JobStatus.COMPLETED
    job.log("INFO", "done", "done")

    got = _store(tmp_path).get(job.id)
    assert got is not None and got.status == JobStatus.COMPLETED


def test_get_by_session_finds_generate_job(tmp_path: Path) -> None:
    s1 = _store(tmp_path)
    upload = s1.create("segment", {"session": "sess-4"})
    gen = s1.create("generate", {"session": "sess-4"})
    gen.log("INFO", "generate log line", "build_script")

    # Fresh process: lookup by SESSION id (what the Logs view sends).
    s2 = _store(tmp_path)
    got = s2.get_by_session("sess-4")
    assert got is not None
    assert got.id == gen.id  # latest job for the session, not the upload
    assert any("generate log line" in ln["msg"]
               for ln in got.to_dict(include_logs=True)["logs"])
    assert upload.id != gen.id


def test_get_by_session_resolves_clock_tie_by_seq(tmp_path: Path) -> None:
    """Two jobs created inside one clock tick must still resolve to the newest.

    On Windows time.time() has ~15.6ms resolution, so a segment job and the
    generate job that follows it can tie on BOTH created_at and updated_at.
    The timestamp-only ordering then fell back to arbitrary glob order and
    could return the older job. The persisted seq is the deterministic
    tie-break; this test pins the timestamps to force the tie.
    """
    s1 = _store(tmp_path)
    upload = s1.create("segment", {"session": "sess-tie"})
    gen = s1.create("generate", {"session": "sess-tie"})

    # Simulate one shared clock tick for both jobs and persist it.
    assert gen.seq > upload.seq
    for job in (upload, gen):
        job.created_at = 1.0
        job.updated_at = 1.0
        s1._save(job, immediate=True)

    # In-memory lookup prefers the later-created job.
    assert s1.get_by_session("sess-tie").id == gen.id
    # Fresh store (disk-only path) must agree.
    got = _store(tmp_path).get_by_session("sess-tie")
    assert got is not None
    assert got.id == gen.id


def test_no_persistence_dir_keeps_old_behavior(tmp_path: Path) -> None:
    s = JobStore()
    job = s.create("generate", {"session": "x"})
    assert s.get(job.id) is job
    assert s.get("missing") is None
    assert s.get_by_session("x") is job
    assert s.get_by_session("nope") is None
