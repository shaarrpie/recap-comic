# webapp/jobs.py
"""Explicit job lifecycle for the recap-comic webapp.

Statuses: queued -> running -> completed | failed | cancelled
Every mutation is locked; every mutation records updated_at so the
frontend (and the watchdog) can detect dead workers. The per-job
log ring buffer is served to the UI, so a job can be followed from
start to finish without touching server logs.
"""
from __future__ import annotations

import contextlib
import enum
import json
import logging
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class JobStatus(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


class CancelledError(RuntimeError):
    """Raised inside a stage when the job was cancelled."""


class Job:
    def __init__(self, job_id: str, kind: str, config: dict[str, Any]):
        self.id = job_id
        self.kind = kind
        self.config = dict(config)
        self.status = JobStatus.QUEUED
        self.stage: str | None = None
        self.progress = 0
        self.error: str | None = None
        self.panels: list[dict] = []
        self.outputs: dict[str, str] = {}
        self.cancel_requested = False
        self.created_at = time.time()
        self.started_at: float | None = None
        self.updated_at = time.time()
        self.finished_at: float | None = None
        self._progress = 0
        self._logs: deque[dict] = deque(maxlen=300)

    @property
    def progress(self) -> int:
        return self._progress

    @progress.setter
    def progress(self, value: int) -> None:
        # Persist on change: pipeline stages set progress without always
        # logging right after, and a restart must not lose it.
        self._progress = int(value)
        store = getattr(self, "_store", None)
        if store is not None:
            store._save(self)

    def log(self, level: str, message: str, stage: str | None = None) -> None:
        line = {"t": round(time.time(), 3), "level": level,
                "stage": stage or self.stage, "msg": message}
        self._logs.append(line)
        store = getattr(self, "_store", None)
        if store is not None:
            store._save(self)

    def touch(self) -> None:
        self.updated_at = time.time()
        store = getattr(self, "_store", None)
        if store is not None:
            store._save(self)

    def check_cancelled(self) -> None:
        if self.cancel_requested:
            raise CancelledError("cancelled by user")

    def fail(self, message: str, tb: str | None = None) -> None:
        self.status = JobStatus.FAILED
        self.error = message
        self.finished_at = time.time()
        self.touch()
        self.log("ERROR", message)
        if tb:
            for ln in tb.strip().splitlines()[-12:]:
                self.log("ERROR", ln)

    def to_dict(self, include_logs: bool = False) -> dict[str, Any]:
        def _safe(v: Any) -> Any:
            if hasattr(v, "item"):
                return v.item()
            if isinstance(v, dict):
                return {k: _safe(val) for k, val in v.items()}
            if isinstance(v, list):
                return [_safe(x) for x in v]
            return v

        d: dict[str, Any] = {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status.value,
            "stage": self.stage,
            "progress": int(self.progress) if hasattr(self, "progress") else 0,
            "error": self.error,
            "panels": _safe(self.panels),
            "outputs": _safe(self.outputs),
            "created_at": float(self.created_at) if self.created_at else None,
            "updated_at": float(self.updated_at) if self.updated_at else None,
            "started_at": float(self.started_at) if self.started_at else None,
            "finished_at": float(self.finished_at) if self.finished_at else None,
        }
        if include_logs:
            d["logs"] = list(self._logs)
        return d


class JobStore:
    """Thread-safe store; single-user local tool.

    persistence (optional): when `persist_dir` is configured (see
    configure_persistence), every job snapshot — status, progress, config
    and the log ring buffer — is written to
    `<persist_dir>/<job_id>.json` on create/log/fail, and re-loaded on
    get() when the in-memory entry is gone. This is what makes the Logs
    view survive a uvicorn restart: without it, ANY restart turns every
    in-flight or historical job lookup into a 404 ("job no longer exists
    on the server (it was restarted?)"). Disk files are kept when
    in-memory entries are evicted, so history outlives the 40-job cap.
    """

    def __init__(self, max_jobs: int = 40, persist_dir: str | Path | None = None,
                 max_files: int = 200):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()   # guards snapshot writes
        self._max = max_jobs
        self._persist_dir: Path | None = None
        self._max_files = max_files
        if persist_dir is not None:
            self.configure_persistence(persist_dir)

    def configure_persistence(self, path: str | Path) -> None:
        self._persist_dir = Path(path)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

    def create(self, kind: str, config: dict[str, Any]) -> Job:
        job = Job(uuid.uuid4().hex[:12], kind, config)
        job._store = self  # type: ignore[attr-defined]
        with self._lock:
            if len(self._jobs) >= self._max:
                done = [j for j in self._jobs.values()
                        if j.status in TERMINAL]
                for j in sorted(done, key=lambda x: x.created_at)[:5]:
                    del self._jobs[j.id]
            self._jobs[job.id] = job
        self._save(job)
        self._prune_files()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            job = self._load(job_id)  # rehydrate after restart/eviction
            if job is not None:
                # Rehydrated jobs must behave like freshly-created ones:
                # without _store, mutations (log/touch/fail) after a
                # restart are memory-only and silently lost again.
                job._store = self  # type: ignore[attr-defined]
                with self._lock:
                    self._jobs[job_id] = job
        return job

    def snapshot(self) -> list[Job]:
        """Locked copy of the live jobs. Iterating store._jobs directly
        races concurrent create() -> "dict changed size during iteration"."""
        with self._lock:
            return list(self._jobs.values())

    def remove(self, job_id: str) -> None:
        """Drop a job from memory and its persistence file (upload
        failures: no phantom sessions in /api/projects)."""
        with self._lock:
            self._jobs.pop(job_id, None)
        if self._persist_dir is not None:
            with contextlib.suppress(OSError):
                (self._persist_dir / f"{job_id}.json").unlink()

    def get_by_session(self, session: str) -> Job | None:
        """Latest job (memory or disk) whose config.session == session.

        The Logs view addresses jobs by SESSION id, which only equals a
        job id for upload jobs — generate jobs get fresh UUIDs. Without
        this fallback, opening Logs for an older session 404s even when
        the server never restarted.
        """
        best: Job | None = None
        with self._lock:
            candidates = [j for j in self._jobs.values()
                          if j.config.get("session") == session or j.id == session]
        for job in candidates:
            if best is None or (job.updated_at or 0) > (best.updated_at or 0):
                best = job
        if self._persist_dir is not None:
            try:
                files = sorted(self._persist_dir.glob("*.json"),
                               key=lambda p: p.stat().st_mtime)
            except OSError:
                files = []
            for path in files:
                try:
                    data = json.loads(path.read_text("utf-8"))
                except (OSError, ValueError):
                    continue
                cfg = data.get("config", {})
                if cfg.get("session") != session and path.stem != session:
                    continue
                job = self._from_dict(data, mark_interrupted=False)
                if job is None:
                    continue
                job._store = self  # type: ignore[attr-defined]
                if best is None or (job.updated_at or 0) > (best.updated_at or 0):
                    best = job
        if best is not None:
            best._store = self  # type: ignore[attr-defined]
            with self._lock:
                self._jobs[best.id] = best
        return best

    # -- persistence -------------------------------------------------- #
    def _save(self, job: Job, *, immediate: bool | None = None) -> None:
        """Persist a job snapshot when its observable content changed.

        The old code serialized the FULL snapshot (status + config + the
        whole 300-line log ring) on every log()/touch()/progress call —
        O(logs^2) write volume per job. Snapshots are now change-gated by
        a compact fingerprint (status, stage, progress, error, timestamps,
        log count + last line id, config): identical consecutive calls
        cost one dict compare, and every real change still persists
        immediately (durability contract: a restart never loses a logged
        line). Terminal states always persist.
        """
        if self._persist_dir is None:
            return
        fp = (job.status, job.stage, int(job._progress), job.error,
              job.created_at, job.started_at, job.updated_at,
              job.finished_at, len(job._logs),
              job._logs[-1]["t"] if job._logs else 0.0,
              len(job.panels), repr(job.config))
        prev = getattr(job, "_persist_fp", None)
        if fp == prev and job.status in TERMINAL:
            # already durable at a terminal state, nothing new to write
            return
        if fp == prev and prev is not None:
            return  # no observable change since the last write
        job._persist_fp = fp  # type: ignore[attr-defined]
        self._write_snapshot(job)

    def flush(self, job: Job | None = None) -> None:
        """Force-write pending snapshots (used at worker exit; the
        fingerprint gate already persists every real change, so this is
        a belt-and-braces final sync)."""
        if self._persist_dir is None:
            return
        jobs = [job] if job is not None else []
        for j in jobs:
            j._persist_fp = None  # type: ignore[attr-defined]
            self._save(j)

    def _write_snapshot(self, job: Job) -> None:
        # Serialize snapshots: concurrent log/touch/progress calls from
        # multiple threads must not interleave on the same .tmp file
        # (corrupted JSON silently discards the job's history).
        with self._io_lock:
            try:
                tmp = self._persist_dir / f"{job.id}.json.tmp"
                tmp.write_text(json.dumps({
                    "job": job.to_dict(include_logs=True),
                    "config": job.config,
                }, indent=2), encoding="utf-8")
                tmp.replace(self._persist_dir / f"{job.id}.json")
            except OSError as exc:
                # Persistence must never break job processing.
                log.debug("job persist failed for %s: %s", job.id, exc)

    def _load(self, job_id: str) -> Job | None:
        if self._persist_dir is None:
            return None
        path = self._persist_dir / f"{job_id}.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            return None
        job = self._from_dict(data, mark_interrupted=True)
        if job is not None:
            job._store = self  # type: ignore[attr-defined]
        return job

    @staticmethod
    def _from_dict(data: dict, *, mark_interrupted: bool) -> Job | None:
        try:
            payload = data["job"]
            job = Job(payload["job_id"], payload.get("kind", ""),
                      dict(data.get("config", {})))
            job.status = JobStatus(payload.get("status", "failed"))
            job.stage = payload.get("stage")
            job.progress = payload.get("progress", 0)
            job.error = payload.get("error")
            job.panels = payload.get("panels", [])
            job.outputs = payload.get("outputs", {})
            job.created_at = payload.get("created_at") or time.time()
            job.started_at = payload.get("started_at")
            job.updated_at = payload.get("updated_at") or time.time()
            job.finished_at = payload.get("finished_at")
            for line in payload.get("logs", []):
                job._logs.append(line)
            if mark_interrupted and job.status not in TERMINAL:
                # The process died mid-run: say so instead of resurrecting
                # a zombie "running" job that will never progress.
                job.status = JobStatus.FAILED
                job.error = ("server restarted while this job was running; "
                             "outputs on disk may be partial — re-run to resume")
                job.finished_at = time.time()
                job.touch()
                job._logs.append({"t": round(time.time(), 3), "level": "ERROR",
                                  "stage": job.stage,
                                  "msg": job.error})
            return job
        except (KeyError, TypeError, ValueError):
            return None

    def _prune_files(self) -> None:
        if self._persist_dir is None:
            return
        try:
            files = sorted(self._persist_dir.glob("*.json"),
                           key=lambda p: p.stat().st_mtime)
        except OSError:
            return
        for stale in files[:-self._max_files]:
            with contextlib.suppress(OSError):
                stale.unlink()


store = JobStore()
