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


class _RevMixin:
    """Bumps the owner Job's revision counter on every mutation.

    The persistence fingerprint (JobStore._save) must notice in-place edits
    to the mutable payloads: `job.outputs["x"] = "y"` and `job.panels = [...]`
    (or an append) are real observable changes, and without a revision counter
    they were only written if some later status/progress change happened to
    fire — an edit could sit in memory until an unrelated flush.
    """

    _owner: Job

    def _bump(self) -> None:
        owner = getattr(self, "_owner", None)
        if owner is not None:
            owner._mutated()


class _RevDict(_RevMixin, dict):
    def __init__(self, owner: Job, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._owner = owner

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        self._bump()

    def __delitem__(self, key: Any) -> None:
        super().__delitem__(key)
        self._bump()

    def pop(self, *args: Any) -> Any:
        result = super().pop(*args)
        self._bump()
        return result

    def popitem(self) -> Any:
        result = super().popitem()
        self._bump()
        return result

    def clear(self) -> None:
        super().clear()
        self._bump()

    def update(self, *args: Any, **kwargs: Any) -> None:
        super().update(*args, **kwargs)
        self._bump()

    def setdefault(self, key: Any, default: Any = None) -> Any:
        result = super().setdefault(key, default)
        self._bump()
        return result


class _RevList(_RevMixin, list):
    def __init__(self, owner: Job, *args: Any) -> None:
        super().__init__(*args)
        self._owner = owner

    def __setitem__(self, index: Any, value: Any) -> None:
        super().__setitem__(index, value)
        self._bump()

    def __delitem__(self, index: Any) -> None:
        super().__delitem__(index)
        self._bump()

    def append(self, value: Any) -> None:
        super().append(value)
        self._bump()

    def extend(self, values: Any) -> None:
        super().extend(values)
        self._bump()

    def insert(self, index: int, value: Any) -> None:
        super().insert(index, value)
        self._bump()

    def pop(self, *args: Any) -> Any:
        result = super().pop(*args)
        self._bump()
        return result

    def remove(self, value: Any) -> None:
        super().remove(value)
        self._bump()

    def clear(self) -> None:
        super().clear()
        self._bump()

    def sort(self, *args: Any, **kwargs: Any) -> None:
        super().sort(*args, **kwargs)
        self._bump()

    def reverse(self) -> None:
        super().reverse()
        self._bump()


class Job:
    def __init__(self, job_id: str, kind: str, config: dict[str, Any]):
        self.id = job_id
        self.kind = kind
        self.config = dict(config)
        self.status = JobStatus.QUEUED
        self.stage: str | None = None
        # Plain backing fields: `progress`/`panels`/`outputs` are properties
        # below, so __init__ must NOT assign through them (Job.progress = 0
        # here would invoke the setter before _store/_progress exist; worse,
        # a property + same-name instance attribute silently breaks the
        # setter path and mutations stop persisting).
        self._progress_count = 0
        self.error: str | None = None
        self._rev = 0              # bumped by _RevDict/_RevList mutations
        self._panels = _RevList(self)
        self._outputs = _RevDict(self)
        self.cancel_requested = False
        # Strictly-increasing creation order, assigned by JobStore.create().
        # Wall-clock timestamps alone CANNOT order jobs: on Windows
        # time.time() has ~15.6ms ticks, so two jobs created in the same
        # tick tie on BOTH created_at and updated_at, and get_by_session /
        # _session_index then fall back to arbitrary glob order and can
        # return the OLDER job. seq is the deterministic tie-break.
        self.seq: int = 0
        self.created_at = time.time()
        self.started_at: float | None = None
        self.updated_at = time.time()
        self.finished_at: float | None = None
        self._progress = 0
        self._logs: deque[dict] = deque(maxlen=300)

    def _mutated(self) -> None:
        """Record a mutation of a mutable payload and persist it.

        Used by the revision-aware panels/outputs containers and their
        setters. Panels are exactly what the review UI reads back, so an
        in-place edit (narration rewrite, crop adjustment) must be durable on
        its own — not only when an unrelated status/progress change happens
        to fire later.
        """
        self._rev += 1
        store = getattr(self, "_store", None)
        if store is not None:
            store._save(self)

    @property
    def panels(self) -> list[dict]:
        return self._panels

    @panels.setter
    def panels(self, value: list[dict] | None) -> None:
        # Assign a fresh revision-aware list (see _RevList): in-place edits
        # and plain re-assignment both have to be visible to _save().
        self._panels = _RevList(self, list(value or []))
        self._mutated()

    @property
    def outputs(self) -> dict[str, str]:
        return self._outputs

    @outputs.setter
    def outputs(self, value: dict[str, str] | None) -> None:
        self._outputs = _RevDict(self, dict(value or {}))
        self._mutated()

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
            "seq": int(self.seq),
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
        # Cached {session: (job_id, updated_at, created_at, seq)} disk index;
        # see _session_index(). None means "rescan on next lookup".
        self._index_cache: tuple[float, dict[str, tuple[str, float, float, int]]] | None = None
        self._SESSION_INDEX_TTL_S = 2.0
        # Monotonic job sequence: seeded from the highest seq on disk so a
        # restarted store keeps assigning numbers above every persisted job.
        self._seq = 0
        self._seq_seeded = False
        if persist_dir is not None:
            self.configure_persistence(persist_dir)

    def configure_persistence(self, path: str | Path) -> None:
        self._persist_dir = Path(path)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

    def _next_seq(self) -> int:
        """Next creation-order number, continuing above any persisted job.

        Seeds from the max seq on disk on first use so a fresh store over
        an existing persist_dir does not hand out numbers that sort below
        old snapshots (that would let a restarted lookup pick a stale job
        over the newly created one).
        """
        with self._lock:
            if not self._seq_seeded and self._persist_dir is not None:
                self._seq_seeded = True
                hi = 0
                with contextlib.suppress(OSError):
                    for path in self._persist_dir.glob("*.json"):
                        try:
                            payload = (json.loads(path.read_text("utf-8"))
                                       .get("job") or {})
                            hi = max(hi, int(payload.get("seq") or 0))
                        except (OSError, ValueError, TypeError):
                            continue
                self._seq = hi
            self._seq += 1
            return self._seq

    def create(self, kind: str, config: dict[str, Any]) -> Job:
        job = Job(uuid.uuid4().hex[:12], kind, config)
        job.seq = self._next_seq()
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
            self._index_cache = None

    def get_by_session(self, session: str) -> Job | None:
        """Latest job (memory or disk) whose config.session == session.

        The Logs view addresses jobs by SESSION id, which only equals a
        job id for upload jobs — generate jobs get fresh UUIDs. Without
        this fallback, opening Logs for an older session 404s even when
        the server never restarted.

        Disk lookups go through a cached session index: the previous code
        globbed and json.loads'ed EVERY snapshot (up to max_files) on each
        call, and the Logs UI polls this endpoint.
        """
        best: Job | None = None
        best_key: tuple[float, float, int] = (0.0, 0.0, 0)
        with self._lock:
            candidates = [j for j in self._jobs.values()
                          if j.config.get("session") == session or j.id == session]
        for job in candidates:
            key = (job.updated_at or 0.0, job.created_at or 0.0,
                   getattr(job, "seq", 0) or 0)
            if best is None or key > best_key:
                best, best_key = job, key
        if self._persist_dir is not None:
            entry = self._session_index().get(session)
            if entry is not None:
                job = self._load_snapshot(entry[0])
                if job is not None:
                    job._store = self  # type: ignore[attr-defined]
                    key = (job.updated_at or 0.0, job.created_at or 0.0,
                           getattr(job, "seq", 0) or 0)
                    if best is None or key > best_key:
                        best, best_key = job, key
        if best is not None:
            best._store = self  # type: ignore[attr-defined]
            with self._lock:
                self._jobs[best.id] = best
        return best

    def _load_snapshot(self, job_id: str) -> Job | None:
        """Load one snapshot without the interrupted-marking side effect."""
        if self._persist_dir is None:
            return None
        path = self._persist_dir / f"{job_id}.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            return None
        return self._from_dict(data, mark_interrupted=False)

    def _session_index(self) -> dict[str, tuple[str, float, float, int]]:
        """{session: (job_id, updated_at, created_at, seq)} for every snapshot.

        Cached for a few seconds (the Logs view polls) and invalidated
        whenever a snapshot is written or removed, so a freshly created job
        is never hidden by a stale index. The seq tie-break makes "latest
        job for this session" deterministic: wall-clock timestamps tie on
        Windows (~15.6ms time.time() resolution), and a strict > on tied
        timestamps kept whichever file glob() happened to return first —
        which could be the OLDER job.
        """
        now = time.time()
        cached = self._index_cache
        if cached is not None and now - cached[0] < self._SESSION_INDEX_TTL_S:
            return cached[1]
        index: dict[str, tuple[str, float, float, int]] = {}
        if self._persist_dir is not None:
            try:
                files = list(self._persist_dir.glob("*.json"))
            except OSError:
                files = []
            for path in files:
                try:
                    data = json.loads(path.read_text("utf-8"))
                except (OSError, ValueError):
                    continue
                payload = data.get("job") or {}
                sess = (data.get("config") or {}).get("session") or path.stem
                updated = float(payload.get("updated_at") or 0)
                created = float(payload.get("created_at") or 0)
                seq = int(payload.get("seq") or 0)
                prev = index.get(sess)
                if prev is None or (updated, created, seq) > (prev[1], prev[2], prev[3]):
                    index[sess] = (path.stem, updated, created, seq)
        self._index_cache = (now, index)
        return index

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
              job._rev, repr(job.config))
        prev = getattr(job, "_persist_fp", None)
        if fp == prev and job.status in TERMINAL:
            # already durable at a terminal state, nothing new to write
            return
        if fp == prev and prev is not None:
            return  # no observable change since the last write
        job._persist_fp = fp  # type: ignore[attr-defined]
        self._write_snapshot(job)

    def flush(self, job: Job | None = None) -> None:
        """Force-write pending snapshots.

        `job=None` used to be a silent no-op (`jobs = []`) even though the
        docstring promised a store-wide sync, so a caller relying on it at
        worker exit wrote nothing. With no argument it now flushes every
        in-memory job.
        """
        if self._persist_dir is None:
            return
        jobs = [job] if job is not None else self.snapshot()
        for j in jobs:
            j._persist_fp = None  # type: ignore[attr-defined]
            self._save(j)

    # Credential detection for persisted snapshots lives in
    # adapters/_logging (is_sensitive_key), shared with the log sanitizer.

    @classmethod
    def _redact_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Copy of job.config with credential fields blanked. The in-memory
        job keeps the real key (the worker needs it); only the persisted
        snapshot is redacted. A rehydrated job therefore resumes with
        key="" — same behavior as a restart after a server that never knew
        the key; run steps read the key from settings/.env again.

        Detection is shared with adapters/_logging.sanitize so the two
        redaction sites cannot drift; the old local marker list matched
        "auth" as a substring and blanked an innocent "author" key.
        """
        from adapters._logging import is_sensitive_key

        out = dict(config or {})
        for k in list(out):
            if not out[k]:
                continue
            if is_sensitive_key(k):
                out[k] = ""
        return out

    def _write_snapshot(self, job: Job) -> None:
        # Serialize snapshots: concurrent log/touch/progress calls from
        # multiple threads must not interleave on the same .tmp file
        # (corrupted JSON silently discards the job's history).
        with self._io_lock:
            try:
                tmp = self._persist_dir / f"{job.id}.json.tmp"
                tmp.write_text(json.dumps({
                    "job": job.to_dict(include_logs=True),
                    "config": self._redact_config(job.config),
                }, indent=2), encoding="utf-8")
                tmp.replace(self._persist_dir / f"{job.id}.json")
                self._index_cache = None      # a new snapshot changes the index
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
            self._persist_if_restarted(job)
        return job

    def _persist_if_restarted(self, job: Job) -> None:
        """Persist the 'server restarted' marking that _from_dict derived.

        _from_dict runs BEFORE _store is attached, so the `job.touch()` it
        used to call could not write anything (touch() no-ops without a
        store): the FAILED/restarted state was memory-only and got
        re-derived on every rehydrate, while the snapshot on disk kept
        saying "running". Attach the store first, then persist.
        """
        if not getattr(job, "_needs_persist", False):
            return
        job._needs_persist = False
        job.touch()

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
            job.seq = int(payload.get("seq") or 0)
            job.started_at = payload.get("started_at")
            job.updated_at = payload.get("updated_at") or time.time()
            job.finished_at = payload.get("finished_at")
            for line in payload.get("logs", []):
                job._logs.append(line)
            if mark_interrupted and job.status not in TERMINAL:
                # The process died mid-run: say so instead of resurrecting
                # a zombie "running" job that will never progress. The write
                # itself happens in JobStore._persist_if_restarted(), once
                # this job has a _store to write through.
                job.status = JobStatus.FAILED
                job.error = ("server restarted while this job was running; "
                             "outputs on disk may be partial — re-run to resume")
                job.finished_at = time.time()
                job._needs_persist = True
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
