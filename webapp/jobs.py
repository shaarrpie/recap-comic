# webapp/jobs.py
"""Explicit job lifecycle for the recap-comic webapp.

Statuses: queued -> running -> completed | failed | cancelled
Every mutation is locked; every mutation records updated_at so the
frontend (and the watchdog) can detect dead workers. The per-job
log ring buffer is served to the UI, so a job can be followed from
start to finish without touching server logs.
"""
from __future__ import annotations

import enum
import threading
import time
import uuid
from collections import deque
from typing import Any


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
        self._logs: deque[dict] = deque(maxlen=300)

    def log(self, level: str, message: str, stage: str | None = None) -> None:
        line = {"t": round(time.time(), 3), "level": level,
                "stage": stage or self.stage, "msg": message}
        self._logs.append(line)

    def touch(self) -> None:
        self.updated_at = time.time()

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
        d = {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status.value,
            "stage": self.stage,
            "progress": self.progress,
            "error": self.error,
            "panels": self.panels,
            "outputs": self.outputs,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if include_logs:
            d["logs"] = list(self._logs)
        return d


class JobStore:
    """Thread-safe in-memory store; single-user local tool."""

    def __init__(self, max_jobs: int = 40):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._max = max_jobs

    def create(self, kind: str, config: dict[str, Any]) -> Job:
        job = Job(uuid.uuid4().hex[:12], kind, config)
        with self._lock:
            if len(self._jobs) >= self._max:
                done = [j for j in self._jobs.values()
                        if j.status in TERMINAL]
                for j in sorted(done, key=lambda x: x.created_at)[:5]:
                    del self._jobs[j.id]
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


store = JobStore()
