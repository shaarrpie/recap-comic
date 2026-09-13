# webapp/render_worker.py
"""Isolated, cancellable, resource-bounded renderer.

Architecture (req 50):  Web server -> JobManager -> RenderWorker -> ffmpeg.
One heavy render at a time (BoundedSemaphore); previous valid output is never
corrupted (render to .tmp, atomic replace); only THIS job's process group
is ever killed (never global taskkill); temp files cleaned in finally.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger("render_worker")

_RENDER_SEM = threading.BoundedSemaphore(1)     # one heavy render (req 39)
_ACTIVE: dict[str, RenderWorker] = {}
_LOCK = threading.Lock()


def render_profiles(threads_all: int) -> dict:    # req 40/41
    half = max(2, threads_all // 2)
    return {
        "performance": {"preset": "ultrafast", "crf": "26", "threads": "4",
                        "scale": "720:-2",  "fps": "24"},
        "balanced":    {"preset": "veryfast", "crf": "23", "threads": str(half),
                        "scale": None,      "fps": "30"},
        "quality":     {"preset": "medium",   "crf": "20", "threads": str(threads_all),
                        "scale": None,      "fps": "30"},
    }


class RenderWorker:
    def __init__(self, job_id: str, build_cmd, out_path: Path,
                 on_done=None, stall_seconds: int = 90,
                 cancel_event: threading.Event | None = None,
                 owns_render_slot: bool = True):
        self.job_id, self.on_done = job_id, on_done
        self.build_cmd = build_cmd
        self.out = Path(out_path)
        self.tmp = self.out.with_name(self.out.stem + ".tmp.mp4")
        self.proc: subprocess.Popen | None = None
        self.stall_seconds = stall_seconds
        self._last_size = -1
        self._last_grow = time.time()
        self._cancelled = False
        # on_done latch: a stall-cancel racing normal completion used to
        # deliver on_done twice (True then False -> spurious soft error).
        # First delivery wins; later calls are no-ops.
        self._done_delivered = threading.Event()
        self.cancel_event = cancel_event   # set => terminate (pipeline slot)
        # When the caller already holds the render slot (webapp.pipeline's
        # render_slot contextmanager), the worker must NOT re-acquire the
        # semaphore — BoundedSemaphore(1) would self-deadlock.
        self.owns_render_slot = owns_render_slot

    def _deliver(self, ok: bool, err: str | None = None) -> None:
        """Deliver on_done exactly once (first caller wins)."""
        if self._done_delivered.is_set() or self.on_done is None:
            return
        self._done_delivered.set()
        self.on_done(ok, err)

    @staticmethod
    def pick_strategy(n_panels: int, total_seconds: float) -> str:
        return "chunked" if (n_panels > 25 or total_seconds > 150) else "direct"

    def start(self):
        if self.owns_render_slot:
            _RENDER_SEM.acquire()
        with _LOCK:
            _ACTIVE[self.job_id] = self
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            cmd = self.build_cmd(self.tmp)
            log.info("[RENDER] job=%s cmd=%s", self.job_id, " ".join(cmd[:8]) + " ...")
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                start_new_session=True)
            stderr_tail: list[bytes] = []
            threading.Thread(target=self._drain_stderr,
                             args=(stderr_tail,), daemon=True).start()
            threading.Thread(target=self._heartbeat, daemon=True).start()
            rc = self.proc.wait()
            if self._cancelled:
                return
            if rc == 0 and self.tmp.is_file() and self.tmp.stat().st_size > 0:
                self.tmp.replace(self.out)
                log.info("[RENDER] job=%s success out=%s", self.job_id, self.out)
                self._deliver(True, None)
            else:
                self._cleanup_tmp()
                tail = b"".join(stderr_tail)[-1500:].decode("utf-8", "replace")
                self._deliver(False, f"ffmpeg exit {rc}\n{tail}")
        except Exception as exc:
            self._cleanup_tmp()
            if not self._cancelled:
                self._deliver(False, str(exc))
        finally:
            with _LOCK:
                _ACTIVE.pop(self.job_id, None)
            if self.owns_render_slot:
                _RENDER_SEM.release()

    def _drain_stderr(self, tail: list[bytes]):
        """Consume stderr, keeping the last ~1500 bytes for error reports."""
        try:
            if self.proc and self.proc.stderr:
                for line in iter(self.proc.stderr.readline, b""):
                    tail.append(line)
                    del tail[:-40]        # bound memory: keep last 40 lines
        except Exception:
            pass

    def _kill_process_group(self) -> None:
        """Terminate ffmpeg on any OS.

        POSIX: kill the process group started with start_new_session=True.
        Windows: os.killpg/getpgid/SIGKILL do not exist (AttributeError,
        not OSError) — use Popen.terminate() then kill() instead.
        """
        p = self.proc
        if p is None or p.poll() is not None:
            return
        if hasattr(os, "killpg"):
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        else:
            try:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
            except OSError:
                pass

    def _heartbeat(self):
        while self.proc and self.proc.poll() is None and not self._cancelled:
            if self.cancel_event is not None and self.cancel_event.is_set():
                self.cancel(reason="render slot released")
                return
            try:
                size = self.tmp.stat().st_size if self.tmp.is_file() else 0
            except OSError:
                size = 0
            if size > self._last_size:
                self._last_size, self._last_grow = size, time.time()
            elif time.time() - self._last_grow > self.stall_seconds:
                log.warning("[RENDER] job=%s stalled at %d bytes", self.job_id, size)
                self.cancel(reason="stalled")
                return
            time.sleep(3)

    def cancel(self, reason: str = "cancelled"):
        if self._cancelled:
            return
        self._cancelled = True
        self._kill_process_group()
        self._cleanup_tmp()
        self._deliver(False, reason)

    def _cleanup_tmp(self):
        try:
            if self.tmp.is_file():
                self.tmp.unlink()
        except OSError:
            pass


def cancel_render(job_id: str) -> bool:
    with _LOCK:
        w = _ACTIVE.get(job_id)
    if w is None:
        return False
    w.cancel()
    return True
