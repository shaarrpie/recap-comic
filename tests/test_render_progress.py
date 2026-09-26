# tests/test_render_progress.py
"""Live render-progress plumbing (ffmpeg `time=` parsing -> callbacks).

Covers the contract added for the render loading screen:
- adapters.render_ffmpeg: parse_ffmpeg_time / timeline_seconds /
  _run_ffmpeg_progress, and the rule that render()/render_chunked() keep the
  ORIGINAL blocking subprocess.run path whenever progress_cb is None.
- webapp.render_worker: RenderWorker._drain_stderr turns streamed ffmpeg
  lines into throttled progress callbacks without ever breaking the tail
  capture that error reports depend on.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import adapters.render_ffmpeg as rf
from adapters.schemas import BBox, Meta, PanSpec, TimelineArtifact, TimelineEntry


def _tl(n: int = 1, dur: float = 2.0) -> TimelineArtifact:
    return TimelineArtifact(
        meta=Meta(schema_version="1", generator="t", config_hash="h",
                  input_hashes={}),
        width=1080, height=1920, fps=30, gap_seconds=0.0,
        min_display_seconds=1.0,
        entries=[TimelineEntry(panel_id=f"p{i + 1}", order=i + 1,
                               source_image="x.png",
                               bbox=BBox(x=0, y=0, w=1080, h=1920),
                               start_seconds=i * dur, duration_seconds=dur,
                               audio_path=None,
                               pan=PanSpec(kind="static", scaled_w=1080,
                                           scaled_h=1920, travel_px=0))
                 for i in range(n)])


# ------------------------------------------------------------- parse_ffmpeg_time
def test_parse_ffmpeg_time_standard_line():
    line = "frame= 2507 fps= 61 q=-1.0 size=   12345kB time=00:01:23.45 bitrate= 812.3k"
    assert rf.parse_ffmpeg_time(line) == pytest.approx(83.45)


def test_parse_ffmpeg_time_hours_and_comma_decimal():
    assert rf.parse_ffmpeg_time("time=01:02:03.50") == pytest.approx(3723.5)
    # some locales build ffmpeg with comma separators
    assert rf.parse_ffmpeg_time("time=00:00:10,25") == pytest.approx(10.25)


def test_parse_ffmpeg_time_no_match_returns_none():
    assert rf.parse_ffmpeg_time("frame= 100 fps= 61 q=28.0") is None
    assert rf.parse_ffmpeg_time("") is None
    assert rf.parse_ffmpeg_time(None) is None


def test_timeline_seconds_sums_entries():
    assert rf.timeline_seconds(_tl(n=3, dur=2.5)) == pytest.approx(7.5)


# --------------------------------------------------------- _run_ffmpeg_progress
class _FakeProc:
    """Minimal Popen stand-in: stderr iterates preset lines, wait() is instant."""

    def __init__(self, lines, rc=0):
        self._lines = lines
        self.stderr = iter(self)
        self.returncode = rc
        self.killed = False

    def __iter__(self):
        yield from self._lines

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True

    def terminate(self):
        self.killed = True


def test_run_ffmpeg_progress_reports_fractions(monkeypatch):
    procs = []
    fake = _FakeProc([
        "frame=  150 fps= 59 time=00:00:05.00 bitrate=\n",
        "video:120kB audio:88kB\n",                     # no time= -> ignored
        "frame=  300 fps= 58 time=00:00:10.00 bitrate=\n",
    ])
    procs.append(fake)
    monkeypatch.setattr(rf.subprocess, "Popen", lambda *a, **k: fake)
    seen: list[tuple[float, str]] = []
    rc, tail = rf._run_ffmpeg_progress(
        ["ffmpeg"], 20.0, lambda f, m: seen.append((f, m)), 60)
    assert rc == 0
    assert [round(f, 3) for f, _ in seen] == [0.25, 0.5]
    assert "time=00:00:10.00" in tail


def test_run_ffmpeg_progress_clamps_past_end(monkeypatch):
    fake = _FakeProc(["time=00:05:00.00\n"])            # way past a 10s total
    monkeypatch.setattr(rf.subprocess, "Popen", lambda *a, **k: fake)
    seen: list[float] = []
    rf._run_ffmpeg_progress(["ffmpeg"], 10.0,
                            lambda f, m: seen.append(f), 60)
    assert seen == [1.0]


# ------------------------------------------------- render(): cb None = old path
def _fake_run_ok(monkeypatch, calls):
    class R:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        return R()
    monkeypatch.setattr(rf.subprocess, "run", fake_run)


def test_render_without_callback_uses_subprocess_run(monkeypatch):
    calls: list = []
    _fake_run_ok(monkeypatch, calls)
    # If the progress path were touched at all, this explodes:
    monkeypatch.setattr(rf, "_run_ffmpeg_progress",
                        lambda *a, **k: pytest.fail("progress path used"))
    rf.render(_tl(), Path("out.mp4"))
    assert len(calls) == 1


def test_render_failure_without_callback_raises(monkeypatch):
    class R:
        returncode = 1
        stderr = "boom\n"

    monkeypatch.setattr(rf.subprocess, "run", lambda cmd, **kw: R())
    with pytest.raises(rf.RenderError):
        rf.render(_tl(), Path("out.mp4"))


def test_render_with_callback_streams_progress(monkeypatch):
    monkeypatch.setattr(rf.subprocess, "run",
                        lambda *a, **k: pytest.fail("blocking path used"))
    seen: list = []

    def fake_stream(cmd, total_s, cb, timeout):
        seen.append(total_s)
        cb(0.5, "")
        return 0, ""
    monkeypatch.setattr(rf, "_run_ffmpeg_progress", fake_stream)
    rf.render(_tl(n=2, dur=3.0), Path("out.mp4"),
              progress_cb=lambda f, m: None)
    assert seen == [pytest.approx(6.0)]                # timeline duration


def test_render_with_callback_failure_raises(monkeypatch):
    monkeypatch.setattr(rf, "_run_ffmpeg_progress",
                        lambda *a, **k: (1, "kaboom"))
    with pytest.raises(rf.RenderError):
        rf.render(_tl(), Path("out.mp4"), progress_cb=lambda f, m: None)


# ---------------------------------------------- render_chunked(): both branches
def _touch_out(cmd):
    """Fake ffmpeg side effect: create the output file. The chunked cmd can
    carry `-threads X` AFTER the output path, so scan instead of [-1]."""
    for a in cmd:
        if a.endswith((".ts", ".mp4")):
            Path(a).touch()


def test_render_chunked_without_callback_uses_subprocess_run(tmp_path, monkeypatch):
    calls: list = []

    class R:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        _touch_out(cmd)                                # segment must exist
        return R()
    monkeypatch.setattr(rf.subprocess, "run", fake_run)
    monkeypatch.setattr(rf, "_run_ffmpeg_progress",
                        lambda *a, **k: pytest.fail("progress path used"))
    rf.render_chunked(_tl(n=2, dur=3.0), tmp_path / "out.mp4", chunk_size=1)
    assert len(calls) == 3                             # 2 segments + concat


def test_render_chunked_callback_is_cumulative_across_segments(tmp_path,
                                                               monkeypatch):
    tl = _tl(n=2, dur=4.0)                             # 2 segments of 4s each
    monkeypatch.setattr(rf.subprocess, "run",
                        lambda cmd, **kw: (_touch_out(cmd), _Rc0())[1])

    def fake_stream(cmd, part_s, cb, timeout):
        cb(0.5, "")                                    # half of this segment
        cb(1.0, "")
        _touch_out(cmd)
        return 0, ""
    monkeypatch.setattr(rf, "_run_ffmpeg_progress", fake_stream)
    seen: list[tuple[float, str]] = []
    rf.render_chunked(tl, tmp_path / "out.mp4", chunk_size=1,
                      progress_cb=lambda f, m: seen.append((round(f, 3), m)))
    msgs = [m for _f, m in seen]
    assert "segment 1/2" in msgs and "segment 2/2" in msgs
    # segment 1 half done = 2/8 = 0.25; segment 2 fully done = 1.0
    assert seen[0][0] == pytest.approx(0.25)
    assert any(f == pytest.approx(0.5) for f, _m in seen)   # seg1 complete
    assert seen[-1] == (1.0, "concatenating")
    # strictly non-decreasing overall
    fracs = [f for f, _m in seen]
    assert fracs == sorted(fracs)


class _Rc0:
    returncode = 0
    stderr = ""


# --------------------------------------------------------- RenderWorker drain
class _TextProc:
    def __init__(self, lines):
        self._lines = list(lines)

    @property
    def stderr(self):
        return self

    def readline(self):
        return self._lines.pop(0) if self._lines else b""


def _worker(tmp_path, **kw):
    from webapp.render_worker import RenderWorker
    return RenderWorker("job1", lambda tmp: [], tmp_path / "out.mp4", **kw)


def test_worker_drain_streams_throttled_progress(tmp_path):
    seen: list[float] = []
    w = _worker(tmp_path, progress_cb=seen.append, total_seconds=10.0)
    w.proc = _TextProc([
        b"frame=  150 time=00:00:02.00\n",    # 20% -> first report
        b"frame=  155 time=00:00:02.10\n",    # +1% -> throttled away
        b"no progress line\n",
        b"frame=  600 time=00:00:08.00\n",    # 80% -> reported
    ])
    tail: list[bytes] = []
    w._drain_stderr(tail)
    assert seen and seen[0] == pytest.approx(0.2) and seen[-1] == pytest.approx(0.8)
    assert len(seen) == 2
    assert len(tail) == 4                    # tail capture untouched


def test_worker_drain_without_callback_is_tail_only(tmp_path):
    w = _worker(tmp_path)                    # no progress_cb, no total_seconds
    w.proc = _TextProc([b"time=00:00:01.00\n"])
    tail: list[bytes] = []
    w._drain_stderr(tail)                    # must not raise
    assert tail == [b"time=00:00:01.00\n"]


def test_worker_drain_survives_broken_callback(tmp_path):
    def boom(_frac):
        raise RuntimeError("job store on fire")
    w = _worker(tmp_path, progress_cb=boom, total_seconds=4.0)
    w.proc = _TextProc([b"time=00:00:02.00\n", b"time=00:00:04.00\n"])
    tail: list[bytes] = []
    w._drain_stderr(tail)
    assert len(tail) == 2                    # drain completed despite cb errors
