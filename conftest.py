# conftest.py (repo root)
"""Session-scoped test isolation for the webapp suite.

The webapp modules resolve OUTPUT_DIR (and the job persistence dir) at
import time, honoring RECAP_OUTPUT_DIR / RECAP_JOBS_DIR env overrides.
Setting them here — BEFORE pytest imports any webapp module — redirects
every test's sessions, panel PNGs, editor projects, checkpoints and job
snapshots into a throwaway directory, so repeated runs stop littering
the real webapp_output/ (4.5k phantom session dirs) and .cache/jobs.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent
# Ensure the repo root is importable no matter where pytest is invoked
# from (webapp/ tests rely on `import webapp`, `import guided_cutter`...).
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_TMP_BASE = Path(tempfile.mkdtemp(prefix="recap-test-"))
os.environ.setdefault("RECAP_OUTPUT_DIR", str(_TMP_BASE / "webapp_output"))
os.environ.setdefault("RECAP_JOBS_DIR", str(_TMP_BASE / "jobs"))


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    # Best-effort cleanup of the throwaway tree (Windows may hold locks
    # on files from daemon threads; leftover dirs go to %TEMP% anyway).
    import shutil
    shutil.rmtree(_TMP_BASE, ignore_errors=True)
