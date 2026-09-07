# adapters/_gemini_keys.py
"""Gemini API key rotation.

Reads keys from:
  - GEMINI_API_KEYS  (comma-separated, preferred)
  - GEMINI_API_KEY   (single-key fallback)

Rotation is process-local and thread-safe. On quota / 429 errors,
callers can advance to the next key and retry.
"""
from __future__ import annotations

import os
import threading


class KeyRotator:
    """Round-robin key rotation with a fixed key list."""

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("no Gemini API keys configured")
        self._keys = [k.strip() for k in keys if k and k.strip()]
        if not self._keys:
            raise ValueError("no Gemini API keys configured")
        self._idx = 0
        self._lock = threading.Lock()

    def current(self) -> str:
        with self._lock:
            return self._keys[self._idx % len(self._keys)]

    def advance(self) -> str:
        with self._lock:
            self._idx += 1
            return self._keys[self._idx % len(self._keys)]

    @property
    def total(self) -> int:
        return len(self._keys)


def from_env() -> KeyRotator:
    """Build a rotator from environment variables."""
    keys_str = os.environ.get("GEMINI_API_KEYS", "").strip()
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if keys:
            return KeyRotator(keys)
    single = os.environ.get("GEMINI_API_KEY", "").strip()
    if single:
        return KeyRotator([single])
    raise RuntimeError(
        "GEMINI_API_KEY or GEMINI_API_KEYS not set; "
        "use --backend none for offline mode"
    )
