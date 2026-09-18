# adapters/ai_models.py
"""Central AI model configuration: Agnes primary + Agnes fallback.

All AI (vision/LLM) calls should obtain their model identifiers from here
instead of hard-coding strings throughout the codebase. Agnes AI is the
SOLE analysis/narration provider (OpenAI-compatible gateway).

Default behavior:
    Primary:  agnes-2.5-flash
    Fallback: agnes-2.0-flash

Provider: OpenAI-compatible endpoint (default https://apihub.agnes-ai.com/v1).
Credentials NEVER appear in logs; only model names / error summaries.

Architecture rule (project requirement): AI is used ONLY for semantic
tasks (panel understanding, narration, dialogue extraction, scene
description). Physical panel cropping and blank-region detection stay
deterministic (guided_cutter / blank_detector) and must never be decided
by these models.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PRIMARY_MODEL = os.environ.get(
    "AGNES_PRIMARY_MODEL", "agnes-2.5-flash")
FALLBACK_MODEL = os.environ.get(
    "AGNES_FALLBACK_MODEL", "agnes-2.0-flash")
# Human-readable labels (UI/logs); the provider requires the IDs above —
# verified live against https://apihub.agnes-ai.com/v1/models. Both models
# accept text, JSON-mode and image input (OpenAI-compatible image_url).
PRIMARY_LABEL = "Agnes 2.5 Flash"
FALLBACK_LABEL = "Agnes 2.0 Flash"
DEFAULT_BASE_URL = os.environ.get(
    "AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1")
DEFAULT_TIMEOUT_S = float(os.environ.get("AGNES_TIMEOUT_S", "120"))


def _norm(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def resolve_model_id(name: str) -> str:
    """Map human/bare names to provider-accepted IDs.

    Accepts "Agnes 2.5 Flash", "2.5", "Agnes 2.0 Flash", "2.0"
    (case/punctuation-insensitive); anything already provider-shaped
    (starts with "agnes-") passes through unchanged.
    """
    if name.startswith("agnes-"):
        return name
    key = _norm(name)
    if key in {"agnes25flash", "25flash", _norm(PRIMARY_MODEL)}:
        return PRIMARY_MODEL
    if key in {"agnes20flash", "20flash", _norm(FALLBACK_MODEL)}:
        return FALLBACK_MODEL
    return name


# Manual webapp key: settings.json written by the webapp settings layer
# (POST /api/settings). CWD-relative by default ("webapp_output/settings.json");
# tests and the webapp can repoint OUTPUT_DIR. Monkeypatchable module
# attribute so the webapp's isolated OUTPUT_DIR applies here too.
OUTPUT_DIR = Path(os.environ.get("RECAP_OUTPUT_DIR", "webapp_output"))


def manual_key() -> str | None:
    """Manual API key from webapp_output/settings.json ("api_key" field).

    The webapp settings layer (POST /api/settings) writes this file; the
    key saved there is the user's explicit manual choice and must win over
    any .env / environment key. Read defensively: missing file, JSON
    errors, or a non-dict / missing / blank field all mean "no manual key"
    (never a crash). Never log the value.
    """
    try:
        raw = (OUTPUT_DIR / "settings.json").read_text("utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    val = data.get("api_key")
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def api_key_from_env(explicit: str | None = None) -> str | None:
    """Resolve the Agnes API key without logging it.

    Resolution order (manual webapp key wins over .env keys):
      1. `explicit` — caller-passed key (api_key param) always wins.
      2. webapp_output/settings.json "api_key" — manual webapp key.
      3. AGNES_API_KEY
      4. AGNES_API_KEYS (comma-separated pool; first entry)
    """
    if explicit and explicit.strip():
        return explicit.strip()
    manual = manual_key()
    if manual:
        return manual
    for var in ("AGNES_API_KEY", "AGNES_API_KEYS"):
        val = os.environ.get(var, "").strip()
        if val:
            # AGNES_API_KEYS may be comma-separated; take the first.
            return val.split(",")[0].strip()
    return None


def require_api_key(explicit: str | None = None) -> str:
    key = api_key_from_env(explicit)
    if not key:
        raise RuntimeError(
            "No AI API key found: pass api_key, save one in the webapp "
            "settings, or set AGNES_API_KEY in .env "
            "(use --backend none for offline mode)")
    return key


def api_key_pool(explicit: str | None = None) -> list[str]:
    """All usable Agnes keys in priority order (deduplicated).

    `[explicit]` wins alone when passed; otherwise the manual settings key,
    then AGNES_API_KEY, then every entry of AGNES_API_KEYS. Callers iterate
    the pool on rate-limit errors (see `is_rate_limit_error`) so a
    per-key problem (revoked key, per-key throttle) degrades to the next
    instead of failing the operation.

    NOTE (per Agnes TOKEN_PLAN_FAQ): limits are shared BY KEY TYPE, not per
    key — N free keys share ONE 20-RPM pool and do NOT multiply throughput.
    Rotation helps with per-key failures, not with pool exhaustion. Only
    keys of different types (free vs Token Plan) have separate pools.
    """
    if explicit and explicit.strip():
        return [explicit.strip()]
    keys: list[str] = []
    manual = manual_key()
    if manual:
        keys.append(manual)
    for var in ("AGNES_API_KEY", "AGNES_API_KEYS"):
        val = os.environ.get(var, "").strip()
        for part in val.split(","):
            part = part.strip()
            if part and part not in keys:
                keys.append(part)
    return keys


_pool_cursor = 0
_pool_lock = threading.Lock()


def pool_start_index(pool_size: int) -> int:
    """Round-robin start offset so concurrent operations spread load across
    keys instead of all hammering key #1 (which would hit its RPM first).

    Thread-safe; never logs or returns key material (just an index).
    """
    global _pool_cursor
    with _pool_lock:
        idx = _pool_cursor % max(1, pool_size)
        _pool_cursor += 1
    return idx


def _rate_wait_config() -> tuple[float, int]:
    """(wait_seconds, wait_rounds) for pool-exhausted rate limits.

    Agnes free tier has no published daily hard cap: throttling is RPM +
    fair-use, and the documented recovery is "pause and wait a few minutes".
    So when EVERY pool key is rate-limited at once, we sleep with linear
    backoff and retry the whole pool instead of failing the operation.
    Tunable via AGNES_RATE_WAIT_S / AGNES_RATE_WAIT_ROUNDS (rounds=0
    restores fail-immediately).
    """
    try:
        wait_s = float(os.environ.get("AGNES_RATE_WAIT_S", "120"))
    except (TypeError, ValueError):
        wait_s = 120.0
    try:
        rounds = int(os.environ.get("AGNES_RATE_WAIT_ROUNDS", "2"))
    except (TypeError, ValueError):
        rounds = 2
    return max(0.0, wait_s), max(0, rounds)


def call_with_key_rotation(operation: str, fn, *,
                           keys: list[str] | None = None,
                           api_key: str | None = None):
    """Run `fn(key)` (primary->fallback pair) across the key pool.

    - Start key round-robins per call (load spread, not just failover).
    - A rate-limited key rotates to the next one immediately.
    - All keys rate-limited at once -> sleep (linear backoff, see
      `_rate_wait_config`) and retry the pool, instead of failing.
    - Non-rate-limit failures raise immediately (no pointless waits).
    - Key values never appear in logs (indexes only).
    Returns fn's result; raises the last AIFallbackError (or the
    non-rate-limit error) when everything is exhausted.
    """
    if keys is None:
        keys = api_key_pool(api_key)
    if not keys:
        raise RuntimeError(
            "AGNES_API_KEY is not set; set it in .env or pass api_key")
    wait_s, rounds = _rate_wait_config()
    start = pool_start_index(len(keys))
    last_err: AIFallbackError | None = None
    for rnd in range(rounds + 1):
        for attempt in range(len(keys)):
            key = keys[(start + attempt) % len(keys)]
            try:
                return fn(key)
            except AIFallbackError as exc:
                last_err = exc
                if is_rate_limit_error(exc) and attempt + 1 < len(keys):
                    log.warning("agnes rate-limit on key %d/%d; rotating (%s)",
                                attempt + 1, len(keys), operation)
                    continue
                if not is_rate_limit_error(exc):
                    raise
                break  # whole pool throttled this round
        # Pool exhausted with rate limits: wait, then retry the pool —
        # unless this was the last round.
        assert last_err is not None
        if rnd >= rounds:
            break
        delay = wait_s * (rnd + 1)
        log.warning("agnes pool throttled (all %d key(s)); waiting %.0fs "
                    "(round %d/%d, %s)", len(keys), delay, rnd + 1, rounds,
                    operation)
        time.sleep(delay)
    assert last_err is not None
    raise last_err


def is_rate_limit_error(exc: BaseException | None) -> bool:
    """True when `exc` (or an AIFallbackError's causes) is a rate-limit /
    quota exhaustion: 429, rate_limit, resource_exhausted. These are the
    errors worth rotating to the next pool key for (never fabricated
    output, just a different credential)."""
    if exc is None:
        return False
    if isinstance(exc, AIFallbackError):
        return (is_rate_limit_error(exc.primary_error)
                or is_rate_limit_error(exc.fallback_error))
    msg = str(exc).lower()
    return ("429" in msg or "rate_limit" in msg
            or "resource_exhausted" in msg or "quota" in msg
            or "too many requests" in msg)


_TRANSIENT_MARKERS = (
    "connection reset", "connection aborted", "timed out", "timeout",
    "temporarily", "try again", "overloaded", "unavailable",
)


def is_transient(exc: Exception) -> bool:
    """True for errors worth ONE short retry: timeouts, conn resets, HTTP 5xx."""
    msg = str(exc).lower()
    if isinstance(exc, TimeoutError):
        return True
    if "timeout" in msg and "timed out" not in msg:
        # covers httpx/openai TimeoutError string forms
        return True
    if any(m in msg for m in _TRANSIENT_MARKERS):
        return True
    for code in ("500", "502", "503", "504"):
        if code in msg:
            return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    try:
        if status is not None and 500 <= int(status) <= 599:
            return True
    except (TypeError, ValueError):
        pass
    return False


class AIFallbackError(RuntimeError):
    """Both primary and fallback models failed. Carries both causes."""

    def __init__(self, operation: str, primary_error: Exception,
                 fallback_error: Exception) -> None:
        self.operation = operation
        self.primary_error = primary_error
        self.fallback_error = fallback_error
        super().__init__(
            f"AI operation {operation!r} failed: primary "
            f"({PRIMARY_MODEL}) failed: {primary_error}; fallback "
            f"({FALLBACK_MODEL}) failed: {fallback_error}")


@dataclass
class AIFallbackResult:
    """Outcome of a primary->fallback AI call."""
    result: Any
    model_used: str
    fallback_used: bool
    primary_error: Exception | None = None
    fallback_error: Exception | None = None
    attempts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "result": self.result,
            "model_used": self.model_used,
            "fallback_used": self.fallback_used,
        }


def _log_safe_error(model: str, exc: Exception) -> str:
    # Never include credentials; truncate long provider payloads.
    from adapters._logging import sanitize
    safe = sanitize(str(exc))
    return f"{type(exc).__name__}: {safe[:300]}"


def _attempt_with_policy(label: str, fn: Callable[[], Any]) -> Any:
    """Call once; on transient errors do ONE short retry, else raise."""
    try:
        return fn()
    except Exception as exc:
        if is_transient(exc):
            log.info("[AI] %s transient error, one short retry: %s",
                     label, _log_safe_error(label, exc))
            time.sleep(1)
            return fn()  # second failure propagates to caller
        raise


def call_ai_with_fallback(
    operation: str,
    primary_fn: Callable[[], Any],
    fallback_fn: Callable[[], Any],
    *,
    primary_model: str = PRIMARY_MODEL,
    fallback_model: str = FALLBACK_MODEL,
) -> AIFallbackResult:
    """Try primary (Agnes 2.5 Flash), then fallback (2.0 Flash). Never both on success.

    Any exception from primary_fn — network/timeout/HTTP/malformed/invalid
    JSON/empty/unavailable/image-rejection/token-limit — triggers the
    fallback. If both fail, raises AIFallbackError (no fabricated results).
    """
    log.info("[AI] Using primary model: %s (operation=%s)",
             primary_model, operation)
    primary_error: Exception | None = None
    try:
        result = _attempt_with_policy(primary_model, primary_fn)
    except Exception as exc:
        primary_error = exc
        log.warning("[AI] Primary model failed: %s", _log_safe_error(primary_model, exc))
    else:
        return AIFallbackResult(result=result, model_used=primary_model,
                                fallback_used=False,
                                attempts={primary_model: 1})
    log.info("[AI] Switching to fallback: %s (operation=%s)",
             fallback_model, operation)
    try:
        result = _attempt_with_policy(fallback_model, fallback_fn)
    except Exception as exc:
        log.warning("[AI] Fallback model failed: %s", _log_safe_error(fallback_model, exc))
        log.error("[AI] AI operation failed after all configured models "
                  "were exhausted (operation=%s)", operation)
        raise AIFallbackError(operation, primary_error, exc) from exc
    log.info("[AI] Fallback succeeded (operation=%s model=%s)",
             operation, fallback_model)
    return AIFallbackResult(result=result, model_used=fallback_model,
                            fallback_used=True, primary_error=primary_error,
                            attempts={primary_model: 1, fallback_model: 1})


async def acall_ai_with_fallback(
    operation: str,
    primary_fn: Callable[[], Awaitable[Any]],
    fallback_fn: Callable[[], Awaitable[Any]],
    *,
    primary_model: str = PRIMARY_MODEL,
    fallback_model: str = FALLBACK_MODEL,
) -> AIFallbackResult:
    """Async variant of call_ai_with_fallback (same contract)."""
    log.info("[AI] Using primary model: %s (operation=%s)",
             primary_model, operation)
    primary_error: Exception | None = None
    try:
        try:
            result = await primary_fn()
        except Exception as exc:
            if is_transient(exc):
                log.info("[AI] %s transient error, one short retry", primary_model)
                await asyncio.sleep(1)
                result = await primary_fn()
            else:
                raise
    except Exception as exc:
        primary_error = exc
        log.warning("[AI] Primary model failed: %s", _log_safe_error(primary_model, exc))
    else:
        return AIFallbackResult(result=result, model_used=primary_model,
                                fallback_used=False)
    log.info("[AI] Switching to fallback: %s (operation=%s)",
             fallback_model, operation)
    try:
        try:
            result = await fallback_fn()
        except Exception as exc:
            if is_transient(exc):
                await asyncio.sleep(1)
                result = await fallback_fn()
            else:
                raise
    except Exception as exc:
        log.warning("[AI] Fallback model failed: %s", _log_safe_error(fallback_model, exc))
        log.error("[AI] AI operation failed after all configured models "
                  "were exhausted (operation=%s)", operation)
        raise AIFallbackError(operation, primary_error, exc) from exc
    log.info("[AI] Fallback succeeded (operation=%s model=%s)",
             operation, fallback_model)
    return AIFallbackResult(result=result, model_used=fallback_model,
                            fallback_used=True, primary_error=primary_error)


def cache_key_parts(*, prompt: str, image_sha: str = "",
                    settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parts that MUST be folded into a cache key to distinguish model /
    prompt / image / settings (callers hash this alongside existing keys)."""
    import hashlib
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    h.update(b"\x00" + image_sha.encode("utf-8"))
    return {
        "primary_model": PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL,
        "prompt_sha256": h.hexdigest()[:32],
        "image_sha256": image_sha[:32],
        "settings": settings or {},
    }


def generate_vision_with_fallback(
    prompt: str, b64_png: str, *,
    operation: str = "vision-generation",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    primary_model: str = PRIMARY_MODEL,
    fallback_model: str = FALLBACK_MODEL,
    request_fn: Callable[..., str] | None = None,
) -> AIFallbackResult:
    """Vision generation (ACTUAL image bytes, never text-only) with the
    same Agnes primary -> fallback contract. `request_fn(model, prompt,
    b64_png) -> text` injects a fake transport for tests; otherwise the
    OpenAI-compatible endpoint is used. Empty responses count as failures.

    On rate-limit errors the whole primary->fallback pair is retried with
    the next key from the pool (round-robin start), so one exhausted key
    degrades instead of failing the call.
    """
    keys = api_key_pool(api_key)
    if not keys and request_fn is None:
        raise RuntimeError(
            "AGNES_API_KEY is not set; set it in .env or pass api_key")
    if not keys:
        keys = [""]
    primary_model = resolve_model_id(primary_model)
    fallback_model = resolve_model_id(fallback_model)

    def _once(model_id: str, key: str) -> str:
        if request_fn is not None:
            text = request_fn(model_id, prompt, b64_png)
        else:
            import openai
            client = openai.OpenAI(
                api_key=key, base_url=(base_url or DEFAULT_BASE_URL),
                timeout=timeout or DEFAULT_TIMEOUT_S)
            resp = client.chat.completions.create(
                model=model_id, max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text":
                         "Describe ONLY what is visible in this panel image."},
                        {"type": "image_url", "image_url": {
                            "url": "data:image/png;base64," + b64_png}}]}])
            text = resp.choices[0].message.content or ""
        if not text.strip():
            raise ValueError(f"model {model_id} returned an empty response")
        return text

    def _pair(key: str):
        return call_ai_with_fallback(
            operation, lambda: _once(primary_model, key),
            lambda: _once(fallback_model, key),
            primary_model=primary_model, fallback_model=fallback_model)

    outcome = call_with_key_rotation(operation, _pair, keys=keys)
    return outcome


def generate_text_with_fallback(
    system: str, user: str, *,
    operation: str = "text-generation",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    primary_model: str = PRIMARY_MODEL,
    fallback_model: str = FALLBACK_MODEL,
    request_fn: Callable[..., str] | None = None,
) -> AIFallbackResult:
    """Text-only generation via the OpenAI-compatible endpoint with the
    same Agnes primary -> fallback contract. `request_fn(model, system,
    user) -> text` injects a fake transport for tests; otherwise the
    `openai` package is used. Empty responses count as failures.

    On rate-limit errors the whole primary->fallback pair is retried with
    the next key from the pool (round-robin start), so one exhausted key
    degrades instead of failing the call.
    """
    keys = api_key_pool(api_key)
    if not keys and request_fn is None:
        raise RuntimeError(
            "AGNES_API_KEY is not set; set it in .env or pass api_key")
    if not keys:
        keys = [""]
    primary_model = resolve_model_id(primary_model)
    fallback_model = resolve_model_id(fallback_model)

    def _once(model_id: str, key: str) -> str:
        model_id = resolve_model_id(model_id)
        if request_fn is not None:
            text = request_fn(model_id, system, user)
        else:
            import openai
            client = openai.OpenAI(
                api_key=key, base_url=(base_url or DEFAULT_BASE_URL),
                timeout=timeout or DEFAULT_TIMEOUT_S)
            resp = client.chat.completions.create(
                model=model_id, max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            text = resp.choices[0].message.content or ""
        if not text.strip():
            raise ValueError(f"model {model_id} returned an empty response")
        return text

    def _pair(key: str):
        return call_ai_with_fallback(
            operation, lambda: _once(primary_model, key),
            lambda: _once(fallback_model, key),
            primary_model=primary_model, fallback_model=fallback_model)

    outcome = call_with_key_rotation(operation, _pair, keys=keys)
    return outcome
