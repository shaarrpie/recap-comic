# adapters/ai_models.py
"""Central AI model configuration: Qwen primary + Mistral fallback.

All AI (vision/LLM) calls should obtain their model identifiers from here
instead of hard-coding strings throughout the codebase.

Default behavior:
    Primary:  Qwen3.5-397B-A17B
    Fallback: Mistral Medium 3.5

Provider: OpenAI-compatible endpoint (default https://api.xkiro.com/v1).
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
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PRIMARY_MODEL = os.environ.get(
    "XKIRO_PRIMARY_MODEL", "qwen/qwen3.5-397b-a17b:free")
FALLBACK_MODEL = os.environ.get(
    "XKIRO_FALLBACK_MODEL", "mistralai/mistral-medium-3.5")
# Human-readable labels (UI/logs); the provider requires the IDs above —
# verified live against https://api.xkiro.com/v1/models: the bare string
# "Qwen3.5-397B-A17B" returns 404, while "qwen/qwen3.5-397b-a17b:free"
# and "mistralai/mistral-medium-3.5" accept text, JSON-mode and image input.
PRIMARY_LABEL = "Qwen3.5-397B-A17B"
FALLBACK_LABEL = "Mistral Medium 3.5"
DEFAULT_BASE_URL = os.environ.get("XKIRO_BASE_URL", "https://api.xkiro.com/v1")
DEFAULT_TIMEOUT_S = float(os.environ.get("XKIRO_TIMEOUT_S", "120"))


def _norm(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def resolve_model_id(name: str) -> str:
    """Map human/bare names to provider-accepted IDs.

    Accepts "Qwen3.5-397B-A17B", "qwen", "Mistral Medium 3.5", "mistral"
    (case/punctuation-insensitive); anything already provider-shaped
    (contains "/") passes through unchanged.
    """
    if "/" in name:
        return name
    key = _norm(name)
    qwen_keys = {"qwen", _norm(PRIMARY_LABEL), _norm(PRIMARY_MODEL),
                 "qwen35397ba17b", "qwen35397ba17bfree"}
    mistral_keys = {"mistral", _norm(FALLBACK_LABEL), _norm(FALLBACK_MODEL)}
    if key in qwen_keys:
        return PRIMARY_MODEL
    if key in mistral_keys:
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
    """Resolve the Xkiro/OpenAI-compatible API key without logging it.

    Resolution order (manual webapp key wins over .env keys):
      1. `explicit` — caller-passed key (api_key param) always wins.
      2. webapp_output/settings.json "api_key" — manual webapp key.
      3. XKIRO_API_KEY
      4. XKIRO_API_KEYS (comma-separated pool; first entry)
      5. GEMINI_API_KEYS pool (first entry)
      6. GEMINI_API_KEY
    """
    if explicit and explicit.strip():
        return explicit.strip()
    manual = manual_key()
    if manual:
        return manual
    for var in ("XKIRO_API_KEY", "XKIRO_API_KEYS",
                "GEMINI_API_KEYS", "GEMINI_API_KEY"):
        val = os.environ.get(var, "").strip()
        if val:
            # XKIRO_API_KEYS / GEMINI_API_KEYS may be comma-separated;
            # take the first.
            return val.split(",")[0].strip()
    return None


def require_api_key(explicit: str | None = None) -> str:
    key = api_key_from_env(explicit)
    if not key:
        raise RuntimeError(
            "No AI API key found: pass api_key, save one in the webapp "
            "settings, or set XKIRO_API_KEY / XKIRO_API_KEYS / "
            "GEMINI_API_KEYS / GEMINI_API_KEY in .env "
            "(use --backend none for offline mode)")
    return key


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
    """Try primary (Qwen), then fallback (Mistral). Never both on success.

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
    same Qwen -> Mistral fallback contract. `request_fn(model, prompt,
    b64_png) -> text` injects a fake transport for tests; otherwise the
    OpenAI-compatible endpoint is used. Empty responses count as failures."""
    key = api_key or api_key_from_env()
    if not key and request_fn is None:
        raise RuntimeError(
            "XKIRO_API_KEY is not set; set it in .env or pass api_key")
    primary_model = resolve_model_id(primary_model)
    fallback_model = resolve_model_id(fallback_model)

    def _once(model_id: str) -> str:
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

    return call_ai_with_fallback(
        operation, lambda: _once(primary_model),
        lambda: _once(fallback_model),
        primary_model=primary_model, fallback_model=fallback_model)


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
    same Qwen -> Mistral fallback contract. `request_fn(model, system,
    user) -> text` injects a fake transport for tests; otherwise the
    `openai` package is used. Empty responses count as failures."""
    key = api_key or api_key_from_env()
    if not key and request_fn is None:
        raise RuntimeError(
            "XKIRO_API_KEY is not set; set it in .env or pass api_key")
    primary_model = resolve_model_id(primary_model)
    fallback_model = resolve_model_id(fallback_model)

    def _once(model_id: str) -> str:
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

    return call_ai_with_fallback(
        operation, lambda: _once(primary_model),
        lambda: _once(fallback_model),
        primary_model=primary_model, fallback_model=fallback_model)
