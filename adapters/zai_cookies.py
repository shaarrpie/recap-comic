# adapters/zai_cookies.py
"""Cookie-based authentication for chat.z.ai (Z AI).

The official Z AI API at api.z.ai/api/paas/v4 requires an API key.  But the
web chat at chat.z.ai authenticates with a JWT ``token`` cookie, and that
same token works as a Bearer header against the ``chat.z.ai`` API endpoints
(``/api/v1/auths/``, ``/api/v1/chats/new``, ``/api/chat/completions``).

This module provides:

  parse_cookie_string : turn a ``"k=v; k2=v2"`` header into a dict
  extract_bearer_token  : pull the JWT ``token`` out of a cookie dict
  save_cookies / load_cookies : persist the cookie jar to disk
  fetch_guest_token    : obtain a fresh anonymous token (no login required,
                          but rate-limited and lower priority)
  ZaiAuth              : unifying auth facade used by backends / adapters.

Auth precedence (first wins)::

  1. cookies passed explicitly to ZaiAuth(cookies=...) / --zai-cookies
  2. ZAI_TOKEN env var  (raw JWT — used directly as Bearer token)
  3. ZAI_COOKIES env var (full cookie string — parsed, token extracted)
  4. auto_fetch=True    → anonymous token from GET /api/v1/auths/
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

log = None  # set lazily so importing this module never needs logging configured


DEFAULT_BASE_URL = "https://chat.z.ai"
AUTH_ENDPOINT = "/api/v1/auths/"
GUEST_TOKEN_TTL = 300  # seconds — refresh proactively before expiry

# Cookie jar location: ~/.cache/recap-comic/zai_cookies.json
_COOKIE_DIR = Path.home() / ".cache" / "recap-comic"


def _logger():
    global log
    if log is None:
        import logging
        log = logging.getLogger("adapters.zai_cookies")
    return log


def default_cookie_path() -> Path:
    return _COOKIE_DIR / "zai_cookies.json"


def parse_cookie_string(cookie_str: str) -> dict[str, str]:
    """Parse a ``Cookie:`` header value into a name->value dict.

    Handles both ``"; "`` and plain ``;`` separators, ignores empty
    segments and strips surrounding whitespace / quotes.
    """
    cookies: dict[str, str] = {}
    for part in cookie_str.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, _, value = part.partition("=")
        name = name.strip()
        if not name:
            continue
        cookies[name] = value.strip().strip('"')
    return cookies


def extract_bearer_token(cookies: dict[str, str] | str) -> str | None:
    """Return the JWT ``token`` cookie value, or None if absent."""
    if isinstance(cookies, str):
        cookies = parse_cookie_string(cookies)
    return cookies.get("token")


def cookie_header(cookies: dict[str, str]) -> str:
    """Render a cookie dict as a ``Cookie:`` header value."""
    return "; ".join(f"{k}={v}" for k, v in cookies.items())


def save_cookies(cookies: dict[str, str],
                 path: Path | None = None) -> Path:
    """Persist *cookies* as JSON to *path* (default ~/.cache/recap-comic/zai_cookies.json)."""
    path = Path(path) if path is not None else default_cookie_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cookies, indent=2), encoding="utf-8")
    _logger().info("saved %d Z AI cookies to %s", len(cookies), path)
    return path


def load_cookies(path: Path | None = None) -> dict[str, str]:
    """Load cookies from *path* (default cookie jar).  Returns {} if missing."""
    path = Path(path) if path is not None else default_cookie_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        _logger().warning("cookie file %s is corrupt; ignoring", path)
        return {}
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    return {}


def fetch_guest_token(base_url: str = DEFAULT_BASE_URL,
                      timeout: int = 30) -> dict[str, str]:
    """Call ``GET /api/v1/auths/`` on chat.z.ai to obtain a fresh anonymous
    token cookie.

    Returns a cookie dict containing at least ``token`` (and sometimes
    ``ssxmod_itna`` / ``ssxmod_itna2``).  Raises RuntimeError on failure.
    """
    url = base_url.rstrip("/") + AUTH_ENDPOINT
    req = urllib.request.Request(
        url, method="GET",
        headers={
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "referer": "https://chat.z.ai/",
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/139.0.0.0 Safari/537.36"
            ),
        })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            # The auths endpoint returns JSON: {"token": "...", "name": "guest", ...}
            data = json.loads(body)
            cookies: dict[str, str] = {}
            token = data.get("token") or data.get("token_info", {}).get("token")
            if token:
                cookies["token"] = token
            # Also capture any Set-Cookie headers
            for raw in resp.headers.get_all("Set-Cookie") or []:
                cookies.update(parse_cookie_string(raw))
            if not cookies.get("token"):
                raise RuntimeError(
                    f"GET {AUTH_ENDPOINT} did not return a token; "
                    f"response: {body[:300]}")
            _logger().info("fetched guest token from %s", url)
            return cookies
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise RuntimeError(
                "Z AI auth endpoint rate-limited (429); wait a moment or "
                "provide your own cookies via --zai-cookies") from exc
        raise RuntimeError(
            f"Z AI auth failed: HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"cannot reach Z AI auth endpoint ({exc.reason}); "
            "check your network or provide cookies via --zai-cookies") from exc


class ZaiAuth:
    """Cookie-based authentication facade for chat.z.ai.

    Exposes:
      bearer_token       — the JWT to use in ``Authorization: Bearer …``
      auth_headers()      — dict with Authorization + standard Z AI headers
      cookie_header()     — full ``Cookie:`` header string for request fidelity
      token_source        — "cookie", "env", "guest", or "explicit"

    Usage::

        auth = ZaiAuth.from_env_or_file()   # auto-resolves credentials
        headers = auth.auth_headers()
        headers["referer"] = f"https://chat.z.ai/c/{chat_id}"
    """

    def __init__(self, cookies: dict[str, str] | str | None = None,
                 *, token: str | None = None,
                 auto_fetch: bool = True,
                 base_url: str = DEFAULT_BASE_URL,
                 cache_path: Path | None = None,
                 timeout: int = 30) -> None:
        self._cookies: dict[str, str] = {}
        self._token: str | None = None
        self._token_source: str = "unset"
        self._token_expiry: float = 0.0
        self.base_url = base_url
        self.timeout = timeout
        self.cache_path = Path(cache_path) if cache_path else default_cookie_path()
        self._auto_fetch = auto_fetch

        if cookies:
            self._set_cookies(cookies, source="cookie")
        elif token:
            self._set_token(token, source="explicit")
        else:
            self._resolve_from_env()
            if not self._token and auto_fetch:
                self._refresh_guest_token()

    # -- construction helpers -------------------------------------------------

    @classmethod
    def from_env_or_file(cls, *, auto_fetch: bool = True,
                         cache_path: Path | None = None,
                          base_url: str = DEFAULT_BASE_URL) -> ZaiAuth:
        """Build a ZaiAuth from env vars (ZAI_TOKEN / ZAI_COOKIES) + on-disk
        cookie cache, falling back to an auto-fetched guest token."""
        return cls(
            cookies=os.environ.get("ZAI_COOKIES"),
            token=os.environ.get("ZAI_TOKEN"),
            auto_fetch=auto_fetch,
            base_url=base_url,
            cache_path=cache_path,
        )

    # -- internal mutators ----------------------------------------------------

    def _set_cookies(self, cookies: dict[str, str] | str,
                     *, source: str) -> None:
        if isinstance(cookies, str):
            cookies = parse_cookie_string(cookies)
        self._cookies = dict(cookies)
        token = extract_bearer_token(self._cookies)
        if token:
            self._set_token(token, source=source)
        else:
            _logger().warning("cookies provided but no 'token' cookie found")

    def _set_token(self, token: str, *, source: str) -> None:
        self._token = token
        self._token_source = source
        self._token_expiry = time.time() + GUEST_TOKEN_TTL

    def _resolve_from_env(self) -> None:
        env_token = os.environ.get("ZAI_TOKEN")
        env_cookies = os.environ.get("ZAI_COOKIES")
        if env_cookies:
            self._set_cookies(env_cookies, source="env")
        if self._token:
            return
        if env_token:
            self._set_token(env_token, source="env")
            return
        cached = load_cookies(self.cache_path)
        if cached:
            self._set_cookies(cached, source="cache")

    def _refresh_guest_token(self) -> None:
        """Auto-fetch an anonymous guest token when no user token is present."""
        try:
            cookies = fetch_guest_token(base_url=self.base_url,
                                        timeout=self.timeout)
            self._set_cookies(cookies, source="guest")
            save_cookies(cookies, self.cache_path)
        except RuntimeError as exc:
            _logger().error("guest token auto-fetch failed: %s", exc)
            if not self._token:
                raise

    # -- public API -----------------------------------------------------------

    @property
    def bearer_token(self) -> str:
        """The current JWT token.  Raises RuntimeError if none is available."""
        if self._token is None:
            raise RuntimeError(
                "no Z AI token available; pass cookies via --zai-cookies, "
                "set ZAI_TOKEN, or enable auto_fetch")
        if time.time() > self._token_expiry and self._token_source == "guest":
            _logger().info("guest token expired; refreshing")
            self._refresh_guest_token()
        return self._token

    @property
    def token_source(self) -> str:
        return self._token_source

    @property
    def cookies(self) -> dict[str, str]:
        return dict(self._cookies)

    def cookie_header(self) -> str:
        """Full ``Cookie:`` header string (all cookies, not just token)."""
        if not self._cookies:
            return ""
        return "; ".join(f"{k}={v}" for k, v in self._cookies.items())

    def auth_headers(self, **extra: str) -> dict[str, str]:
        """Standard headers for a chat.z.ai API request.

        Includes Authorization (Bearer token), Content-Type, Accept, User-Agent,
        and Referer.  Pass ``referer=`` or other headers via keyword args to
        override / add.
        """
        headers: dict[str, str] = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "authorization": f"Bearer {self.bearer_token}",
            "content-type": "application/json",
            "referer": "https://chat.z.ai/",
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/139.0.0.0 Safari/537.36"
            ),
        }
        cookie_hdr = self.cookie_header()
        if cookie_hdr:
            headers["cookie"] = cookie_hdr
        headers.update(extra)
        return headers

    def __repr__(self) -> str:
        source = self._token_source
        masked = ""
        if self._token:
            masked = f", token={self._token[:8]}…{self._token[-4:]}"
        return f"ZaiAuth(source={source}{masked})"


# Backwards-compat: expose as ZaiCookieAuth alias
ZaiCookieAuth = ZaiAuth
