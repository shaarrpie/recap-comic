# website_downloader.py
"""Website image downloader for manhwa/webtoon scanlation chapters.

Grabs the full-size page images from a chapter *reader* page on scanlation
sites -- Asura Scans, Vortex Scans, Drake Scans and the many other sites that
run the **LeviScanner** or **Madara** WordPress themes -- plus a generic
fallback for unknown domains. Output is a folder of numbered page images
(``page_001.webp`` ...) that feeds straight into ``scripts/cut_pages_and_merge.py``
(the per-page ``PAGES_DIR``) or the ``guided`` CBZ/stitch path, so a download
can flow directly into the recap pipeline.

Design notes
------------
* **No new heavy deps.** Parsing uses the stdlib ``html.parser`` (bs4/lxml are
  not project dependencies); HTTP uses ``requests`` (already available).
* **A profile registry maps a domain to its reader theme.** Each theme names
  the container that holds the chapter images (LeviScanner ``#readerarea``;
  Madara ``.reading-content`` / ``img.wp-manga-chapter-img``). The extractor
  prefers images *inside* that container -- which is exactly the page art --
  and only falls back to a whole-page, thumbnail-filtered sweep when the
  container is absent (theme version drift or an unknown site).
* **Lazy-loading aware.** Reader markup hides the real URL behind
  ``data-src`` / ``data-lazy-src`` / ``data-original`` / ``srcset`` while
  ``src`` holds a spinner; the extractor tries them in that order.
* **Robust, resumable downloads.** Shared ``requests.Session`` with a browser
  User-Agent and the chapter page as ``Referer`` (CDNs often 403 without it),
  bounded thread pool, retry + backoff on 429/5xx, magic-byte + size
  verification (an HTML "rate limited" page saved as .jpg is a silent
  corruption bug we refuse), and existing files skipped unless ``--force``.

Keep the legal note in ``__main__`` help in mind: this is a personal/archival
fetcher for chapters you can already view; preserve credit and do not
redistribute scraped artwork.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

from adapters._logging import get_logger

try:  # requests is available in the env; degrade gracefully if not.
    import requests
except ImportError:  # pragma: no cover - exercised only without the dep
    requests = None  # type: ignore[assignment]

log = get_logger("website_downloader")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
DEFAULT_TIMEOUT = 30.0
DEFAULT_CONCURRENCY = 4          # weak-PC friendly; raise for fast links
MIN_IMAGE_BYTES = 1500           # below this it is a tracking pixel / stub
_MAX_RETRIES = 4
_BACKOFF_BASE = 0.6              # seconds; grows geometrically per retry

# src values that are never the real page (spinners, 1px spacers, icons)
_PLACEHOLDER_TOKENS = (
    "loading", "loader", "blank", "placeholder", "spacer", "preloader",
    "data:image/gif;base64", "data:image/svg", "1x1", "transparent")
# Site furniture (not page art). Matched as WHOLE url path segments so the
# token "ads" cannot nuke a legitimate "/wp-content/uploads/..." page -- the
# classic naive-substring scraper bug. Slash/hyphen forms live in the
# substring set instead.
_JUNK_WORDS = {
    "avatar", "avatars", "gravatar", "logo", "logos", "favicon", "icon",
    "icons", "emoji", "emojis", "sprite", "sprites", "banner", "banners",
    "ads", "ad", "advert", "adverts", "thumb", "thumbs", "thumbnail",
    "thumbnails", "cover", "covers", "badge", "badges", "carousel",
    "slider", "sliders", "widget", "widgets", "profile", "profiles",
    "pattern", "patterns"}
_JUNK_SUBSTRINGS = (
    "/flags/", "user_", "noimg", "no-image", "no_image", "default-image",
    "default_image", "watermark")
# extensions that are page art, not vector chrome
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".gif")

_MAGIC: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "jpg"),                      # JPEG
    (b"\x89PNG\r\n\x1a\n", "png"),                 # PNG
    (b"GIF87a", "gif"), (b"GIF89a", "gif"),        # GIF
    (b"BM", "bmp"),                                # BMP
]


# --------------------------------------------------------------------------- #
# Site profiles
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SiteProfile:
    name: str
    # ids/classes whose subtree holds the chapter page images.
    container_ids: frozenset[str] = frozenset()
    container_classes: frozenset[str] = frozenset()
    # class of the individual <img> tags (Madara exposes wp-manga-chapter-img).
    img_classes: frozenset[str] = frozenset()


_LEVISCANNER = SiteProfile(
    name="leviscanner",
    container_ids=frozenset({"readerarea", "reader-area"}),
    container_classes=frozenset({"reading-content", "readcontent",
                                 "main-readingarea", "reader-area"}),
    img_classes=frozenset({"img-responsive", "wp-manga-chapter-img"}))

_MADARA = SiteProfile(
    name="madara",
    container_ids=frozenset(),
    container_classes=frozenset({"reading-content", "text-right",
                                 "page-break"}),
    img_classes=frozenset({"wp-manga-chapter-img"}))

_GENERIC = SiteProfile(name="generic")

# domain substring -> profile. Asura/Vortex/Drake share LeviScanner markup.
_DOMAIN_PROFILES: dict[str, SiteProfile] = {
    "asurascans": _LEVISCANNER, "asura-scans": _LEVISCANNER,
    "asuracomic": _LEVISCANNER,
    "vortexscans": _LEVISCANNER, "vortex-scans": _LEVISCANNER,
    "drakescans": _LEVISCANNER, "drake-scans": _LEVISCANNER,
    "manhuascans": _LEVISCANNER, "leviatanscans": _LEVISCANNER,
    "readertransl": _MADARA, "manganelo": _MADARA, "mangadex": _GENERIC,
}


def profile_for_url(url: str) -> SiteProfile:
    host = (urlparse(url).netloc or "").lower()
    for needle, prof in _DOMAIN_PROFILES.items():
        if needle in host:
            return prof
    log.debug("no named profile for host %s; using generic extractor", host)
    return _GENERIC


# --------------------------------------------------------------------------- #
# HTML parsing (stdlib only)
# --------------------------------------------------------------------------- #
_VOID_TAGS = {"img", "br", "input", "meta", "link", "source", "hr",
              "area", "base", "col", "embed", "param", "track", "wbr"}
_BG_RE = re.compile(r"background(?:-image)?\s*:\s*url\(([^)]+)\)", re.I)


def _classes(attr: str | None) -> set[str]:
    return set((attr or "").lower().split())


@dataclass
class _ImgHit:
    """One candidate image occurrence, in document order."""
    order: int
    urls: list[str]            # candidate URLs, most-trusted first
    in_container: bool
    is_chapter_img_class: bool
    width_hint: int = 0        # declared width=".." attr (0 = none)


class _ReaderParser(HTMLParser):
    """Collect <img> and background-image candidates with container ancestry.

    An ancestor stack lets us flag images that live inside the reader theme's
    container -- the pages -- versus site furniture elsewhere on the document.
    Tolerant of malformed HTML (unclosed tags) by treating endtag pops as best
    effort.
    """

    def __init__(self, container_ids: frozenset[str],
                 container_classes: frozenset[str],
                 img_classes: frozenset[str]) -> None:
        super().__init__(convert_charrefs=True)
        self._cids = container_ids
        self._cclasses = container_classes
        self._iimg_classes = img_classes
        self._stack: list[bool] = []          # per open tag: is-container?
        self.hits: list[_ImgHit] = []
        self._order = 0

    # -- ancestry bookkeeping ------------------------------------------------
    def handle_starttag(self, tag, attrs_list):  # type: ignore[override]
        attrs = dict(attrs_list)
        # ancestor-inside state is the CURRENT stack top (before any push).
        inside = bool(self._stack) and self._stack[-1]
        if tag == "img":
            # void: it is inside the container purely by ancestry; a bare
            # id="readerarea" on an <img> must NOT make itself "the container".
            self._add_img(attrs, inside)
            return
        is_container = (str(attrs.get("id", "")).lower() in self._cids
                        or bool(_classes(attrs.get("class")) & self._cclasses))
        style = str(attrs.get("style", "") or "")
        if style and inside:
            m = _BG_RE.search(style)
            if m:
                self._append_candidate([_strip_url(m.group(1))],
                                       True, False, 0)
        if tag not in _VOID_TAGS:
            self._stack.append(inside or is_container)

    def handle_startendtag(self, tag, attrs_list):  # <img ... /> form
        attrs = dict(attrs_list)
        inside = bool(self._stack) and self._stack[-1]
        if tag == "img":
            self._add_img(attrs, inside)
        elif str(attrs.get("style", "") or "") and inside:
            m = _BG_RE.search(str(attrs.get("style", "")))
            if m:
                self._append_candidate([_strip_url(m.group(1))],
                                       True, False, 0)

    def handle_endtag(self, tag):
        # Pop only tags we pushed (void tags were never pushed). Keeps the
        # stack depth aligned with the real element tree even with a single
        # stray </div>.
        if tag not in _VOID_TAGS and self._stack:
            self._stack.pop()

    # -- img extraction ------------------------------------------------------
    def _add_img(self, attrs: dict[str, str | None], in_container: bool) -> None:
        # most-trusted first: lazy-load data attrs, then src, then srcset
        cands: list[str] = []
        for key in ("data-src", "data-lazy-src", "data-original",
                    "data-cfsrc", "data-url", "src"):
            v = (attrs.get(key) or "").strip()
            if v:
                cands.append(_strip_url(v))
        cands.extend(_srcset_urls(attrs.get("srcset")))
        cands.extend(_srcset_urls(attrs.get("data-srcset")))
        is_cls = bool(_classes(attrs.get("class")) & self._iimg_classes)
        width = _to_int(attrs.get("width"))
        self._append_candidate(cands, in_container, is_cls, width)

    def _append_candidate(self, urls, in_container, is_cls, width) -> None:
        clean = [u for u in urls if u and not _is_placeholder(u)]
        if not clean:
            return
        self.hits.append(_ImgHit(order=self._order, urls=clean,
                                 in_container=in_container,
                                 is_chapter_img_class=is_cls,
                                 width_hint=width))
        self._order += 1


def _strip_url(raw: str) -> str:
    return raw.strip().strip("'\"").strip()


def _srcset_urls(srcset: str | None) -> list[str]:
    """Highest-declaration-wins extraction from a srcset string."""
    out: list[str] = []
    for part in (srcset or "").split(","):
        bits = part.strip().split()
        if bits:
            out.append(_strip_url(bits[0]))
    return out


def _to_int(v) -> int:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return 0


def _is_placeholder(url: str) -> bool:
    low = url.lower()
    return any(tok in low for tok in _PLACEHOLDER_TOKENS)


def _is_junk(url: str) -> bool:
    low = url.lower()
    if any(s in low for s in _JUNK_SUBSTRINGS):
        return True
    # whole-segment match: "uploads" != "ads", "load" != "ad", "favicon" ->
    # caught by its own word, while "/ads/x.jpg" is still filtered.
    return bool(_JUNK_WORDS.intersection(re.split(r"[^a-z0-9]+", low)))


def extract_image_urls(html: str, base_url: str,
                       profile: SiteProfile | None = None) -> list[str]:
    """Ordered, de-duplicated absolute page-image URLs for a reader page."""
    prof = profile or profile_for_url(base_url)
    parser = _ReaderParser(prof.container_ids, prof.container_classes,
                           prof.img_classes)
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # noqa: BLE001 - malformed HTML must not crash
        log.warning("html parse aborted early (%s); using partial results", exc)

    hits = parser.hits
    if not hits:
        return []
    # Prefer reader-container / chapter-class images; only scan the whole page
    # (junk-filtered) when the theme container matched nothing.
    container_hits = [h for h in hits if h.in_container or h.is_chapter_img_class]
    use_container = bool(container_hits)

    ordered: list[str] = []
    seen: set[str] = set()
    for h in sorted(container_hits or hits, key=lambda x: x.order):
        # generic sweep: drop obvious site furniture and vector chrome
        if not use_container and all(_is_junk(u) for u in h.urls):
            continue
        url = _pick_url(h, use_container)
        if not url:
            continue
        absolute = urljoin(base_url, url)
        if absolute in seen:
            continue
        if not use_container and not _looks_like_page(absolute):
            continue
        seen.add(absolute)
        ordered.append(absolute)
    return ordered


def _pick_url(hit: _ImgHit, trusted_container: bool) -> str | None:
    """Choose the best URL of a hit; in the reader container we trust even
    extension-less CDN paths, in the generic sweep we require page art."""
    for u in hit.urls:
        if _is_junk(u):
            continue
        if trusted_container or _looks_like_page(u):
            return u
    return None


def _looks_like_page(url: str) -> bool:
    path = urlparse(url).path.lower()
    if path.endswith(".svg"):
        return False
    return any(path.endswith(ext) for ext in _IMAGE_EXTS)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _require_requests() -> None:
    if requests is None:  # pragma: no cover
        raise RuntimeError(
            "the 'requests' package is required for the downloader; "
            "install it with: pip install 'recap-comic[download]'")


def make_session(user_agent: str = DEFAULT_USER_AGENT) -> requests.Session:
    _require_requests()
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent,
                      "Accept-Language": "en-US,en;q=0.9",
                      "Accept": "text/html,application/xhtml+xml,"
                                "image/avif,image/webp,*/*;q=0.8"})
    return s


def fetch_html(url: str, session: requests.Session | None = None,
               timeout: float = DEFAULT_TIMEOUT) -> str:
    session = session or make_session()
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype and "xml" not in ctype and "text" not in ctype:
        log.warning("expected HTML from %s, got %s", url, ctype or "?")
    return resp.text


# --------------------------------------------------------------------------- #
# Chapter + series download
# --------------------------------------------------------------------------- #
@dataclass
class ChapterResult:
    url: str
    out_dir: Path
    saved: list[Path] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.saved)


def download_chapter(url: str, out_dir: Path, *,
                     session: requests.Session | None = None,
                     concurrency: int = DEFAULT_CONCURRENCY,
                     force: bool = False,
                     timeout: float = DEFAULT_TIMEOUT,
                     on_progress=None) -> ChapterResult:
    """Download every page image of one chapter reader page into ``out_dir``.

    ``on_progress(done, total, message)`` is called as pages land (CLI hook;
    optional). Returns a :class:`ChapterResult`; network/verification failures
    are collected (not raised) so a partial chapter still lands on disk and the
    caller sees exactly which pages failed.
    """
    session = session or make_session()
    session.headers.setdefault("Referer", url)
    html = fetch_html(url, session, timeout=timeout)
    profile = profile_for_url(url)
    image_urls = extract_image_urls(html, url, profile)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = ChapterResult(url=url, out_dir=out_dir)
    total = len(image_urls)
    if total == 0:
        log.warning("no page images found at %s (theme drift or wrong URL?)",
                    url)
        return result
    log.info("chapter %s -> %d pages via %s profile", url, total, profile.name)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency),
                            thread_name_prefix="dl") as pool:
        futs = {pool.submit(_download_one, u, out_dir, session, i, force,
                            timeout): (i, u) for i, u in enumerate(image_urls)}
        for fut in futs:
            i, src = futs[fut]
            try:
                status, path = fut.result()
            except Exception as exc:  # noqa: BLE001 - collect, never abort run
                status, path = "error", str(exc)
            done += 1
            if status == "saved" and path is not None:
                result.saved.append(path)
            elif status == "skipped":
                result.skipped.append(src)
            else:
                result.failed.append(src)
            if on_progress:
                on_progress(done, total, f"{status} {Path(src).name}")
    result.saved.sort(key=lambda p: _natural_key(p.name))
    _write_manifest(result, image_urls)
    return result


def _download_one(url: str, out_dir: Path, session, index: int, force: bool,
                  timeout: float) -> tuple[str, object]:
    stem = out_dir / f"page_{index + 1:03d}"
    if not force:
        for ext in _IMAGE_EXTS:
            if stem.with_suffix(ext).is_file():
                return "skipped", None
    last_err = "unknown"
    for attempt in range(_MAX_RETRIES):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code}"
                time.sleep(_BACKOFF_BASE * (2 ** attempt))
                continue
            resp.raise_for_status()
            data = resp.content
            ext = _image_ext(data, resp.headers.get("content-type", ""), url)
            if ext is None:
                return "error", "not an image (blocked/interstitial?)"
            if len(data) < MIN_IMAGE_BYTES:
                return "error", f"too small ({len(data)}B)"
            dest = stem.with_suffix(f".{ext}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return "saved", dest
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            time.sleep(_BACKOFF_BASE * (2 ** attempt))
    return "error", last_err


def _image_ext(data: bytes, content_type: str, url: str) -> str | None:
    """Trust magic bytes first, then content-type, then the URL extension.
    Returns None for non-images (an HTML block page saved as .jpg is a silent
    corruption bug)."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[4:12] == b"ftypavif":
        return "avif"
    for magic, ext in _MAGIC:
        if data.startswith(magic):
            return ext
    ct = content_type.lower()
    if "image" in ct:
        for cand in ("webp", "avif", "png", "jpeg", "jpg", "gif", "bmp"):
            if cand in ct:
                return "jpg" if cand == "jpeg" else cand
        return "jpg"
    if "html" in ct or "text" in ct:
        return None
    path = urlparse(url).path.lower()
    for cand in ("webp", "avif", "png", "jpeg", "jpg", "gif", "bmp"):
        if path.endswith("." + cand):
            return "jpg" if cand == "jpeg" else cand
    return None


def _write_manifest(result: ChapterResult, image_urls: list[str]) -> None:
    import json
    payload = {
        "url": result.url,
        "pages": result.page_count,
        "saved": [p.name for p in result.saved],
        "source_urls": image_urls,
        "failed": result.failed,
        "skipped": result.skipped,
    }
    (result.out_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", "utf-8")


# --------------------------------------------------------------------------- #
# Series: chapter-list parsing + range selection
# --------------------------------------------------------------------------- #
# Allow '/' as a separator too: some themes (e.g. Asura) use /chapter/N URLs
# whose label carries no number ("First Chapter"), so the number must be
# recovered from the path or range selection silently drops that chapter.
_CHAPTER_ID_RE = re.compile(r"chapter[-_/\s]?(\d+(?:[.,]\d+)?)", re.I)
_CHAP_LINK_HINTS = ("-chapter-", "/chapter", "chapter-", "/read-", "/ch/")
_EXCLUDE_LINK_HINTS = ("/chapter-list", "-chapter-list", "chapter_page")


@dataclass
class ChapterLink:
    url: str
    label: str
    number: float | None


def parse_chapter_links(html: str, base_url: str) -> list[ChapterLink]:
    """Every plausible chapter link on a series page, ascending by number.

    Deliberately loose (an <a> whose href/label smells like a chapter) so it
    survives theme drift; callers filter with a range.
    """
    links: dict[str, ChapterLink] = {}
    for m in re.finditer(r"<a\b[^>]*>(.*?)</a>", html, re.I | re.S):
        inner = re.sub(r"<[^>]+>", " ", m.group(1))
        label = " ".join(inner.split())
        href_m = re.search(r"href=[\"']([^\"']+)[\"']", m.group(0), re.I)
        if not href_m:
            continue
        href = _strip_url(href_m.group(1))
        low = href.lower()
        if any(x in low for x in _EXCLUDE_LINK_HINTS):
            continue
        if not (any(h in low for h in _CHAP_LINK_HINTS)
                or _CHAPTER_ID_RE.search(label or "")):
            continue
        absolute = urljoin(base_url, href)
        num = _chapter_number(label, absolute)
        links.setdefault(absolute, ChapterLink(absolute, label, num))
    ordered = sorted(links.values(),
                     key=lambda c: (c.number if c.number is not None else 0))
    return ordered


def _chapter_number(label: str, url: str) -> float | None:
    for text in (label, urlparse(url).path):
        m = _CHAPTER_ID_RE.search(text or "")
        if m:
            try:
                return float(m.group(1).replace(",", "."))
            except ValueError:
                pass
    return None


def select_chapters(links: list[ChapterLink], spec: str | None
                    ) -> list[ChapterLink]:
    """Filter chapter links by a range spec like ``1-10``, ``1,3,5-8`` or
    ``last`` (``None``/``all`` keeps everything)."""
    if not spec or spec.lower() in ("all", "*"):
        return links
    wanted: set[float] = set()
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token == "last":
            nums = [c.number for c in links if c.number is not None]
            if nums:
                wanted.add(max(nums))
            continue
        nums = [c.number for c in links if c.number is not None]
        hi = (max(nums) if nums else 0) + 1
        if "-" in token:
            a, _, b = token.partition("-")
            lo_v = float(a) if a else 0.0
            hi_v = float(b) if b else hi
            wanted.update(c.number for c in links
                          if c.number is not None and lo_v <= c.number <= hi_v)
        else:
            try:
                wanted.add(float(token))
            except ValueError:
                log.warning("ignoring unparsable chapter token %r", token)
    return [c for c in links if c.number in wanted]


def download_series(series_url: str, out_dir: Path, *,
                    chapters: str | None = None,
                    session: requests.Session | None = None,
                    concurrency: int = DEFAULT_CONCURRENCY,
                    force: bool = False,
                    timeout: float = DEFAULT_TIMEOUT,
                    on_progress=None) -> list[ChapterResult]:
    """Download a chapter range of a series; one sub-folder per chapter."""
    session = session or make_session()
    html = fetch_html(series_url, session, timeout=timeout)
    links = select_chapters(parse_chapter_links(html, series_url), chapters)
    if not links:
        log.warning("no chapters matched %r at %s", chapters, series_url)
        return []
    log.info("series %s: %d chapter(s) to fetch", series_url, len(links))
    results: list[ChapterResult] = []
    for c in links:
        slug = _chapter_slug(c)
        results.append(download_chapter(
            c.url, Path(out_dir) / slug, session=session,
            concurrency=concurrency, force=force, timeout=timeout,
            on_progress=(on_progress and (
                lambda d, t, m, s=slug: on_progress(d, t, f"[{s}] {m}")))))
    return results


def _chapter_slug(c: ChapterLink) -> str:
    if c.number is not None:
        num = (f"{c.number:g}")
        return "chapter_" + num.replace(".", "_").zfill(4)
    tail = [p for p in urlparse(c.url).path.split("/") if p]
    return re.sub(r"[^a-z0-9]+", "_", tail[-1] if tail else "chapter").lower()


def _natural_key(name: str) -> list:
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", name)]
