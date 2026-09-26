# tests/test_website_downloader.py
"""Offline tests for the website image downloader (no network).

Parses reader HTML fixtures for the LeviScanner / Madara / generic themes,
checks the junk-vs-page filtering, magic-byte verification, resumable
download and manifest, and the series chapter-range selection. All HTTP is
faked -- nothing here touches the network.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import website_downloader as wd

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 2000       # > MIN_IMAGE_BYTES, valid magic
WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 2000


# ------------------------------------------------------------------- profiles
@pytest.mark.parametrize("url,name", [
    ("https://asurascans.com/x/ch-1/", "leviscanner"),
    ("https://vortexscans.org/x/ch-1/", "leviscanner"),
    ("https://drakescans.com/x/ch-1/", "leviscanner"),
    ("https://www.manganelo.com/chap/x", "madara"),
    ("https://totally-unknown-manga.io/read/1", "generic"),
])
def test_profile_for_url(url, name):
    assert wd.profile_for_url(url).name == name


# ---------------------------------------------------------------- junk filter
@pytest.mark.parametrize("url,junk", [
    ("https://x/wp-content/uploads/2024/01/p1.webp", False),  # "ads" in up-ADS
    ("https://x/wp-content/load/page.jpg", False),            # "ad" in loAD
    ("https://x/shadow/adventure/ch1/p1.jpg", False),         # "ad" in ADventure
    ("https://x/theme/logo.png", True),
    ("https://x/wp-content/ads/banner.jpg", True),
    ("https://x/img/avatar/96/gravatar.png", True),
    ("https://x/favicon.ico", True),
    ("https://x/wp-content/noimg.png", True),
])
def test_is_junk_whole_segment(url, junk):
    assert wd._is_junk(url) is junk


# ------------------------------------------------------------- extraction themes
def test_extract_leviscanner_container_and_lazy_src():
    html = (
        '<body><img src="/logo.png"><div id="readerarea">'
        '<img class="img-responsive" src="/loading.gif" '
        'data-src="/wp-content/uploads/2024/01/p1.webp">'
        '<img data-src="/wp-content/uploads/2024/01/p2.webp">'
        '</div><img class="avatar" src="/gravatar/u/96/a.jpg"></body>')
    got = wd.extract_image_urls(html, "https://asurascans.com/series/ch-1/")
    assert got == [
        "https://asurascans.com/wp-content/uploads/2024/01/p1.webp",
        "https://asurascans.com/wp-content/uploads/2024/01/p2.webp"]


def test_extract_madara_preserves_reading_order():
    html = (
        '<div class="reading-content">'
        '<div class="page-break"><img class="wp-manga-chapter-img" '
        'data-src="https://cdn.m/p3.jpg"></div>'
        '<div class="page-break"><img class="wp-manga-chapter-img" '
        'src="/uploads/p1.jpg"></div>'
        '<div class="page-break"><img class="wp-manga-chapter-img" '
        'data-src="https://cdn.m/p2.jpg"></div></div>')
    got = wd.extract_image_urls(html, "https://readertranslators.com/m/ch/")
    # document order, not numeric sort
    assert got == ["https://cdn.m/p3.jpg",
                   "https://readertranslators.com/uploads/p1.jpg",
                   "https://cdn.m/p2.jpg"]


def test_extract_generic_filters_chrome_and_keeps_pages():
    html = ('<div><img src="/uploads/a/page-001.jpg">'
            '<img src="/uploads/a/page-002.jpg">'
            '<img src="/theme/logo.png"><img src="/ads/b.jpg">'
            '<img src="/flags/en.png"></div>')
    got = wd.extract_image_urls(html, "https://unknown.io/read/1")
    assert got == ["https://unknown.io/uploads/a/page-001.jpg",
                   "https://unknown.io/uploads/a/page-002.jpg"]


def test_extract_dedupes_and_drops_placeholders():
    html = ('<div id="readerarea">'
            '<img data-src="/up/p1.png"><img data-src="/up/p1.png">'
            '<img src="data:image/gif;base64,R0lGOD"></div>')
    got = wd.extract_image_urls(html, "https://asurascans.com/s/ch/")
    assert got == ["https://asurascans.com/up/p1.png"]


def test_extract_empty_page_returns_nothing():
    assert wd.extract_image_urls("<div>no images here</div>",
                                 "https://x.io/a") == []


# ------------------------------------------------------------- magic byte check
def test_image_ext_uses_magic_then_content_type():
    assert wd._image_ext(PNG, "", "https://x/y") == "png"
    webp = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 2000
    assert wd._image_ext(webp, "", "https://x/y") == "webp"
    jpeg = b"\xff\xd8\xff" + b"\x00" * 2000
    assert wd._image_ext(jpeg, "", "https://x/y") == "jpg"
    # an HTML "rate limited" page saved as .jpg would be silent corruption
    assert wd._image_ext(b"<html>blocked</html>", "text/html", "https://x/y.jpg") is None


# ------------------------------------------------------------------- downloads
class FakeResp:
    def __init__(self, content=b"", status=200, ctype="image/png"):
        self.content, self.status_code = content, status
        self.headers = {"content-type": ctype}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, html="", image=b"BLOCKED", ctype="image/png", status=200):
        self.headers: dict[str, str] = {}
        self.html = html
        self.image, self.ctype, self.status = image, ctype, status
        self.gets: list[str] = []

    def get(self, url, timeout=None):
        self.gets.append(url)
        return FakeResp(self.image, self.status, self.ctype)


_PAGE_HTML = ('<div id="readerarea">'
              '<img data-src="/up/p1.webp"><img data-src="/up/p2.webp">'
              '<img data-src="/up/p3.webp"></div>')


def test_download_chapter_saves_numbered_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "fetch_html", lambda *a, **k: _PAGE_HTML)
    s = FakeSession(_PAGE_HTML, image=WEBP)
    res = wd.download_chapter("https://asurascans.com/s/ch-1/", tmp_path / "ch",
                              session=s, concurrency=3)
    out = tmp_path / "ch"
    names = sorted(p.name for p in out.glob("page_*"))
    assert names == ["page_001.webp", "page_002.webp", "page_003.webp"]
    assert res.page_count == 3 and not res.failed
    manifest = (out / "manifest.json").read_text("utf-8")
    assert "source_urls" in manifest and "page_001.webp" in manifest


def test_download_chapter_resume_skips_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "fetch_html", lambda *a, **k: _PAGE_HTML)
    out = tmp_path / "ch"
    out.mkdir(parents=True)
    (out / "page_001.webp").write_bytes(PNG)        # pretend already downloaded
    s = FakeSession(image=PNG)
    res = wd.download_chapter("https://asurascans.com/s/ch-1/", out, session=s)
    assert len(res.skipped) == 1 and res.page_count == 2


def test_download_chapter_collects_non_image_block(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "fetch_html", lambda *a, **k: _PAGE_HTML)
    s = FakeSession(image=b"<html>Rate limited</html>", ctype="text/html")
    res = wd.download_chapter("https://asurascans.com/s/ch-1/", tmp_path / "c",
                              session=s)
    assert res.page_count == 0 and len(res.failed) == 3
    assert not list((tmp_path / "c").glob("page_*"))


def test_download_retries_then_gives_up(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "fetch_html", lambda *a, **k: _PAGE_HTML)
    monkeypatch.setattr(wd.time, "sleep", lambda *a: None)   # no real backoff

    class Status503(FakeSession):
        def get(self, url, timeout=None):
            self.gets.append(url)
            return FakeResp(b"", status=503)

    s = Status503(_PAGE_HTML)
    res = wd.download_chapter("https://asurascans.com/s/ch-1/", tmp_path / "c",
                              session=s, concurrency=1)
    assert res.page_count == 0 and len(res.failed) == 3
    # _MAX_RETRIES attempts per page (1 page checked is enough to prove retry)
    assert len(s.gets) == 3 * wd._MAX_RETRIES


# ------------------------------------------------------------------- series
_SERIES_HTML = (
    '<ul><li><a href="/s/chapter-5/">Chapter 5</a></li>'
    '<li><a href="/s/chapter-1/">Chapter 1</a></li>'
    '<li><a href="/s/chapter-2/">Chapter 2</a></li>'
    '<li><a href="/s/chapter-10/">Chapter 10</a></li>'
    '<li><a href="/s/chapter-list/">Chapter List</a></li></ul>')


def test_parse_chapter_links_ascending_and_excludes_list_page():
    links = wd.parse_chapter_links(_SERIES_HTML, "https://asurascans.com/s/")
    nums = [c.number for c in links]
    assert nums == [1.0, 2.0, 5.0, 10.0]           # numeric, not lexical
    assert all("chapter-list" not in c.url for c in links)


@pytest.mark.parametrize("spec,expect", [
    ("all", [1.0, 2.0, 5.0, 10.0]),
    ("1-2", [1.0, 2.0]),
    ("1,5-10", [1.0, 5.0, 10.0]),
    ("last", [10.0]),
])
def test_select_chapters(spec, expect):
    links = wd.parse_chapter_links(_SERIES_HTML, "https://asurascans.com/s/")
    assert [c.number for c in wd.select_chapters(links, spec)] == expect


def test_parse_chapters_numbered_from_url_when_label_has_no_number():
    # Asura-style list: a "First Chapter" link with a /chapter/N URL and no
    # number in its label. The number must come from the path or a 1-N range
    # selection silently drops chapter 1 (it would sort as None/unnumbered).
    html = (
        '<ul>'
        '<li><a href="/s/chapter/2/">Chapter 2 The Real Deal</a></li>'
        '<li><a href="/s/chapter/1/">First Chapter</a></li>'
        '</ul>')
    links = wd.parse_chapter_links(html, "https://asurascans.com/s/")
    assert [c.number for c in links] == [1.0, 2.0]
    got = wd.select_chapters(links, "1-2")
    assert [c.number for c in got] == [1.0, 2.0]


def test_download_series_one_folder_per_chapter(tmp_path, monkeypatch):
    # series page -> chapter list; each chapter url -> reader pages
    def fake_fetch(url, session=None, timeout=None):
        return _PAGE_HTML if "chapter-" in url else _SERIES_HTML
    monkeypatch.setattr(wd, "fetch_html", fake_fetch)
    s = FakeSession(image=PNG)
    results = wd.download_series("https://asurascans.com/series/",
                                 tmp_path / "root", chapters="1-2",
                                 session=s)
    assert len(results) == 2
    dirs = sorted(p.name for p in (tmp_path / "root").iterdir() if p.is_dir())
    assert dirs == ["chapter_0001", "chapter_0002"]
    assert all(r.page_count == 3 for r in results)


# ------------------------------------------------------------------- CLI glue
def test_cli_download_chapter_wires_up(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import cli

    calls: dict = {}

    def fake_download(url, out, **kw):
        calls["url"] = url
        calls["out"] = Path(out)
        return wd.ChapterResult(url=url, out_dir=Path(out),
                                saved=[Path("page_001.webp")])
    monkeypatch.setattr(wd, "download_chapter", fake_download)
    runner = CliRunner()
    res = runner.invoke(cli.app, ["download", "chapter",
                                  "https://asurascans.com/s/ch-1/",
                                  "--out", str(tmp_path / "c")])
    assert res.exit_code == 0, res.output
    assert calls["url"] == "https://asurascans.com/s/ch-1/"
    assert "saved 1 pages" in res.output
