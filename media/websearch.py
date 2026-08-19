"""Keyless-first web search and readable-page extraction for the research reasoners.

Two public coroutines, both of which are *total*: they never raise, they only
return empty/annotated results.  Research reasoners run dozens of these
concurrently and a single dead host must never take down a run.

``search()`` walks a provider ladder and stops at the first provider that
actually returns rows:

1. **Tavily**     — if ``TAVILY_API_KEY`` is set.
2. **Brave**      — if ``BRAVE_API_KEY`` is set.
3. **Serper**     — if ``SERPER_API_KEY`` is set.
4. **DuckDuckGo** — keyless HTML scrape of ``html.duckduckgo.com`` (with a
   POST retry and the ``lite.duckduckgo.com`` mirror as backstops).  DDG hides
   result targets behind ``//duckduckgo.com/l/?uddg=<urlencoded>``; those are
   decoded back to the real URL.  Rate limiting / captchas are swallowed.

**Wikipedia is always queried in parallel** with the ladder and merged in (up
to 3 hits, guaranteed slots at the tail of the result list).  Even when every
keyless provider is throttled into silence there is still one high-quality,
citable source per query — which is the difference between a video essay with
references and one that hallucinates them.

``fetch()`` turns a URL into clean plaintext.  Wikipedia articles go through
the MediaWiki ``extracts`` API (proper plaintext, redirects resolved); anything
else is downloaded (capped at 2.5 MB) and run through ``trafilatura`` when it
is installed, falling back to a built-in tag-stripping extractor when it is
not.  PDFs and other binary payloads report an ``error`` instead of crashing.

**HTTP/2 matters here.**  Wikimedia's anti-crawler layer answers httpx's
HTTP/1.1 handshake with ``403 Please respect our robot policy`` no matter what
User-Agent is sent, but serves HTTP/2 requests normally.  The clients below
negotiate h2 whenever the ``h2`` package is importable (``pip install
httpx[http2]``); if it is not, Wikimedia calls fall back to a stdlib
``urllib`` request on a worker thread, whose TLS handshake sometimes gets
through.  Install ``h2`` — without it Wikipedia coverage is unreliable.

Pure ``httpx``.  No AgentField imports, no global state beyond module
constants, so this file can be imported and unit-tested on its own.
"""

from __future__ import annotations

import asyncio
import html as _htmllib
import json
import os
import re
import urllib.request
from html.parser import HTMLParser
from typing import Any
from collections.abc import Awaitable, Callable, Iterable
from urllib.parse import quote, unquote, urlencode, urlsplit, urlunsplit

import httpx

# Optional: much better article extraction when present, cleanly absent when not.
try:  # pragma: no cover - depends on the deployment image
    import trafilatura
except Exception:  # pragma: no cover
    trafilatura = None  # type: ignore[assignment]

# Optional but strongly recommended: without HTTP/2 every Wikimedia request 403s.
try:  # pragma: no cover - depends on the deployment image
    import h2 as _h2  # noqa: F401

    HTTP2 = True
except Exception:  # pragma: no cover
    HTTP2 = False


# ───── Module constants ──────────────────────────────────────────────

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 video-essay-af/0.1"
)

#: Hard ceiling on a single download; a 2.5 MB HTML page is already pathological.
MAX_DOWNLOAD_BYTES = 2_500_000

#: How many Wikipedia hits are merged into every search result set.
WIKI_MAX_HITS = 3

#: Backoff before the single retry of a throttled (429/503) Wikimedia call.
WIKI_RETRY_DELAY = 1.0

_BROWSER_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_TEXTUAL_TYPES = (
    "application/xhtml+xml",
    "application/xml",
    "application/json",
    "application/rss+xml",
    "application/atom+xml",
    "application/javascript",
)

#: Wikipedia namespaces with no useful prose extract — fall through to HTML.
_WIKI_SKIP_NAMESPACES = frozenset(
    {"file", "category", "special", "help", "template", "talk", "portal", "user", "media"}
)

_TAG_RE = re.compile(r"(?s)<[^>]+>")
_COMMENT_RE = re.compile(r"(?s)<!--.*?-->")
_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)\b[^>]*>.*?</\1\s*>")
_BOILERPLATE_RE = re.compile(
    r"(?is)<(head|script|style|nav|footer|header|aside|noscript|svg|form|iframe|template)"
    r"\b[^>]*>.*?</\1\s*>"
)
_BLOCK_BREAK_RE = re.compile(
    r"(?i)<(?:br|/p|/div|/li|/tr|/h[1-6]|/section|/article|/blockquote|/pre|/td)\b[^>]*>"
)
_TITLE_RE = re.compile(r"(?is)<title[^>]*>(.*?)</title>")
_META_RE = re.compile(r"(?is)<meta\b[^>]*>")
_OG_TITLE_RE = re.compile(r"""(?i)(?:property|name)\s*=\s*["']og:title["']""")
_CONTENT_ATTR_RE = re.compile(r"""(?is)content\s*=\s*["']([^"']*)["']""")
_META_CHARSET_RE = re.compile(rb"""(?i)<meta[^>]+charset\s*=\s*["']?\s*([a-z0-9_\-]+)""")
_UDDG_RE = re.compile(r"[?&]uddg=([^&]+)")


# ───── URL helpers ───────────────────────────────────────────────────


def normalize_url(url: str) -> str:
    """Canonical form used for de-duplication: lowercase scheme/host, no
    ``www.``, no trailing slash, no fragment.  Query string is preserved
    because it frequently *is* the identity of the page."""
    raw = (url or "").strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = "https:" + raw
    elif "://" not in raw:
        raw = "https://" + raw
    try:
        parts = urlsplit(raw)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return raw.lower().rstrip("/")
    scheme = (parts.scheme or "https").lower()
    if host.startswith("www."):
        host = host[4:]
    if port and not (scheme == "http" and port == 80) and not (scheme == "https" and port == 443):
        host = f"{host}:{port}"
    path = parts.path or ""
    if len(path) > 1:
        path = path.rstrip("/")
    if path == "/":
        path = ""
    return urlunsplit((scheme, host, path, parts.query, ""))


def _wiki_url(title: str, host: str = "en.wikipedia.org") -> str:
    slug = quote((title or "").strip().replace(" ", "_"), safe="/:()',!*-._~")
    return f"https://{host}/wiki/{slug}"


def _wikipedia_article(url: str) -> tuple[str, str]:
    """``(article_title, host)`` for a Wikipedia article URL, else ``("", "")``."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "", ""
    host = (parts.hostname or "").lower()
    if host != "wikipedia.org" and not host.endswith(".wikipedia.org"):
        return "", ""
    path = parts.path or ""
    if not path.startswith("/wiki/"):
        return "", ""
    title = unquote(path[len("/wiki/") :]).replace("_", " ").strip()
    if not title:
        return "", ""
    if ":" in title and title.split(":", 1)[0].strip().lower() in _WIKI_SKIP_NAMESPACES:
        return "", ""
    return title, host


# ───── Text helpers ──────────────────────────────────────────────────


def _one_line(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _collapse_block(value: str) -> str:
    """Collapse whitespace while keeping paragraph structure."""
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_tags(value: str) -> str:
    """HTML fragment → single line of plain text (used on provider snippets)."""
    if not value:
        return ""
    text = _SCRIPT_STYLE_RE.sub(" ", value)
    text = _TAG_RE.sub(" ", text)
    return _one_line(_htmllib.unescape(text))


def _err(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _result(title: str, url: str, snippet: str, source: str) -> dict:
    return {
        "title": _one_line(str(title or "")),
        "url": str(url or "").strip(),
        "snippet": _one_line(str(snippet or "")),
        "source": source,
    }


# ───── Search providers ──────────────────────────────────────────────


async def _tavily(client: httpx.AsyncClient, query: str, limit: int, key: str) -> list[dict]:
    resp = await client.post(
        "https://api.tavily.com/search",
        json={"api_key": key, "query": query, "max_results": max(1, limit)},
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("results") or []
    out = [
        _result(row.get("title") or row.get("url") or "", row.get("url") or "",
                row.get("content") or row.get("snippet") or "", "tavily")
        for row in rows
        if isinstance(row, dict) and row.get("url")
    ]
    return out[:limit]


async def _brave(client: httpx.AsyncClient, query: str, limit: int, key: str) -> list[dict]:
    resp = await client.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": max(1, min(20, limit))},
        headers={"X-Subscription-Token": key, "Accept": "application/json"},
    )
    resp.raise_for_status()
    payload = resp.json()
    rows = ((payload.get("web") or {}).get("results")) or []
    out = [
        _result(row.get("title") or "", row.get("url") or "",
                _strip_tags(row.get("description") or ""), "brave")
        for row in rows
        if isinstance(row, dict) and row.get("url")
    ]
    return out[:limit]


async def _serper(client: httpx.AsyncClient, query: str, limit: int, key: str) -> list[dict]:
    resp = await client.post(
        "https://google.serper.dev/search",
        json={"q": query, "num": max(1, min(20, limit))},
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("organic") or []
    out = [
        _result(row.get("title") or "", row.get("link") or "",
                row.get("snippet") or "", "serper")
        for row in rows
        if isinstance(row, dict) and row.get("link")
    ]
    return out[:limit]


def _decode_ddg_href(href: str) -> str:
    """``//duckduckgo.com/l/?uddg=<enc>&rut=..`` → the real target URL."""
    raw = _htmllib.unescape((href or "").strip())
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = "https:" + raw
    hit = _UDDG_RE.search(raw)
    if hit:
        raw = unquote(hit.group(1))
    if not raw.lower().startswith(("http://", "https://")):
        return ""
    try:
        host = (urlsplit(raw).hostname or "").lower()
    except ValueError:
        return ""
    if host.endswith("duckduckgo.com"):  # ad slots and internal links
        return ""
    return raw


class _DDGResultParser(HTMLParser):
    """Pulls ``(title, url, snippet)`` triples out of the DDG HTML endpoints.

    Handles both shapes: ``html.duckduckgo.com`` (``result__a`` /
    ``result__snippet``) and ``lite.duckduckgo.com`` (``result-link`` /
    ``result-snippet``).  A snippet attaches to the most recently opened
    result, so a result with no snippet cannot shift the pairing.
    """

    _LINK_CLASSES = ("result__a", "result-link")
    _SNIPPET_CLASSES = ("result__snippet", "result-snippet")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._mode: str | None = None
        self._mode_tag = ""
        self._buf: list[str] = []
        self._href = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {(k or "").lower(): (v or "") for k, v in attrs}
        css = attributes.get("class", "")
        if not css:
            return
        if tag == "a" and any(c in css for c in self._LINK_CLASSES):
            self._close()
            self._mode, self._mode_tag, self._buf = "title", tag, []
            self._href = attributes.get("href", "")
        elif any(c in css for c in self._SNIPPET_CLASSES):
            self._close()
            self._mode, self._mode_tag, self._buf = "snippet", tag, []

    def handle_endtag(self, tag: str) -> None:
        if self._mode and tag == self._mode_tag:
            self._close()

    def handle_data(self, data: str) -> None:
        if self._mode:
            self._buf.append(data)

    def _close(self) -> None:
        if not self._mode:
            return
        text = _one_line("".join(self._buf))
        if self._mode == "title":
            target = _decode_ddg_href(self._href)
            if target and text:
                self.results.append(_result(text, target, "", "duckduckgo"))
        elif self._mode == "snippet" and self.results and not self.results[-1]["snippet"]:
            self.results[-1]["snippet"] = text
        self._mode, self._mode_tag, self._buf, self._href = None, "", [], ""


async def _duckduckgo(client: httpx.AsyncClient, query: str, limit: int, key: str = "") -> list[dict]:
    attempts: tuple[tuple[str, str], ...] = (
        ("GET", "https://html.duckduckgo.com/html/"),
        ("POST", "https://html.duckduckgo.com/html/"),
        ("GET", "https://lite.duckduckgo.com/lite/"),
    )
    headers = dict(_BROWSER_HEADERS)
    headers["Referer"] = "https://duckduckgo.com/"
    for method, url in attempts:
        try:
            if method == "GET":
                resp = await client.get(url, params={"q": query, "kl": "us-en"}, headers=headers)
            else:
                resp = await client.post(url, data={"q": query, "kl": "us-en"}, headers=headers)
            if resp.status_code != 200:  # 202/403 == rate limited, just move on
                continue
            body = resp.text or ""
            if not body or "captcha" in body[:4000].lower():
                continue
            parser = _DDGResultParser()
            parser.feed(body)
            parser.close()
            rows = [row for row in parser.results if row["url"]]
            if rows:
                return rows[:limit]
        except Exception:
            continue
    return []


def _stdlib_json(url: str, params: dict, timeout: float) -> Any | None:
    """Blocking stdlib GET, run on a worker thread.

    Only used as the last resort for Wikimedia when HTTP/2 is unavailable:
    ``urllib`` presents a different TLS handshake than httpx and is sometimes
    let through where httpx-over-h1 is rejected outright.
    """
    try:
        full = f"{url}?{urlencode(params)}" if params else url
        request = urllib.request.Request(
            full, headers={"User-Agent": UA, "Accept": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310
            if getattr(resp, "status", 200) != 200:
                return None
            raw = resp.read(MAX_DOWNLOAD_BYTES)
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return None


async def _wikimedia_json(client: httpx.AsyncClient, url: str, params: dict) -> Any:
    """GET a MediaWiki API endpoint, retrying once through a throttle."""
    detail = "no response"
    for attempt in range(2):
        try:
            resp = await client.get(
                url, params=params, headers={"User-Agent": UA, "Accept": "application/json"}
            )
            if resp.status_code == 200:
                return resp.json()
            detail = f"HTTP {resp.status_code}"
            if resp.status_code in (429, 503) and attempt == 0:
                await asyncio.sleep(WIKI_RETRY_DELAY)
                continue
        except Exception as exc:
            detail = _err(exc)
        break
    if not HTTP2:
        payload = await asyncio.to_thread(_stdlib_json, url, params, 20.0)
        if payload is not None:
            return payload
    raise RuntimeError(f"wikimedia request failed ({detail})")


async def _wikipedia_search(client: httpx.AsyncClient, query: str, limit: int = WIKI_MAX_HITS) -> list[dict]:
    payload = await _wikimedia_json(
        client,
        "https://en.wikipedia.org/w/api.php",
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max(1, limit),
            "utf8": 1,
        },
    )
    rows = ((payload.get("query") or {}).get("search")) or []
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = (row.get("title") or "").strip()
        if not title:
            continue
        out.append(_result(title, _wiki_url(title), _strip_tags(row.get("snippet") or ""), "wikipedia"))
    return out[:limit]


#: The ladder, in order.  ``None`` env var == keyless, always eligible.
_PROVIDERS: tuple[tuple[str, str | None, Callable[..., Awaitable[list[dict]]]], ...] = (
    ("tavily", "TAVILY_API_KEY", _tavily),
    ("brave", "BRAVE_API_KEY", _brave),
    ("serper", "SERPER_API_KEY", _serper),
    ("duckduckgo", None, _duckduckgo),
)


# ───── Public: search ────────────────────────────────────────────────


async def _safe(awaitable: Awaitable[list[dict]]) -> list[dict]:
    try:
        return await awaitable
    except Exception:
        return []


async def _run_ladder(client: httpx.AsyncClient, query: str, limit: int) -> list[dict]:
    for _name, env_var, provider in _PROVIDERS:
        key = ""
        if env_var:
            key = (os.getenv(env_var) or "").strip()
            if not key:
                continue
        try:
            rows = await provider(client, query, limit, key)
        except Exception:
            rows = []
        if rows:
            return rows
    return []


def _merge(primary: Iterable[dict], wiki: Iterable[dict], max_results: int) -> list[dict]:
    """De-dupe by normalized URL, then reserve tail slots so the Wikipedia
    hits always survive the ``max_results`` cap."""
    seen: set[str] = set()
    kept_primary: list[dict] = []
    for row in primary or []:
        key = normalize_url(row.get("url", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        kept_primary.append(row)
    kept_wiki: list[dict] = []
    for row in wiki or []:
        key = normalize_url(row.get("url", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        kept_wiki.append(row)
    kept_wiki = kept_wiki[:WIKI_MAX_HITS]
    room = max(0, max_results - len(kept_wiki))
    return (kept_primary[:room] + kept_wiki)[:max_results]


async def search(query: str, max_results: int = 8, *, timeout: float = 20.0) -> list[dict]:
    """Search the web.  Returns ``[{title, url, snippet, source}]``.

    First provider on the ladder that yields rows wins; Wikipedia is queried
    concurrently and always merged in.  Never raises — an all-providers-down
    situation returns ``[]``.
    """
    text = (query or "").strip()
    try:
        limit = int(max_results)
    except (TypeError, ValueError):
        limit = 8
    if not text or limit <= 0:
        return []

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            http2=HTTP2,
            timeout=httpx.Timeout(float(timeout)),
            headers=dict(_BROWSER_HEADERS),
        ) as client:
            primary, wiki = await asyncio.gather(
                _safe(_run_ladder(client, text, limit)),
                _safe(_wikipedia_search(client, text, WIKI_MAX_HITS)),
            )
    except Exception:
        return []
    return _merge(primary, wiki, limit)


# ───── Page extraction ───────────────────────────────────────────────


def _is_textual(content_type: str) -> bool:
    ctype = (content_type or "").lower()
    return ctype.startswith("text/") or ctype in _TEXTUAL_TYPES


def _decode(raw: bytes, header_encoding: str | None) -> str:
    encoding = (header_encoding or "").strip()
    if not encoding:
        hit = _META_CHARSET_RE.search(raw[:4096])
        if hit:
            encoding = hit.group(1).decode("ascii", "ignore")
    for candidate in (encoding, "utf-8"):
        if not candidate:
            continue
        try:
            return raw.decode(candidate, errors="replace")
        except (LookupError, TypeError, ValueError):
            continue
    return raw.decode("utf-8", errors="replace")


def _html_title(body: str) -> str:
    hit = _TITLE_RE.search(body)
    if hit:
        title = _strip_tags(hit.group(1))
        if title:
            return title
    for meta in _META_RE.finditer(body[:200_000]):
        tag = meta.group(0)
        if _OG_TITLE_RE.search(tag):
            content = _CONTENT_ATTR_RE.search(tag)
            if content:
                title = _one_line(_htmllib.unescape(content.group(1)))
                if title:
                    return title
    return ""


def _fallback_extract(body: str) -> str:
    """Built-in extractor for when trafilatura is absent or gives up."""
    text = _COMMENT_RE.sub(" ", body)
    text = _BOILERPLATE_RE.sub("\n", text)
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _BLOCK_BREAK_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    return _collapse_block(_htmllib.unescape(text))


async def _extract_text(body: str, url: str) -> str:
    if trafilatura is not None:
        try:
            extracted = await asyncio.to_thread(
                trafilatura.extract,
                body,
                include_comments=False,
                include_tables=True,
                favor_recall=True,
            )
            if extracted and extracted.strip():
                return _collapse_block(extracted)
        except Exception:
            pass
    return _fallback_extract(body)


def _blank_page(url: str) -> dict:
    return {"url": url, "title": "", "text": "", "truncated": False, "error": None}


async def _wikipedia_extract(client: httpx.AsyncClient, host: str, title: str) -> dict | None:
    payload = await _wikimedia_json(
        client,
        f"https://{host}/w/api.php",
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "format": "json",
            "redirects": 1,
            "titles": title,
        },
    )
    pages = ((payload.get("query") or {}).get("pages")) or {}
    for page_id, page in pages.items():
        if str(page_id) == "-1" or not isinstance(page, dict) or "missing" in page:
            continue
        extract = (page.get("extract") or "").strip()
        if not extract:
            continue
        resolved = (page.get("title") or title).strip()
        out = _blank_page(_wiki_url(resolved, host))
        out["title"] = resolved
        out["text"] = _collapse_block(extract)
        return out
    return None


async def _fetch_html(client: httpx.AsyncClient, url: str) -> dict:
    out = _blank_page(url)
    async with client.stream("GET", url, headers=dict(_BROWSER_HEADERS)) as resp:
        out["url"] = str(resp.url)
        ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        if resp.status_code >= 400:
            out["error"] = f"HTTP {resp.status_code}"
            return out
        if ctype and not _is_textual(ctype):
            out["error"] = f"unsupported content-type: {ctype}"
            return out
        chunks: list[bytes] = []
        total = 0
        async for chunk in resp.aiter_bytes(65_536):
            chunks.append(chunk)
            total += len(chunk)
            if total >= MAX_DOWNLOAD_BYTES:
                break
        header_encoding = resp.charset_encoding
    raw = b"".join(chunks)[:MAX_DOWNLOAD_BYTES]
    if not raw:
        out["error"] = "empty response body"
        return out
    if raw[:5] == b"%PDF-":
        out["error"] = "unsupported content-type: application/pdf"
        return out
    if b"\x00" in raw[:2048]:
        out["error"] = f"unsupported content-type: {ctype or 'application/octet-stream'}"
        return out

    body = _decode(raw, header_encoding)
    if ctype == "text/plain" or (not ctype and "<" not in body[:512]):
        out["text"] = _collapse_block(body)
    else:
        out["title"] = _html_title(body)
        out["text"] = await _extract_text(body, out["url"])
    if not out["text"]:
        out["error"] = "no readable text extracted"
    return out


def _finalize(page: dict, max_chars: int) -> dict:
    text = page.get("text") or ""
    truncated = False
    if max_chars > 0 and len(text) > max_chars:
        cut = text[:max_chars]
        boundary = cut.rfind(" ")
        if boundary > int(max_chars * 0.8):
            cut = cut[:boundary]
        text = cut.rstrip()
        truncated = True
    return {
        "url": str(page.get("url") or ""),
        "title": str(page.get("title") or ""),
        "text": text,
        "truncated": truncated,
        "error": page.get("error"),
    }


# ───── Public: fetch ─────────────────────────────────────────────────


async def fetch(url: str, max_chars: int = 6000, *, timeout: float = 25.0) -> dict:
    """Download ``url`` and return ``{url, title, text, truncated, error}``.

    Wikipedia articles use the MediaWiki plaintext API; everything else is
    extracted from HTML.  Never raises: transport failures, bad status codes
    and binary payloads all come back as a populated ``error`` with
    ``text == ""``.
    """
    try:
        limit = int(max_chars)
    except (TypeError, ValueError):
        limit = 6000

    target = (url or "").strip()
    out = _blank_page(target)
    if not target:
        out["error"] = "empty url"
        return out
    if target.startswith("//"):
        target = "https:" + target
    if not re.match(r"(?i)^https?://", target):
        if "://" in target:
            out["url"] = target
            out["error"] = f"unsupported scheme: {target.split('://', 1)[0].lower()}"
            return out
        target = "https://" + target
    out["url"] = target

    async def _run() -> dict:
        async with httpx.AsyncClient(
            follow_redirects=True,
            http2=HTTP2,
            timeout=httpx.Timeout(float(timeout)),
            headers={"User-Agent": UA},
        ) as client:
            wiki_title, wiki_host = _wikipedia_article(target)
            if wiki_title:
                try:
                    page = await _wikipedia_extract(client, wiki_host, wiki_title)
                except Exception:
                    page = None
                if page and page.get("text"):
                    return _finalize(page, limit)
            return _finalize(await _fetch_html(client, target), limit)

    try:
        # httpx's timeout is per-socket-operation; a slow-drip server can keep a
        # research lane hanging for minutes.  Bound the WHOLE fetch by wall clock.
        return await asyncio.wait_for(_run(), timeout=float(timeout) * 2 + 5)
    except Exception as exc:
        out["error"] = _err(exc)
        return out
