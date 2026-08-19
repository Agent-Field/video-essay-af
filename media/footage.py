"""Free stock-footage search, download, probe — plus a generated text-card fallback.

Pure async Python: ``httpx`` for HTTP, ``asyncio`` subprocesses for ffmpeg/ffprobe.
Nothing in here imports agentfield or any project module, so it can be unit-tested
and reused standalone.

Providers (the ladder tried by :func:`search_footage`, in this order):

===============  ==========================  =========================================
Provider         Env var                     How to get a key
===============  ==========================  =========================================
Pexels Videos    ``PEXELS_API_KEY``          free at https://www.pexels.com/api/
Pixabay Videos   ``PIXABAY_API_KEY``         free at https://pixabay.com/api/docs/
Wikimedia        *(none — always on)*        keyless, but requires a descriptive
Commons                                      User-Agent, which this module sends
===============  ==========================  =========================================

With **no keys at all** the module still works: Wikimedia Commons is keyless and is
always consulted when the keyed providers came up short.  Every provider call is
wrapped in ``try/except`` — a provider being down, rate-limited or misconfigured
degrades the result set but never raises out of :func:`search_footage`.

Concurrency: no module-level mutable state.  Every coroutine builds its own
``httpx.AsyncClient`` and its own temp/output paths, so many searches, downloads and
card renders can run at once.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import re
import textwrap
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import httpx

__all__ = [
    "Candidate",
    "search_footage",
    "rank_candidates",
    "download",
    "probe",
    "make_card",
    "credit_line",
]

# Wikimedia's API policy requires a descriptive UA with contact info; a generic
# python-httpx UA gets 403'd.  We reuse it for every request — it is honest and harmless.
USER_AGENT = "video-essay-af/0.1 (https://github.com/Agent-Field; contact: video-essay-af)"

PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"
PIXABAY_SEARCH_URL = "https://pixabay.com/api/videos/"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


# ─────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Candidate:
    """One stock clip we could use for a shot."""

    provider: str          # "pexels" | "pixabay" | "wikimedia"
    id: str
    title: str             # short human description
    page_url: str          # where a human would see it (for credits)
    download_url: str      # direct media URL (mp4 preferred; webm/ogv are fine — ffmpeg reads them)
    width: int
    height: int
    duration_s: float      # 0.0 if unknown
    credit: str            # "Video by <user> on Pexels", "<author> via Wikimedia Commons (CC BY-SA 4.0)"
    license: str           # "Pexels License", "Pixabay Content License", "CC BY-SA 4.0"
    score: float = 0.0     # filled by rank_candidates()


def credit_line(cand: Candidate) -> str:
    """The one-line attribution string that goes in the description / on-screen credits."""
    return f"{cand.credit} — {cand.page_url}"


# ─────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any) -> str:
    """Strip HTML tags/entities and collapse whitespace (Commons extmetadata is HTML)."""
    if not value:
        return ""
    text = _TAG_RE.sub(" ", str(value))
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def _strip_tracking(url: str) -> str:
    """Drop Wikimedia's ``?utm_source=…`` so the URL ends in its real extension."""
    if "upload.wikimedia.org" in url and "?" in url:
        return url.split("?", 1)[0]
    return url


def _is_mp4(url: str) -> bool:
    return url.split("?", 1)[0].lower().endswith(".mp4")


def _client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )


# ─────────────────────────────────────────────────────────────────────
# Provider: Pexels
# ─────────────────────────────────────────────────────────────────────


def _pick_pexels_file(video_files: Sequence[dict]) -> dict | None:
    """Prefer an mp4 at 1080p, then 720p, then the largest rendition ≤ 1440p."""
    mp4s = [
        f for f in video_files
        if str(f.get("file_type", "")).lower() == "video/mp4" or _is_mp4(str(f.get("link", "")))
    ]
    pool = mp4s or [f for f in video_files if f.get("link")]
    if not pool:
        return None
    for target in (1080, 720):
        exact = [f for f in pool if _as_int(f.get("height")) == target]
        if exact:
            return max(exact, key=lambda f: _as_int(f.get("width")))
    capped = [f for f in pool if 0 < _as_int(f.get("height")) <= 1440]
    if capped:
        return max(capped, key=lambda f: _as_int(f.get("height")))
    return min(pool, key=lambda f: _as_int(f.get("height"), 10 ** 6))


async def _search_pexels(query: str, limit: int, orientation: str, timeout: float) -> list[Candidate]:
    key = os.environ.get("PEXELS_API_KEY", "").strip()
    if not key:
        return []
    params = {
        "query": query,
        "orientation": orientation,
        "size": "medium",
        "per_page": max(1, min(80, limit)),
    }
    async with _client(timeout) as client:
        resp = await client.get(PEXELS_SEARCH_URL, params=params, headers={"Authorization": key})
        resp.raise_for_status()
        payload = resp.json()

    out: list[Candidate] = []
    for video in payload.get("videos") or []:
        chosen = _pick_pexels_file(video.get("video_files") or [])
        if not chosen or not chosen.get("link"):
            continue
        user = ((video.get("user") or {}).get("name") or "a Pexels contributor").strip()
        vid = str(video.get("id", ""))
        out.append(
            Candidate(
                provider="pexels",
                id=vid,
                title=_clean_text(video.get("alt")) or f"Pexels clip {vid}",
                page_url=str(video.get("url") or f"https://www.pexels.com/video/{vid}/"),
                download_url=str(chosen["link"]),
                width=_as_int(chosen.get("width")) or _as_int(video.get("width")),
                height=_as_int(chosen.get("height")) or _as_int(video.get("height")),
                duration_s=_as_float(video.get("duration")),
                credit=f"Video by {user} on Pexels",
                license="Pexels License",
            )
        )
    return out


# ─────────────────────────────────────────────────────────────────────
# Provider: Pixabay
# ─────────────────────────────────────────────────────────────────────


def _pick_pixabay_file(videos: dict) -> dict | None:
    """Prefer the ``medium`` (1280x720) rendition, then ``large``, then ``small``."""
    for name in ("medium", "large", "small", "tiny"):
        entry = videos.get(name)
        if isinstance(entry, dict) and entry.get("url"):
            return entry
    return None


async def _search_pixabay(query: str, limit: int, timeout: float) -> list[Candidate]:
    key = os.environ.get("PIXABAY_API_KEY", "").strip()
    if not key:
        return []
    params = {
        "key": key,
        "q": query,
        "per_page": max(3, min(200, limit)),
        "video_type": "film",
        "safesearch": "true",
    }
    async with _client(timeout) as client:
        resp = await client.get(PIXABAY_SEARCH_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()

    out: list[Candidate] = []
    for hit in payload.get("hits") or []:
        chosen = _pick_pixabay_file(hit.get("videos") or {})
        if not chosen:
            continue
        hid = str(hit.get("id", ""))
        user = str(hit.get("user") or "a Pixabay contributor").strip()
        tags = _clean_text(hit.get("tags"))
        out.append(
            Candidate(
                provider="pixabay",
                id=hid,
                title=tags or f"Pixabay clip {hid}",
                page_url=str(hit.get("pageURL") or f"https://pixabay.com/videos/id-{hid}/"),
                download_url=str(chosen["url"]),
                width=_as_int(chosen.get("width")),
                height=_as_int(chosen.get("height")),
                duration_s=_as_float(hit.get("duration")),
                credit=f"{user} on Pixabay",
                license="Pixabay Content License",
            )
        )
    return out


# ─────────────────────────────────────────────────────────────────────
# Provider: Wikimedia Commons (keyless)
# ─────────────────────────────────────────────────────────────────────


def _pick_commons_derivative(
    derivatives: Sequence[dict],
    original_url: str,
    original_height: int = 0,
) -> str:
    """720p if offered, else the largest ≤ 1080p; mp4 beats webm; original is the fallback.

    One guard on top of that ladder: Commons often transcodes only up to 480p even for a
    1080p source.  Upscaling 480p into a 1080p essay looks bad, so when every mp4/webm
    derivative is sub-720p and the original itself is 720–2160p we take the original
    (ffmpeg reads .ogv/theora fine) — that also keeps ``download_url`` consistent with the
    ``width``/``height`` we report from imageinfo.
    """
    usable: list[tuple[str, str, int]] = []  # (mime-ish type, url, height)
    for der in derivatives or []:
        url = der.get("src") or der.get("url")
        if not url:
            continue
        dtype = str(der.get("type", "")).lower()
        height = _as_int(der.get("height"))
        usable.append((dtype, str(url), height))

    original = _strip_tracking(original_url)
    for wanted in ("video/mp4", "video/webm"):
        pool = [d for d in usable if wanted in d[0]]
        if not pool:
            continue
        exact = [d for d in pool if d[2] == 720]
        if exact:
            return _strip_tracking(exact[0][1])
        capped = [d for d in pool if 0 < d[2] <= 1080]
        best = max(capped or pool, key=lambda d: d[2])
        if best[2] < 720 and 720 <= original_height <= 2160 and original:
            return original
        return _strip_tracking(best[1])
    return original


def _commons_title(page_title: str) -> str:
    """``File:Ocean_surface_waves_02.ogv`` → ``Ocean surface waves 02``."""
    name = page_title.split(":", 1)[1] if page_title.lower().startswith("file:") else page_title
    name = name.replace("_", " ").strip()
    if "." in name.rsplit(" ", 1)[-1]:
        stem, _, ext = name.rpartition(".")
        if stem and 1 <= len(ext) <= 5 and ext.isalnum():
            name = stem
    return name.strip()


async def _search_commons(query: str, limit: int, timeout: float) -> list[Candidate]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{query} filetype:video",
        "gsrnamespace": 6,
        "gsrlimit": max(1, min(50, limit)),
        "prop": "imageinfo|videoinfo",
        "iiprop": "url|size|mime|extmetadata",
        "viprop": "url|size|mime|derivatives",
    }
    async with _client(timeout) as client:
        resp = await client.get(COMMONS_API_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()

    pages = (payload.get("query") or {}).get("pages") or {}
    out: list[Candidate] = []
    # `pages` is a dict keyed by pageid; sort by the search index so results stay deterministic.
    ordered = sorted(pages.values(), key=lambda p: _as_int(p.get("index"), 10 ** 6))
    for page in ordered:
        info_list = page.get("imageinfo") or page.get("videoinfo") or []
        if not info_list:
            continue
        info = info_list[0]
        original_url = str(info.get("url") or "")
        if not original_url:
            continue
        vinfo = (page.get("videoinfo") or [{}])[0]
        derivatives = vinfo.get("derivatives") or []
        download_url = _pick_commons_derivative(
            derivatives, original_url, _as_int(info.get("height"))
        )
        # Report the dimensions of the file we will actually download (a transcode
        # is usually smaller than the original imageinfo dims).
        cand_w, cand_h = _as_int(info.get("width")), _as_int(info.get("height"))
        for der in derivatives:
            der_url = _strip_tracking(str(der.get("src") or der.get("url") or ""))
            if der_url and der_url == download_url:
                if _as_int(der.get("width")) and _as_int(der.get("height")):
                    cand_w, cand_h = _as_int(der.get("width")), _as_int(der.get("height"))
                break

        page_title = str(page.get("title") or "")
        nice_title = _commons_title(page_title)
        meta = info.get("extmetadata") or {}
        artist = _clean_text((meta.get("Artist") or {}).get("value")) or "Wikimedia Commons contributor"
        lic = _clean_text((meta.get("LicenseShortName") or {}).get("value")) or "see Commons page"
        quoted = urllib.parse.quote(page_title.replace(" ", "_"), safe=":/")

        out.append(
            Candidate(
                provider="wikimedia",
                id=str(page.get("pageid") or page_title),
                title=nice_title,
                page_url=f"https://commons.wikimedia.org/wiki/{quoted}",
                download_url=download_url,
                width=cand_w,
                height=cand_h,
                duration_s=_as_float(info.get("duration")),
                credit=f"{artist} via Wikimedia Commons ({lic})",
                license=lic,
            )
        )
    return out


# ─────────────────────────────────────────────────────────────────────
# Ranking + search
# ─────────────────────────────────────────────────────────────────────


_PROVIDER_BONUS = {"pexels": 3.0, "pixabay": 2.0, "wikimedia": 1.0}


def rank_candidates(cands: list[Candidate], min_duration_s: float) -> list[Candidate]:
    """Deterministic scoring — no LLM, no randomness.  Mutates ``.score`` and returns a sorted copy."""
    for cand in cands:
        score = _PROVIDER_BONUS.get(cand.provider, 0.0)
        if cand.duration_s >= min_duration_s:
            score += 2.0
        if 720 <= cand.height <= 1080:
            score += 1.0
        if 0 < cand.height < 480:
            score -= 1.0
        elif cand.height == 0:
            pass
        if cand.duration_s > 0 and min_duration_s > 0 and cand.duration_s < min_duration_s / 2:
            score -= 2.0
        if _is_mp4(cand.download_url):
            score += 0.5
        # Very long clips are slow to download and rarely better B-roll for a
        # shot of a few seconds — nudge them down (a good short clip wins).
        if cand.duration_s > 120:
            score -= 1.0
        if cand.duration_s > 600:
            score -= 4.0  # multi-minute files: huge downloads for a few seconds of B-roll
        cand.score = score
    # `sorted` is stable, so equal scores keep provider/merge order.
    return sorted(cands, key=lambda c: c.score, reverse=True)


def _orientation_ok(cand: Candidate, orientation: str) -> bool:
    if not cand.width or not cand.height:
        return True  # unknown dims — keep, probe() will tell us later
    if orientation == "landscape":
        return cand.width >= cand.height
    if orientation == "portrait":
        return cand.height >= cand.width
    return True


async def search_footage(
    query: str,
    *,
    min_duration_s: float = 0.0,
    limit: int = 8,
    orientation: str = "landscape",
    timeout: float = 25.0,
) -> list[Candidate]:
    """Search the provider ladder for clips matching ``query``.

    Never raises on provider failure: each provider is guarded, and an empty list is a
    perfectly normal result (the caller then falls back to :func:`make_card`).
    """
    query = (query or "").strip()
    if not query:
        return []
    limit = max(1, int(limit))

    keyed = await asyncio.gather(
        _search_pexels(query, limit, orientation, timeout),
        _search_pixabay(query, limit, timeout),
        return_exceptions=True,
    )
    merged: list[Candidate] = []
    for result in keyed:  # Pexels first, Pixabay second
        if isinstance(result, list):
            merged.extend(result)

    # Commons is keyless and always available; consult it whenever the keyed providers
    # came up short (which includes the very common "no API keys configured" case).
    if len(merged) < limit / 2:
        try:
            merged.extend(await _search_commons(query, limit, timeout))
        except Exception:
            pass

    # Dedupe, then filter.
    seen: set[tuple[str, str]] = set()
    seen_urls: set[str] = set()
    filtered: list[Candidate] = []
    # A clip can be looped, so "shorter than we need" is allowed (ranking penalises it) —
    # but sub-1.5s fragments are useless no matter what, so drop those.
    floor = min(min_duration_s, 3.0) * 0.5
    for cand in merged:
        if not cand.download_url:
            continue
        key = (cand.provider, cand.id)
        if key in seen or cand.download_url in seen_urls:
            continue
        if not _orientation_ok(cand, orientation):
            continue
        if cand.duration_s > 0 and cand.duration_s < floor:
            continue
        seen.add(key)
        seen_urls.add(cand.download_url)
        filtered.append(cand)

    return rank_candidates(filtered, min_duration_s)[:limit]


# ─────────────────────────────────────────────────────────────────────
# Download + probe
# ─────────────────────────────────────────────────────────────────────


async def download(
    cand: Candidate,
    dest: Path,
    *,
    max_bytes: int = 150_000_000,
    timeout: float = 120.0,
) -> Path:
    """Stream ``cand.download_url`` to ``dest``.  Raises RuntimeError on any failure."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        async with _client(timeout) as client:
            async with client.stream("GET", cand.download_url) as resp:
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"download failed: HTTP {resp.status_code} for {cand.download_url}"
                    )
                declared = _as_int(resp.headers.get("content-length"), 0)
                if declared > max_bytes:
                    raise RuntimeError(
                        f"download refused: {declared} bytes > max_bytes={max_bytes} "
                        f"for {cand.download_url}"
                    )
                with dest.open("wb") as handle:
                    async for chunk in resp.aiter_bytes(chunk_size=262_144):
                        written += len(chunk)
                        if written > max_bytes:
                            raise RuntimeError(
                                f"download aborted: exceeded max_bytes={max_bytes} "
                                f"for {cand.download_url}"
                            )
                        handle.write(chunk)
    except RuntimeError:
        dest.unlink(missing_ok=True)
        raise
    except Exception as exc:  # network / DNS / timeout / malformed response
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"download failed for {cand.download_url}: {exc!r}") from exc

    if not dest.exists() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"download produced an empty file for {cand.download_url}")
    return dest


async def _run(cmd: Sequence[str]) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return proc.returncode or 0, out, err


async def probe(path: Path) -> dict:
    """ffprobe a media file.  Never raises — ``ok`` is False when anything is wrong."""
    bad = {
        "duration": 0.0,
        "width": 0,
        "height": 0,
        "has_video": False,
        "has_audio": False,
        "vcodec": "",
        "ok": False,
    }
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return bad
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries",
        "format=duration:stream=codec_type,width,height,r_frame_rate,codec_name",
        "-of", "json", str(path),
    ]
    try:
        rc, out, _err = await _run(cmd)
        if rc != 0:
            return bad
        data = json.loads(out.decode("utf-8", "replace") or "{}")
    except Exception:
        return bad

    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = _as_float((data.get("format") or {}).get("duration"))
    width = _as_int((video or {}).get("width"))
    height = _as_int((video or {}).get("height"))
    result = {
        "duration": duration,
        "width": width,
        "height": height,
        "has_video": video is not None,
        "has_audio": audio is not None,
        "vcodec": str((video or {}).get("codec_name") or ""),
        "ok": video is not None and duration > 0,
    }
    return result


# ─────────────────────────────────────────────────────────────────────
# Kinetic text card (the "no footage fits" fallback)
# ─────────────────────────────────────────────────────────────────────


def _pick_font(font_path: str | None) -> str | None:
    if font_path and Path(font_path).exists():
        return font_path
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def _wrap(text: str, width: int = 28) -> str:
    lines: list[str] = []
    for para in (text or "").splitlines() or [""]:
        para = para.strip()
        if not para:
            continue
        lines.extend(textwrap.wrap(para, width=width, break_long_words=True, break_on_hyphens=True) or [""])
    return "\n".join(lines)


def _escape_drawtext(text: str) -> str:
    """Escape a string for inline use as a drawtext ``text=`` value in a filtergraph.

    A filter argument goes through **two** unescape passes — the filtergraph description
    parser and then ``av_opt_set_from_string`` — and the two passes care about different
    characters, so the escape depth differs per character.  Every rule below was validated
    byte-for-byte against a ``textfile=`` render of the same string (which needs no
    escaping at all), on ffmpeg 4.4:

    * ``\\`` → ``\\\\\\\\`` and ``'`` → ``\\\\\\'`` (special at both levels)
    * ``:`` and ``=`` → ``\\\\:`` (AVOption separators, so two levels of escaping)
    * ``,`` ``;`` ``[`` ``]`` → ``\\,`` (filtergraph-level only)
    * newline → left as a real newline character; drawtext breaks lines on those, while the
      two-character sequence ``\\n`` is NOT interpreted
    * ``%`` → left alone; we set ``expansion=none`` which makes it literal, whereas
      escaping it makes ffmpeg log a "Stray %" warning

    Getting ``'`` wrong is not a cosmetic bug: an under-escaped apostrophe swallows the rest
    of the filter string into ``text=``, which silently renders a card with no text on it.
    """
    text = text.replace("\\", "\\\\\\\\")
    text = text.replace("'", "\\\\\\'")
    for ch in (":", "="):
        text = text.replace(ch, "\\\\" + ch)
    for ch in (",", ";", "[", "]"):
        text = text.replace(ch, "\\" + ch)
    return text


def _card_filtergraph(
    text: str,
    duration_s: float,
    width: int,
    height: int,
    fps: int,
    theme: str,
    font: str | None,
    use_gradients: bool,
) -> tuple[str, str]:
    """Return ``(lavfi_source, filter_chain)`` for the card."""
    light = str(theme).lower() in {"light", "paper", "white"}
    c0, c1 = ("#f8fafc", "#e2e8f0") if light else ("#0f172a", "#1e293b")
    fontcolor = "#0f172a" if light else "#ffffff"
    shadow = "white@0.35" if light else "black@0.45"

    if use_gradients:
        # A slowly-rotating two-stop gradient: the "kinetic" part of the card.
        source = (
            f"gradients=s={width}x{height}:c0={c0}:c1={c1}:speed=0.02:"
            f"d={duration_s:.3f}:r={fps}:seed=1729"
        )
    else:
        flat = "#111827" if not light else "#f1f5f9"
        source = f"color=c={flat}:s={width}x{height}:d={duration_s:.3f}:r={fps}"

    steps: list[str] = []
    if text.strip():
        fontsize = max(18, round(64 * width / 1920))
        wrapped = _wrap(text, 28)
        drawtext = [
            f"text={_escape_drawtext(wrapped)}",
            f"fontsize={fontsize}",
            f"fontcolor={fontcolor}",
            f"line_spacing={max(6, round(fontsize * 0.35))}",
            "x=(w-text_w)/2",
            "y=(h-text_h)/2",
            f"shadowcolor={shadow}",
            "shadowx=2",
            "shadowy=2",
            "expansion=none",
        ]
        if font:
            drawtext.insert(0, f"fontfile={font}")
        steps.append("drawtext=" + ":".join(drawtext))

    fade_in = min(0.4, max(0.05, duration_s / 4))
    fade_out = min(0.5, max(0.05, duration_s / 4))
    steps.append(f"fade=t=in:st=0:d={fade_in:.3f}")
    steps.append(f"fade=t=out:st={max(0.0, duration_s - fade_out):.3f}:d={fade_out:.3f}")
    steps.append("format=yuv420p")
    return source, ",".join(steps)


async def make_card(
    text: str,
    duration_s: float,
    dest: Path,
    *,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    theme: str = "dark",
    font_path: str | None = None,
) -> Path:
    """Render a silent H.264 "kinetic text card" — the fallback when no footage fits.

    Empty ``text`` renders just the drifting background.  Raises RuntimeError (with the
    tail of ffmpeg's stderr) if rendering fails.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    duration_s = max(0.2, float(duration_s))
    font = _pick_font(font_path)

    errors: list[str] = []
    # `gradients` needs ffmpeg >= 4.3; if this build lacks it, fall back to a flat colour.
    for use_gradients in (True, False):
        source, chain = _card_filtergraph(
            text or "", duration_s, width, height, fps, theme, font, use_gradients
        )
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-v", "error",
            "-f", "lavfi", "-i", source,
            "-vf", chain,
            "-t", f"{duration_s:.3f}",
            "-r", str(fps),
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(dest),
        ]
        try:
            rc, _out, err = await _run(cmd)
        except FileNotFoundError as exc:
            raise RuntimeError(f"ffmpeg not found on PATH: {exc!r}") from exc
        if rc == 0 and dest.exists() and dest.stat().st_size > 0:
            return dest
        errors.append(err.decode("utf-8", "replace").strip()[-1200:])
        dest.unlink(missing_ok=True)

    raise RuntimeError("make_card: ffmpeg failed.\n" + "\n---\n".join(e for e in errors if e))
