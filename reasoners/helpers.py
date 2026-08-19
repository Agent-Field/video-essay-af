"""Plain-Python utilities shared by the routers.  No decorators, no LLM calls.

Three families live here:

* **Prose renderers** — Pydantic → natural language.  Anything another LLM
  reads is rendered to prose first; JSON is only for code that branches on it.
* **Deterministic work** — sentence splitting, claim keys, URL normalisation,
  word budgets, coverage repair.  Code does code work.
* **Fallback constructors** — safe-default instances returned when an
  ``.ai()`` gate reports ``confident=False`` and there is no deeper slot.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Sequence
from urllib.parse import urlsplit, urlunsplit

from .models import (
    Brief,
    Claim,
    ClaimVerdict,
    Dossier,
    EditorReport,
    OutlineSection,
    Sentence,
    SentenceTiming,
    Shot,
    ShotAsset,
)

# ───── Run directories ───────────────────────────────────────────────

OUTPUT_ROOT = Path(os.getenv("VEA_OUTPUT_DIR", "output")).resolve()

# Spoken-word pace used for every word budget (~150 wpm is documentary pace).
WORDS_PER_MINUTE = float(os.getenv("VEA_WPM", "150"))


def new_run_id(prefix: str = "essay") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


def run_dir(run_id: str, *parts: str) -> Path:
    p = OUTPUT_ROOT / run_id
    for part in parts:
        p = p / part
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def slug(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:limit] or "essay"


# ───── Words, seconds, sentences ─────────────────────────────────────


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9'’\-]+", text))


def seconds_for_words(n_words: int) -> float:
    return n_words / WORDS_PER_MINUTE * 60.0


def words_for_seconds(seconds: float) -> int:
    return int(round(seconds / 60.0 * WORDS_PER_MINUTE))


_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "u.s", "u.k", "no", "fig", "al"}
_SENT_END = re.compile(r"(?<=[.!?])[\"'”’)]*\s+(?=[\"'“‘(]*[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Split prose into spoken sentences.  Conservative: keeps abbreviations
    and decimals together, splits on . ! ? followed by whitespace + capital."""
    text = " ".join(text.split())
    if not text:
        return []
    pieces: list[str] = []
    start = 0
    for m in _SENT_END.finditer(text):
        candidate = text[start : m.end()].strip()
        last_word = re.findall(r"[A-Za-z.]+(?=[.!?][\"'”’)]*$)", candidate)
        if last_word and last_word[-1].lower().rstrip(".") in _ABBREV:
            continue
        pieces.append(candidate)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return [p for p in pieces if p]


def index_sentences(sections: Sequence[str]) -> list[Sentence]:
    out: list[Sentence] = []
    for s_idx, section in enumerate(sections):
        for sent in split_sentences(section):
            out.append(Sentence(index=len(out), section_index=s_idx, text=sent))
    return out


# ───── Claims & citations ────────────────────────────────────────────


def normalize_url(url: str) -> str:
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip().lower()
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower() or "https", host, path, parts.query, ""))


def assign_claim_keys(claims: Iterable[Claim], start: int = 1) -> list[Claim]:
    """Dedupe near-identical claims and assign C01.. keys deterministically."""
    seen: set[str] = set()
    out: list[Claim] = []
    n = start
    for c in claims:
        fingerprint = re.sub(r"[^a-z0-9 ]", "", c.text.lower())
        fingerprint = " ".join(fingerprint.split()[:12])
        if fingerprint in seen or not c.text.strip():
            continue
        seen.add(fingerprint)
        out.append(c.model_copy(update={"key": f"C{n:02d}"}))
        n += 1
    return out


_STRENGTH_RANK = {"strong": 0, "moderate": 1, "weak": 2}
_KIND_RANK = {"number": 0, "fact": 1, "quote": 2, "counterpoint": 3, "anecdote": 4, "context": 5}


def load_bearing_claims(claims: Sequence[Claim], k: int) -> list[Claim]:
    """The claims most likely to carry the essay — verify these first."""
    ranked = sorted(claims, key=lambda c: (_STRENGTH_RANK[c.strength], _KIND_RANK[c.kind]))
    return ranked[:k]


def apply_verdicts(claims: Sequence[Claim], verdicts: Sequence[ClaimVerdict]) -> tuple[list[Claim], list[str]]:
    """Drop unsupported claims, apply corrections, collect caveats for contested ones."""
    by_key = {v.key: v for v in verdicts}
    kept: list[Claim] = []
    caveats: list[str] = []
    for c in claims:
        v = by_key.get(c.key)
        if v is None:
            kept.append(c)
            continue
        if v.verdict == "unsupported" and v.confident:
            caveats.append(f"Dropped {c.key} ('{c.text[:80]}…'): {v.rationale}")
            continue
        fixable = v.verdict in {"contested", "unsupported"} and v.confident and bool(v.corrected_text and v.corrected_text.strip())
        text = v.corrected_text.strip() if fixable else c.text
        strength = c.strength
        if v.verdict == "contested":
            strength = "weak" if c.strength == "moderate" else ("moderate" if c.strength == "strong" else "weak")
            caveats.append(f"{c.key} is contested: {v.rationale}")
        if v.verdict == "unverifiable":
            strength = "weak"
        kept.append(c.model_copy(update={"text": text, "strength": strength}))
    return kept, caveats


def claims_by_keys(claims: Sequence[Claim], keys: Iterable[str]) -> list[Claim]:
    wanted = set(keys)
    return [c for c in claims if c.key in wanted]


def known_keys(claims: Sequence[Claim]) -> set[str]:
    return {c.key for c in claims}


# ───── Prose renderers (Pydantic → text another LLM reads) ───────────


def brief_prose(b: Brief) -> str:
    rq = "\n".join(f"  - {q}" for q in b.research_questions)
    avoid = "\n".join(f"  - {a}" for a in b.must_avoid) or "  - (none)"
    return (
        f"Topic: {b.topic}\n"
        f"Central question: {b.central_question}\n"
        f"Audience: {b.audience}\n"
        f"Tone: {b.tone}\n"
        f"Target runtime: {b.target_minutes:g} minutes (~{words_for_seconds(b.target_minutes * 60)} spoken words)\n"
        f"Research questions:\n{rq}\n"
        f"Avoid:\n{avoid}"
    )


def claims_prose(claims: Sequence[Claim], with_evidence: bool = True) -> str:
    if not claims:
        return "(no claims)"
    lines = []
    for c in claims:
        line = f"[{c.key}] ({c.kind}, {c.strength}) {c.text}"
        if with_evidence and c.evidence:
            line += f'\n      evidence: "{c.evidence[:240]}"'
        line += f"\n      source: {c.source_title or c.source_url} — {c.source_url}"
        lines.append(line)
    return "\n".join(lines)


def dossier_prose(d: Dossier, claims: Sequence[Claim]) -> str:
    story = "\n".join(f"  - {s}" for s in d.storylines)
    tens = "\n".join(f"  - {t}" for t in d.tensions) or "  - (none recorded)"
    key = claims_by_keys(claims, d.key_claim_keys) or list(claims)
    return (
        f"RESEARCH SUMMARY\n{d.summary}\n\n"
        f"STORYLINES WORTH TELLING\n{story}\n\n"
        f"TENSIONS / CAVEATS\n{tens}\n\n"
        f"LOAD-BEARING CLAIMS (cite by key)\n{claims_prose(key)}"
    )


def outline_prose(sections: Sequence[OutlineSection]) -> str:
    out = []
    for i, s in enumerate(sections):
        beats = "\n".join(f"      • {b}" for b in s.beats)
        out.append(
            f"Section {i + 1}: {s.title} (~{s.target_seconds}s, ~{words_for_seconds(s.target_seconds)} words)\n"
            f"    purpose: {s.purpose}\n    beats:\n{beats}\n    claims: {', '.join(s.claim_keys) or '—'}"
        )
    return "\n".join(out)


def draft_prose(sections: Sequence[str], titles: Sequence[str] | None = None) -> str:
    parts = []
    for i, text in enumerate(sections):
        title = titles[i] if titles and i < len(titles) else f"Section {i + 1}"
        parts.append(f"### [{i}] {title}\n{text.strip()}")
    return "\n\n".join(parts)


def draft_markdown(sections: Sequence[str], titles: Sequence[str] | None = None) -> str:
    parts = []
    for i, text in enumerate(sections):
        title = titles[i] if titles and i < len(titles) else f"Section {i + 1}"
        parts.append(f"## {title}\n\n{text.strip()}")
    return "\n\n".join(parts) + "\n"


def reports_prose(reports: Sequence[EditorReport]) -> str:
    out = []
    for r in reports:
        if not r.notes:
            out.append(f"{r.lens.upper()} EDITOR — no notes. {r.overall}")
            continue
        notes = "\n".join(
            f"  - [{n.severity}] §{n.section_index} “{n.quote[:90]}” — {n.issue} → fix: {n.suggested_fix}"
            for n in r.notes
        )
        out.append(f"{r.lens.upper()} EDITOR — {r.overall}\n{notes}")
    return "\n\n".join(out)


def sentences_prose(sentences: Sequence[Sentence]) -> str:
    return "\n".join(f"[{s.index}] (§{s.section_index}) {s.text}" for s in sentences)


def shot_prose(shot: Shot, sentences: Sequence[Sentence]) -> str:
    spoken = " ".join(s.text for s in sentences if shot.sentence_start <= s.index <= shot.sentence_end)
    return (
        f"Shot {shot.index} — mood: {shot.mood}\n"
        f"Narration over this shot: {spoken}\n"
        f"Search queries tried: {', '.join(shot.queries)}"
    )


# ───── Footage query simplification (code, not LLM) ──────────────────

_QUERY_DROP = {
    "archival", "archive", "vintage", "historic", "historical", "old", "footage", "video", "clip",
    "aerial", "drone", "timelapse", "time-lapse", "slow", "motion", "close-up", "closeup", "close",
    "up", "shot", "wide", "cinematic", "stock", "b-roll", "broll", "scene", "view", "of", "the", "a",
    "an", "in", "at", "on", "with", "and", "from", "to", "by", "for", "into", "over", "showing",
    "looking", "being", "modern", "busy", "large", "small", "big", "beautiful", "dramatic", "era",
    "century", "style", "black", "white", "colour", "color", "1900s", "1910s", "1920s", "1930s",
    "1940s", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s",
}


def simplify_query(query: str, max_words: int = 3) -> str:
    """'converted tanker ship leaving port 1950s archival' → 'tanker ship port'."""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]+", query.lower()) if w not in _QUERY_DROP]
    return " ".join(words[:max_words])


# ───── Coverage repair for shot lists (rung 3, in code) ──────────────


def repair_shot_coverage(shots: Sequence[Shot], n_sentences: int, max_shots: int) -> list[Shot]:
    """Make the shot list a clean partition of sentence indices 0..n-1.

    Sorts by start, clamps, removes overlaps, fills gaps by extending the
    previous shot, and merges down to ``max_shots``.  Never returns empty when
    n_sentences > 0.
    """
    if n_sentences <= 0:
        return []
    cleaned: list[Shot] = []
    for s in sorted(shots, key=lambda x: (x.sentence_start, x.sentence_end)):
        a = max(0, min(n_sentences - 1, s.sentence_start))
        b = max(a, min(n_sentences - 1, s.sentence_end))
        if cleaned and a <= cleaned[-1].sentence_end:
            a = cleaned[-1].sentence_end + 1
            if a > b:
                continue
        if cleaned and a > cleaned[-1].sentence_end + 1:
            prev = cleaned[-1]
            cleaned[-1] = prev.model_copy(update={"sentence_end": a - 1})
        if not cleaned and a > 0:
            a = 0
        cleaned.append(s.model_copy(update={"sentence_start": a, "sentence_end": b}))
    if not cleaned:
        cleaned = [Shot(index=0, sentence_start=0, sentence_end=n_sentences - 1, queries=["abstract light background"], mood="neutral")]
    if cleaned[-1].sentence_end < n_sentences - 1:
        cleaned[-1] = cleaned[-1].model_copy(update={"sentence_end": n_sentences - 1})
    while len(cleaned) > max_shots:
        # merge the shortest neighbouring pair
        spans = [s.sentence_end - s.sentence_start + 1 for s in cleaned]
        i = min(range(len(cleaned) - 1), key=lambda k: spans[k] + spans[k + 1])
        merged = cleaned[i].model_copy(update={"sentence_end": cleaned[i + 1].sentence_end})
        cleaned[i : i + 2] = [merged]
    return [s.model_copy(update={"index": i}) for i, s in enumerate(cleaned)]


def shot_seconds(shot: Shot, timings: Sequence[SentenceTiming]) -> tuple[float, float]:
    by_idx = {t.index: t for t in timings}
    start = by_idx[shot.sentence_start].start_s if shot.sentence_start in by_idx else 0.0
    end = by_idx[shot.sentence_end].end_s if shot.sentence_end in by_idx else start
    return start, max(end, start + 0.5)


def estimate_shot_seconds(shot: Shot, sentences: Sequence[Sentence]) -> float:
    words = sum(word_count(s.text) for s in sentences if shot.sentence_start <= s.index <= shot.sentence_end)
    return max(2.0, seconds_for_words(words))


# ───── Fallback constructors (rung-4 safe defaults) ──────────────────


def fallback_brief(prompt: str, target_minutes: float, style: str) -> Brief:
    return Brief(
        topic=prompt.strip()[:120],
        central_question=f"What should a curious viewer understand about: {prompt.strip()[:120]}?",
        audience="General curious adults with no specialist background",
        tone=style or "curious explainer",
        target_minutes=target_minutes,
        research_questions=[
            f"What is {prompt.strip()[:80]} and why does it matter?",
            "What is the origin story or history?",
            "How does it actually work, mechanically?",
            "What is surprising, contested, or commonly misunderstood?",
        ],
        must_avoid=[],
        confident=False,
    )


def fallback_dossier(claims: Sequence[Claim], reason: str) -> Dossier:
    keys = [c.key for c in load_bearing_claims(claims, 12)]
    return Dossier(
        summary=f"Automatic dossier (synthesizer not confident: {reason}). {len(claims)} claims gathered; see key claims.",
        storylines=["Tell the story the strongest claims support, in chronological order."],
        key_claim_keys=keys,
        tensions=[],
        confident=False,
    )


def fallback_assets_credit_lines(assets: Sequence[ShotAsset]) -> list[str]:
    lines = []
    for a in assets:
        if a.kind == "footage" and a.credit:
            lines.append(f"{a.credit} — {a.source_url}")
    # dedupe preserving order
    seen: set[str] = set()
    out = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def fmt_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


__all__ = [name for name in globals() if not name.startswith("_")]
