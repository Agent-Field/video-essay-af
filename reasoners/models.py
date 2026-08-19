"""Every Pydantic schema used anywhere in video-essay-af.

One module so every reasoner, helper and media module type-checks against the
same contract.  Rules:

* Every schema that is the target of an ``app.ai(schema=...)`` call carries a
  ``confident: bool`` field — that is verification rung 4 (self-report +
  escalate) and the call site always has a fallback path.
* Schemas that only travel between Python functions (``ShotAsset``,
  ``SentenceTiming`` …) do not need ``confident``.
* ``app.call`` crosses a serialization boundary: a Pydantic instance goes in,
  a plain dict comes out.  Receivers reconstruct with ``Model(**payload)``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────
# Phase 0 — brief
# ─────────────────────────────────────────────────────────────────────


class Brief(BaseModel):
    """What the essay is *about* and who it is *for* — derived from the prompt."""

    topic: str = Field(description="Short noun-phrase topic, e.g. 'the invention of the shipping container'")
    central_question: str = Field(description="The one question the essay exists to answer")
    audience: str = Field(description="Who is watching and what they already know")
    tone: str = Field(description="Narrator register, e.g. 'curious explainer', 'wry investigative', 'warm documentary'")
    target_minutes: float = Field(description="Target runtime in minutes")
    research_questions: list[str] = Field(description="3–8 concrete questions research must answer")
    must_avoid: list[str] = Field(default_factory=list, description="Angles, clichés or claims to steer away from")
    confident: bool


# ─────────────────────────────────────────────────────────────────────
# Phase 1 — research
# ─────────────────────────────────────────────────────────────────────


class ResearchLane(BaseModel):
    """One researcher's assignment, written at runtime by the research lead."""

    name: str = Field(description="Short lane label, e.g. 'origins & key figures'")
    brief: str = Field(description="2–4 sentences: what to find, what 'good' looks like, what to ignore")
    queries: list[str] = Field(description="2–4 seed search queries")


class ResearchPlan(BaseModel):
    lanes: list[ResearchLane] = Field(description="3–6 independent research lanes")
    confident: bool


class Claim(BaseModel):
    key: str = Field(default="", description="Citation key, assigned in code (C01, C02 …)")
    text: str = Field(description="One self-contained factual sentence")
    kind: Literal["fact", "number", "anecdote", "quote", "counterpoint", "context"]
    source_url: str
    source_title: str = ""
    evidence: str = Field(default="", description="Short quote or close paraphrase from the source that supports the claim")
    strength: Literal["strong", "moderate", "weak"] = "moderate"


class LaneFindings(BaseModel):
    lane: str
    claims: list[Claim]
    notes: str = Field(description="Prose summary of what this lane learned, 80–200 words")
    dead_ends: list[str] = Field(default_factory=list)
    confident: bool


class ClaimVerdict(BaseModel):
    key: str
    verdict: Literal["supported", "contested", "unsupported", "unverifiable"]
    rationale: str
    corrected_text: str | None = Field(default=None, description="If the claim is wrong in a fixable way, the corrected sentence")
    confident: bool


class GapReport(BaseModel):
    gaps: list[ResearchLane] = Field(default_factory=list, description="Up to 2 new lanes that would close the most important holes")
    needs_more: bool
    rationale: str
    confident: bool


class Dossier(BaseModel):
    summary: str = Field(description="Prose overview of what is known, 150–300 words")
    storylines: list[str] = Field(description="2–5 narrative threads worth telling, each one sentence")
    key_claim_keys: list[str] = Field(description="Citation keys of the claims that could carry an essay")
    tensions: list[str] = Field(default_factory=list, description="Contested points, caveats, or things the evidence does NOT support")
    confident: bool


# ─────────────────────────────────────────────────────────────────────
# Phase 2 — script
# ─────────────────────────────────────────────────────────────────────


class Angle(BaseModel):
    framing: str
    thesis: str = Field(description="One sentence the whole essay argues")
    hook: str = Field(description="The opening 1–2 spoken sentences")
    promise: str = Field(description="What the viewer will understand by the end")
    claim_keys: list[str] = Field(description="Citation keys the angle leans on")
    confident: bool


class AngleVerdict(BaseModel):
    winner_index: int
    scores: list[float] = Field(description="One 0–10 score per candidate, same order")
    rationale: str
    confident: bool


class OutlineSection(BaseModel):
    title: str
    purpose: str = Field(description="What this section must accomplish for the argument")
    beats: list[str] = Field(description="2–5 beats, each one sentence")
    claim_keys: list[str] = Field(default_factory=list)
    target_seconds: int


class Outline(BaseModel):
    sections: list[OutlineSection]
    confident: bool


class SectionDraft(BaseModel):
    narration: str = Field(description="Spoken narration for this section only, plain prose, no headings")
    confident: bool


class FullDraft(BaseModel):
    sections: list[str] = Field(description="Narration per section, same count and order as the input")
    confident: bool


# ─────────────────────────────────────────────────────────────────────
# Phase 3 — editorial
# ─────────────────────────────────────────────────────────────────────


class EditorNote(BaseModel):
    section_index: int
    quote: str = Field(description="Short verbatim excerpt the note refers to")
    issue: str
    severity: Literal["blocking", "major", "minor"]
    suggested_fix: str


class EditorReport(BaseModel):
    lens: str
    notes: list[EditorNote]
    overall: str = Field(description="Two-sentence overall verdict from this lens")
    confident: bool


class ShipVerdict(BaseModel):
    ship: bool
    blocking_reasons: list[str] = Field(default_factory=list)
    confident: bool


class Package(BaseModel):
    title: str = Field(description="YouTube-style title, ≤ 70 chars, no clickbait lies")
    description: str = Field(description="2–4 sentence description")
    chapter_titles: list[str] = Field(description="One short chapter title per section, same order")
    tags: list[str] = Field(default_factory=list)
    confident: bool


# ─────────────────────────────────────────────────────────────────────
# Phase 4 — production
# ─────────────────────────────────────────────────────────────────────


class Shot(BaseModel):
    index: int
    sentence_start: int = Field(description="First sentence index covered (inclusive)")
    sentence_end: int = Field(description="Last sentence index covered (inclusive)")
    queries: list[str] = Field(description="2–3 stock-footage search queries: concrete, visual nouns (e.g. 'container ship at port aerial')")
    card_text: str = Field(default="", description="2–6 word headline for this shot (a name, number, place or key phrase) — shown on a text card if no footage fits")
    on_screen_text: str | None = Field(default=None, description="Optional lower-third text for KEY moments only (a name, a number), else null")
    mood: str = Field(default="neutral")


class ShotList(BaseModel):
    shots: list[Shot]
    confident: bool


class FootagePick(BaseModel):
    candidate_index: int = Field(description="Index of the best candidate, or -1 if none fits")
    rationale: str
    confident: bool


class AssemblyReport(BaseModel):
    """What the finishing editor (the aforge harness) reports back."""

    output_path: str = Field(description="Absolute path of the rendered final.mp4")
    duration_s: float
    method: str = Field(description="One paragraph: the approach and the ffmpeg filter graph used")
    issues: list[str] = Field(default_factory=list)
    confident: bool


# ─────────────────────────────────────────────────────────────────────
# Plain data carriers (no LLM) — travel between python functions
# ─────────────────────────────────────────────────────────────────────


class Sentence(BaseModel):
    index: int
    section_index: int
    text: str


class SentenceTiming(BaseModel):
    index: int
    start_s: float
    end_s: float


class ShotAsset(BaseModel):
    shot_index: int
    path: str
    kind: Literal["footage", "card"]
    provider: str
    credit: str = ""
    source_url: str = ""
    duration_s: float = 0.0
    query_used: str = ""


class EDLEntry(BaseModel):
    index: int
    clip: str
    start_s: float
    duration_s: float
    on_screen_text: str | None = None
    sentence_start: int
    sentence_end: int


__all__ = [name for name in globals() if name[:1].isupper()]
