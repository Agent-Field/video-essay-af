"""Phase 2 — from dossier to first full draft.

Cognitive jobs:

* ``propose_angle``     — three parallel angle-writers, each given a different
                           framing (meta-prompted: the framing is a kwarg).
* ``judge_angles``      — a critic picks the angle (rung 5: independent judge).
* ``build_outline``     — the script architect: sections, beats, claim keys,
                           seconds per section (code normalises the budget).
* ``write_section``     — parallel section writers; each gets its section,
                           the neighbouring sections' purposes for continuity,
                           only the claims it may cite, and a word budget that
                           code checks (rung 3, one retry).
* ``smooth_transitions``— one pass to make the sections read as one essay.
* ``write_script``      — orchestrator.
"""

from __future__ import annotations

import asyncio
import os

from agentfield import AgentRouter

from .helpers import (
    claims_by_keys,
    claims_prose,
    draft_markdown,
    draft_prose,
    known_keys,
    run_dir,
    word_count,
    words_for_seconds,
    write_json,
    write_text,
)
from .models import (
    Angle,
    AngleVerdict,
    Claim,
    FullDraft,
    Outline,
    OutlineSection,
    SectionDraft,
)

NODE_ID = os.getenv("AGENT_NODE_ID", "video-essay-af")
router = AgentRouter(prefix="", tags=["script"])

LLM_TIMEOUT = float(os.getenv("VEA_LLM_TIMEOUT", "300"))

FRAMINGS = [
    "the counter-intuitive reversal — the thing everyone assumes is wrong, and here is what is actually true",
    "the human story — follow one real, claim-backed person, place or decisive moment (no invented characters) and let the big idea emerge from it",
    "the stakes-and-consequences frame — start from why it matters now, then go back to how we got here",
]
MAX_SECTIONS = 9
MIN_SECTIONS = 3
WORD_TOLERANCE = 0.28  # ±28 % of the word budget before a rewrite is requested


STYLE_GUIDE = (
    "Style: spoken narration for a video essay — conversational but precise, short "
    "sentences, concrete nouns, no bullet points, no headings, no stage directions, no "
    "'in this video'. Numbers written as words when small. Every factual sentence must be "
    "supportable by a provided claim; cite nothing you were not given. NEVER invent named "
    "people, quotes, dates, place-specific events or numbers that are not in the claims — a "
    "scene may be illustrative only if it is openly hypothetical ('Picture a dockworker…', "
    "'Imagine…') and names no real-sounding person. Do not mention claim keys in the "
    "narration. Avoid clichés ('in today's fast-paced world'). Vary rhythm."
)


@router.reasoner()
async def propose_angle(dossier_text: str, framing: str, audience: str, tone: str, model: str | None = None) -> Angle:
    """One angle-writer, one framing."""
    angle = await router.ai(
        system=(
            "You are a video-essay showrunner pitching the ANGLE for an essay. Using ONLY the "
            f"research dossier, propose an angle in this framing: {framing}. Give a one-sentence "
            "thesis the evidence can defend, a 1–2 sentence spoken hook that opens with a "
            "concrete image or question (no 'have you ever wondered'), the promise to the "
            "viewer, and the citation keys the angle leans on. confident=false if the "
            "dossier cannot support this framing honestly."
        ),
        user=f"AUDIENCE: {audience}\nTONE: {tone}\n\nDOSSIER\n{dossier_text}",
        schema=Angle,
        model=model,
        timeout=LLM_TIMEOUT,
    )
    return angle.model_copy(update={"framing": framing})


@router.reasoner()
async def judge_angles(angles_text: str, dossier_text: str, model: str | None = None) -> AngleVerdict:
    """Independent critic: which angle wins on hook, defensibility, narratability."""
    return await router.ai(
        system=(
            "You are a ruthless story editor judging candidate angles for a video essay. Score "
            "each 0–10 on: hook strength (would a viewer stay past 15 s?), defensibility (does "
            "the dossier actually support the thesis?), narratability (is there a clear "
            "through-line?), and freshness. Pick the winner (index in the given order). "
            "confident=false if all candidates are weak."
        ),
        user=f"CANDIDATE ANGLES\n{angles_text}\n\nDOSSIER (for defensibility)\n{dossier_text}",
        schema=AngleVerdict,
        model=model,
        timeout=LLM_TIMEOUT,
    )


@router.reasoner()
async def build_outline(dossier_text: str, angle_text: str, target_minutes: float, model: str | None = None) -> Outline:
    """Script architect: sections with purpose, beats, claim keys, seconds."""
    total = int(target_minutes * 60)
    n_hint = max(MIN_SECTIONS, min(MAX_SECTIONS, int(round(target_minutes * 1.4)) + 1))
    outline = await router.ai(
        system=(
            "You are the script architect of a video essay. Build an outline that delivers the "
            f"chosen angle in about {total} seconds total, using roughly {n_hint} sections: a cold "
            "open that earns attention in the first 20 seconds, a setup of the question, 2–5 "
            "body sections that each move the argument one step (each with a concrete example, "
            "number or story from the dossier), a turn/complication, and an ending that pays off "
            "the hook without a generic summary. For each section give title, purpose, 2–5 beats, "
            "the citation keys it will use, and target_seconds (they should sum to the total). "
            "confident=false if the dossier cannot fill this runtime."
        ),
        user=f"TARGET: {total} seconds\n\nANGLE\n{angle_text}\n\nDOSSIER\n{dossier_text}",
        schema=Outline,
        model=model,
        timeout=LLM_TIMEOUT,
    )
    return outline


def _normalize_outline(outline: Outline, target_seconds: int, claims: list[Claim]) -> Outline:
    """Rung 3 in code: bounded section count, seconds sum to target, keys exist."""
    secs = list(outline.sections)[:MAX_SECTIONS]
    if not secs:
        secs = [
            OutlineSection(title="Cold open", purpose="Hook the viewer with the central question", beats=["Open on a concrete image", "Pose the question"], claim_keys=[], target_seconds=20),
            OutlineSection(title="How we got here", purpose="Give the origin story", beats=["The origin", "The turning point"], claim_keys=[], target_seconds=60),
            OutlineSection(title="How it works", purpose="Explain the mechanism", beats=["The mechanism", "A concrete example"], claim_keys=[], target_seconds=60),
            OutlineSection(title="Why it matters", purpose="Pay off the hook", beats=["Consequences", "Return to the opening image"], claim_keys=[], target_seconds=40),
        ]
    valid = known_keys(claims)
    cleaned: list[OutlineSection] = []
    for s in secs:
        keys = [k for k in s.claim_keys if k in valid]
        cleaned.append(s.model_copy(update={"claim_keys": keys, "target_seconds": max(10, int(s.target_seconds))}))
    # The 10 s floor must fit: merge the smallest neighbouring sections until it does.
    while len(cleaned) > MIN_SECTIONS and len(cleaned) * 10 > target_seconds:
        i = min(range(len(cleaned) - 1), key=lambda k: cleaned[k].target_seconds + cleaned[k + 1].target_seconds)
        a, b = cleaned[i], cleaned[i + 1]
        cleaned[i : i + 2] = [a.model_copy(update={
            "title": a.title, "purpose": f"{a.purpose} {b.purpose}", "beats": a.beats + b.beats,
            "claim_keys": list(dict.fromkeys(a.claim_keys + b.claim_keys)), "target_seconds": a.target_seconds + b.target_seconds,
        })]
    total = sum(s.target_seconds for s in cleaned) or 1
    scale = target_seconds / total
    secs_out = [max(10, int(round(s.target_seconds * scale))) for s in cleaned]
    # The floor can push the sum over target — take the excess from the largest sections.
    for _ in range(200):
        excess = sum(secs_out) - target_seconds
        if excess <= 0:
            break
        i = max(range(len(secs_out)), key=lambda k: secs_out[k])
        if secs_out[i] <= 10:
            break
        secs_out[i] -= 1
    scaled = [s.model_copy(update={"target_seconds": t}) for s, t in zip(cleaned, secs_out, strict=True)]
    return Outline(sections=scaled, confident=outline.confident)


@router.reasoner()
async def write_section(
    section: dict,
    index: int,
    n_sections: int,
    prev_purpose: str,
    next_purpose: str,
    claims_text: str,
    angle_text: str,
    tone: str,
    model: str | None = None,
) -> SectionDraft:
    """One section writer.  Word budget is checked in code; one rewrite allowed."""
    sec = OutlineSection(**section)
    budget = words_for_seconds(sec.target_seconds)
    beats = "\n".join(f"  - {b}" for b in sec.beats)
    position = (
        "This is the COLD OPEN — first 20 seconds; open on a concrete image, question or "
        "number, no throat-clearing." if index == 0 else
        "This is the ENDING — pay off the hook, land the thesis, end on a line that echoes the "
        "opening image. No 'in conclusion'." if index == n_sections - 1 else
        f"This is section {index + 1} of {n_sections}."
    )
    base_user = (
        f"ANGLE\n{angle_text}\n\nSECTION {index + 1}: {sec.title}\nPurpose: {sec.purpose}\n"
        f"Beats:\n{beats}\n\nPrevious section's purpose: {prev_purpose or '(none — this is the start)'}\n"
        f"Next section's purpose: {next_purpose or '(none — this is the end)'}\n\n"
        f"CLAIMS YOU MAY USE (cite nothing else)\n{claims_text}\n\n"
        f"Write ~{budget} words of spoken narration for THIS section only."
    )
    system = (
        f"You write narration for a video essay. Tone: {tone}. {position} {STYLE_GUIDE} "
        "Do not recap the previous section; do not preview the next; just write this section "
        "so it flows from the previous purpose into the next. confident=false if the claims "
        "cannot support the beats."
    )
    draft = await router.ai(system=system, user=base_user, schema=SectionDraft, model=model, timeout=LLM_TIMEOUT)
    n = word_count(draft.narration)
    if n and abs(n - budget) / max(budget, 1) > WORD_TOLERANCE:
        direction = "cut" if n > budget else "expand"
        retry = await router.ai(
            system=system,
            user=base_user + (
                f"\n\nYOUR PREVIOUS DRAFT WAS {n} WORDS — {direction} it to about {budget} words "
                f"(hard range {int(budget * 0.85)}–{int(budget * 1.15)}). Keep the best sentences.\n\n"
                f"PREVIOUS DRAFT\n{draft.narration}"
            ),
            schema=SectionDraft,
            model=model,
            timeout=LLM_TIMEOUT,
        )
        n2 = word_count(retry.narration)
        if abs(n2 - budget) < abs(n - budget):
            draft = retry
    return draft


@router.reasoner()
async def smooth_transitions(draft_text: str, n_sections: int, tone: str, model: str | None = None) -> FullDraft:
    """Continuity pass: one essay, not N sections.  Must return the same count."""
    full = await router.ai(
        system=(
            "You are the continuity editor for a video essay. You receive the narration as "
            f"{n_sections} numbered sections. Make it read as ONE essay: fix transitions, remove "
            "repetition across sections, keep each section's facts and length (±10 %), keep the "
            "section count and order EXACTLY (return the same number of sections). Do not add "
            f"headings. Tone: {tone}. {STYLE_GUIDE} confident=false if you had to cut facts."
        ),
        user=draft_text,
        schema=FullDraft,
        model=model,
        timeout=LLM_TIMEOUT,
    )
    return full


@router.reasoner()
async def write_script(
    dossier_text: str,
    claims: list[dict],
    target_minutes: float,
    audience: str,
    tone: str,
    run_id: str,
    model: str | None = None,
) -> dict:
    """Orchestrator: angles (∥) → judge → outline → sections (∥) → smooth.

    Returns {angle, outline, sections, titles, word_count, confident}.
    """
    out_dir = run_dir(run_id, "script")
    claim_objs = [Claim(**c) for c in claims]

    angle_results = await asyncio.gather(
        *[
            router.call(f"{NODE_ID}.propose_angle", dossier_text=dossier_text, framing=f, audience=audience, tone=tone, model=model)
            for f in FRAMINGS
        ],
        return_exceptions=True,
    )
    angles = [Angle(**a) for a in angle_results if not isinstance(a, Exception)]
    if not angles:
        angles = [Angle(framing="default", thesis="The evidence tells a story worth knowing.", hook="It started smaller than anyone expected.", promise="By the end you will understand why it matters.", claim_keys=[], confident=False)]
    angles_text = "\n\n".join(
        f"[{i}] framing: {a.framing}\n  thesis: {a.thesis}\n  hook: {a.hook}\n  promise: {a.promise}\n  leans on: {', '.join(a.claim_keys) or '—'}\n  writer confident: {a.confident}"
        for i, a in enumerate(angles)
    )
    if len(angles) > 1:
        v_raw = await router.call(f"{NODE_ID}.judge_angles", angles_text=angles_text, dossier_text=dossier_text, model=model)
        verdict = AngleVerdict(**v_raw)
        idx = verdict.winner_index if 0 <= verdict.winner_index < len(angles) else 0
        if not verdict.confident:
            confident_angles = [i for i, a in enumerate(angles) if a.confident]
            idx = confident_angles[0] if confident_angles else idx
    else:
        verdict = AngleVerdict(winner_index=0, scores=[5.0], rationale="single candidate", confident=False)
        idx = 0
    angle = angles[idx]
    angle_text = f"Framing: {angle.framing}\nThesis: {angle.thesis}\nHook: {angle.hook}\nPromise: {angle.promise}"
    write_json(out_dir / "angles.json", {"candidates": [a.model_dump() for a in angles], "verdict": verdict.model_dump(), "winner": idx})

    o_raw = await router.call(f"{NODE_ID}.build_outline", dossier_text=dossier_text, angle_text=angle_text, target_minutes=target_minutes, model=model)
    outline = _normalize_outline(Outline(**o_raw), int(target_minutes * 60), claim_objs)
    write_json(out_dir / "outline.json", outline.model_dump())

    secs = outline.sections
    n = len(secs)

    def _claims_for(s: OutlineSection) -> str:
        chosen = claims_by_keys(claim_objs, s.claim_keys)
        if len(chosen) < 3:  # give the writer some room, but never the whole pile
            extra = [c for c in claim_objs if c.key not in s.claim_keys and c.strength != "weak"][: 6 - len(chosen)]
            chosen = chosen + extra
        return claims_prose(chosen)

    section_results = await asyncio.gather(
        *[
            router.call(
                f"{NODE_ID}.write_section",
                section=s.model_dump(), index=i, n_sections=n,
                prev_purpose=secs[i - 1].purpose if i > 0 else "",
                next_purpose=secs[i + 1].purpose if i < n - 1 else "",
                claims_text=_claims_for(s), angle_text=angle_text, tone=tone, model=model,
            )
            for i, s in enumerate(secs)
        ],
        return_exceptions=True,
    )
    sections: list[str] = []
    unconfident = 0
    failed_idx: list[int] = []
    for i, res in enumerate(section_results):
        if isinstance(res, Exception):
            # One retry; a transient LLM failure must not put outline meta-text into the narration.
            try:
                res = await router.call(
                    f"{NODE_ID}.write_section",
                    section=secs[i].model_dump(), index=i, n_sections=n,
                    prev_purpose=secs[i - 1].purpose if i > 0 else "",
                    next_purpose=secs[i + 1].purpose if i < n - 1 else "",
                    claims_text=_claims_for(secs[i]), angle_text=angle_text, tone=tone, model=model,
                )
            except Exception:  # noqa: BLE001
                failed_idx.append(i)
                sections.append("")
                continue
        d = SectionDraft(**res)
        if not d.narration.strip():
            failed_idx.append(i)
            sections.append("")
            continue
        sections.append(d.narration.strip())
        unconfident += 0 if d.confident else 1
    if failed_idx:
        keep = [i for i in range(n) if i not in failed_idx]
        if len(keep) < 2:
            raise RuntimeError(f"write_script: {len(failed_idx)} of {n} sections failed to write")
        secs = [secs[i] for i in keep]
        sections = [sections[i] for i in keep]
        n = len(secs)
        unconfident += len(failed_idx)

    titles = [s.title for s in secs]
    write_text(out_dir / "draft_v0_sections.md", draft_markdown(sections, titles))

    sm_raw = await router.call(f"{NODE_ID}.smooth_transitions", draft_text=draft_prose(sections, titles), n_sections=n, tone=tone, model=model)
    smoothed = FullDraft(**sm_raw)
    if len(smoothed.sections) == n and all(s.strip() for s in smoothed.sections):
        before = sum(word_count(s) for s in sections)
        after = sum(word_count(s) for s in smoothed.sections)
        if before == 0 or 0.75 <= after / before <= 1.25:  # rung 3: the pass must not gut or bloat the script
            sections = [s.strip() for s in smoothed.sections]
    write_text(out_dir / "draft_v1.md", draft_markdown(sections, titles))

    return {
        "angle": angle.model_dump(),
        "angle_verdict": verdict.model_dump(),
        "outline": outline.model_dump(),
        "sections": sections,
        "titles": titles,
        "word_count": sum(word_count(s) for s in sections),
        "confident": unconfident == 0 and outline.confident,
    }


__all__ = ["router"]
