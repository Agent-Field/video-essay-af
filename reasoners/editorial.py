"""Phase 3 — the editorial room.

Three editors read the same draft through three different lenses, in
parallel (rung 5: independent perspectives), a gate decides whether it
ships, and a revision writer applies the notes.  The loop is bounded
(``MAX_ROUNDS``); the gate is a classification that decides whether the
revise branch runs at all (dynamism rung 2).

* ``edit_facts``   — every factual sentence must map to a provided claim.
* ``edit_story``   — hook, pacing, payoff, retention.
* ``edit_voice``   — read-aloud rhythm, clarity, jargon, clichés.
* ``gate_script``  — ship / don't ship, with blocking reasons.
* ``revise_draft`` — applies the notes, keeps section count and facts.
* ``finalize_package`` — title, description, chapter titles, tags.
* ``run_editorial`` — orchestrator.
"""

from __future__ import annotations

import asyncio
import os

from agentfield import AgentRouter

from .helpers import (
    draft_markdown,
    draft_prose,
    reports_prose,
    run_dir,
    word_count,
    write_json,
    write_text,
)
from .models import EditorReport, FullDraft, Package, ShipVerdict

NODE_ID = os.getenv("AGENT_NODE_ID", "video-essay-af")
router = AgentRouter(prefix="", tags=["editorial"])

LLM_TIMEOUT = float(os.getenv("VEA_LLM_TIMEOUT", "300"))
MAX_ROUNDS = int(os.getenv("VEA_EDIT_ROUNDS", "2"))


def _editor(system: str, lens: str):
    async def _run(draft_text: str, extra: str, model: str | None) -> EditorReport:
        report = await router.ai(
            system=system + (
                " Return notes with section_index (the [n] in the draft), a short verbatim quote, "
                "the issue, severity (blocking = must fix before publishing; major = hurts the "
                "essay; minor = polish) and a concrete suggested fix. At most 8 notes, most "
                "important first. confident=false if you could not assess properly."
            ),
            user=(extra + "\n\n" if extra else "") + f"DRAFT\n{draft_text}",
            schema=EditorReport,
            model=model,
            timeout=LLM_TIMEOUT,
        )
        return report.model_copy(update={"lens": lens})

    return _run


_facts = _editor(
    "You are the fact editor of a documentary video essay. Check EVERY factual sentence "
    "of the draft against the CLAIMS list: flag sentences that state facts, numbers, "
    "names, dates or quotes that no claim supports (blocking), that distort a claim "
    "(blocking/major), that overstate a weak or contested claim as certain (major), or that "
    "could mislead by omission (minor). Do not judge style.",
    "fact",
)
_story = _editor(
    "You are the story editor. Judge the draft as a viewer would: does the cold open earn "
    "15 seconds? Is there a clear question and a real payoff? Does every section move the "
    "argument (flag sections that stall or repeat)? Is the turn/complication real? Does the "
    "ending echo the opening without summarising? Flag pacing problems and missed "
    "opportunities from the angle.",
    "story",
)
_voice = _editor(
    "You are the line editor for spoken narration. Flag sentences that are hard to say "
    "aloud, jargon without a gloss, clichés, filler, tonal slips, run-ons, accidental "
    "rhymes, and written-English constructions that don't work in speech. Suggest the "
    "spoken rewrite.",
    "voice",
)


@router.reasoner()
async def edit_facts(draft_text: str, claims_text: str, model: str | None = None) -> EditorReport:
    return await _facts(draft_text, f"CLAIMS (the only allowed sources)\n{claims_text}", model)


@router.reasoner()
async def edit_story(draft_text: str, angle_text: str, model: str | None = None) -> EditorReport:
    return await _story(draft_text, f"THE ANGLE THE ESSAY PROMISED\n{angle_text}", model)


@router.reasoner()
async def edit_voice(draft_text: str, tone: str, model: str | None = None) -> EditorReport:
    return await _voice(draft_text, f"INTENDED TONE: {tone}", model)


@router.reasoner()
async def gate_script(draft_text: str, reports_text: str, model: str | None = None) -> ShipVerdict:
    """Ship / don't ship.  Blocking fact problems always block."""
    return await router.ai(
        system=(
            "You are the executive producer. Given the draft and the three editors' reports, "
            "decide whether the script is ready to record. It ships only if there are NO "
            "blocking notes and the major notes are tolerable for a first cut. List the "
            "blocking reasons if not. confident=false if the reports contradict each other badly."
        ),
        user=f"EDITOR REPORTS\n{reports_text}\n\nDRAFT\n{draft_text}",
        schema=ShipVerdict,
        model=model,
        timeout=LLM_TIMEOUT,
    )


@router.reasoner()
async def revise_draft(draft_text: str, notes_text: str, claims_text: str, n_sections: int, tone: str, model: str | None = None) -> FullDraft:
    """Apply the editors' notes.  Same section count; no new facts."""
    return await router.ai(
        system=(
            "You are the revising writer. Apply the editors' notes to the draft: fix every "
            "blocking and major note, take the minor ones when cheap. Keep the section count "
            f"and order EXACTLY ({n_sections} sections, returned in order), keep the length "
            "within ±10 %, and introduce NO facts that are not in the claims list — if a fact "
            "must go, replace it with something the claims support, make it openly hypothetical "
            "('Picture…'), or cut it. Do NOT add new names, numbers, dates, quotes or anecdotes "
            f"that are not in the claims; every revision must be strictly safer, never riskier. Tone: {tone}. "
            "Spoken narration only, no headings. confident=false if notes conflicted."
        ),
        user=f"NOTES\n{notes_text}\n\nCLAIMS\n{claims_text}\n\nDRAFT\n{draft_text}",
        schema=FullDraft,
        model=model,
        timeout=LLM_TIMEOUT,
    )


@router.reasoner()
async def finalize_package(final_text: str, titles: list[str], topic: str, model: str | None = None) -> Package:
    """Title, description, chapter titles, tags for publishing."""
    pkg = await router.ai(
        system=(
            "You package a finished video essay for YouTube. Write a title (≤ 70 characters, "
            "specific, no clickbait lies), a 2–4 sentence description that states the thesis and "
            "invites the viewer, one short chapter title per section (same count and order as the "
            "sections), and 5–10 tags. confident=false if the script is unclear."
        ),
        user=f"TOPIC: {topic}\nSECTION TITLES (draft): {titles}\n\nFINAL SCRIPT\n{final_text}",
        schema=Package,
        model=model,
        timeout=LLM_TIMEOUT,
    )
    if len(pkg.chapter_titles) != len(titles):
        pkg = pkg.model_copy(update={"chapter_titles": list(titles)})
    return pkg


@router.reasoner()
async def run_editorial(
    sections: list[str],
    titles: list[str],
    claims_text: str,
    angle_text: str,
    tone: str,
    run_id: str,
    model: str | None = None,
) -> dict:
    """Orchestrator: ≤ MAX_ROUNDS × (3 editors ∥ → gate → revise).

    Returns {sections, rounds, shipped, reports, blocking_left, word_count, confident}.
    """
    out_dir = run_dir(run_id, "script")
    draft = [s.strip() for s in sections]
    n = len(draft)
    history: list[dict] = []
    shipped = False
    blocking_left: list[str] = []

    for round_idx in range(MAX_ROUNDS):
        draft_text = draft_prose(draft, titles)
        reports_raw = await asyncio.gather(
            router.call(f"{NODE_ID}.edit_facts", draft_text=draft_text, claims_text=claims_text, model=model),
            router.call(f"{NODE_ID}.edit_story", draft_text=draft_text, angle_text=angle_text, model=model),
            router.call(f"{NODE_ID}.edit_voice", draft_text=draft_text, tone=tone, model=model),
            return_exceptions=True,
        )
        reports = [EditorReport(**r) for r in reports_raw if not isinstance(r, Exception)]
        editor_errors = [f"{type(r).__name__}: {r}"[:200] for r in reports_raw if isinstance(r, Exception)]
        notes_text = reports_prose(reports)
        blocking = [nt for r in reports for nt in r.notes if nt.severity == "blocking"]

        g_raw = await router.call(f"{NODE_ID}.gate_script", draft_text=draft_text, reports_text=notes_text, model=model)
        gate = ShipVerdict(**g_raw)
        history.append({
            "round": round_idx + 1,
            "reports": [r.model_dump() for r in reports],
            "editor_errors": editor_errors,
            "gate": gate.model_dump(),
            "blocking_count": len(blocking),
        })

        # Ship only on a complete, clean round: all three editors reported and none blocked.
        if gate.ship and not blocking and not editor_errors:
            shipped = True
            break
        if not reports:
            break  # editors all failed — nothing actionable, ship what we have

        # Last round: revise anyway so blocking notes get one fix attempt.
        r_raw = await router.call(
            f"{NODE_ID}.revise_draft", draft_text=draft_text, notes_text=notes_text,
            claims_text=claims_text, n_sections=n, tone=tone, model=model,
        )
        revised = FullDraft(**r_raw)
        if len(revised.sections) == n and all(s.strip() for s in revised.sections):
            before = sum(word_count(s) for s in draft)
            after = sum(word_count(s) for s in revised.sections)
            if before == 0 or 0.7 <= after / before <= 1.3:
                draft = [s.strip() for s in revised.sections]
        write_text(out_dir / f"draft_v{round_idx + 2}.md", draft_markdown(draft, titles))
        blocking_left = [f"§{b.section_index}: {b.issue}" for b in blocking]

    if not shipped and history:
        # Final scrub (bounded: one revise + one fact check): remove or generalise the
        # remaining unsupported specifics, then re-check facts so the report is honest.
        draft_text = draft_prose(draft, titles)
        f_raw = await router.call(f"{NODE_ID}.edit_facts", draft_text=draft_text, claims_text=claims_text, model=model)
        facts = EditorReport(**f_raw)
        blocking_notes = [nt for nt in facts.notes if nt.severity == "blocking"]
        if blocking_notes:
            scrub_notes = "FACT EDITOR — BLOCKING ONLY. For each note, REMOVE the unsupported specific or make it "
            scrub_notes += "openly hypothetical; add nothing new.\n" + "\n".join(
                f"  - [blocking] §{nt.section_index} “{nt.quote[:90]}” — {nt.issue} → fix: {nt.suggested_fix}" for nt in blocking_notes
            )
            r_raw = await router.call(
                f"{NODE_ID}.revise_draft", draft_text=draft_text, notes_text=scrub_notes,
                claims_text=claims_text, n_sections=n, tone=tone, model=model,
            )
            revised = FullDraft(**r_raw)
            if len(revised.sections) == n and all(s.strip() for s in revised.sections):
                before = sum(word_count(s) for s in draft)
                after = sum(word_count(s) for s in revised.sections)
                if before == 0 or 0.6 <= after / before <= 1.3:
                    draft = [s.strip() for s in revised.sections]
            draft_text = draft_prose(draft, titles)
            f_raw = await router.call(f"{NODE_ID}.edit_facts", draft_text=draft_text, claims_text=claims_text, model=model)
            facts = EditorReport(**f_raw)
            history.append({"round": "scrub", "reports": [facts.model_dump()], "editor_errors": [], "gate": None,
                            "blocking_count": sum(1 for nt in facts.notes if nt.severity == "blocking")})
        blocking_left = [f"§{nt.section_index}: {nt.issue}" for nt in facts.notes if nt.severity == "blocking"]
        shipped = not blocking_left

    write_text(out_dir / "final.md", draft_markdown(draft, titles))
    write_json(out_dir / "editorial.json", {"history": history, "shipped": shipped, "blocking_left": blocking_left})
    return {
        "sections": draft,
        "titles": titles,
        "rounds": sum(1 for h in history if h["round"] != "scrub"),
        "scrubbed": any(h["round"] == "scrub" for h in history),
        "shipped": shipped,
        "blocking_left": blocking_left,
        "reports": history,
        "word_count": sum(word_count(s) for s in draft),
        "confident": shipped,
    }


__all__ = ["router"]
