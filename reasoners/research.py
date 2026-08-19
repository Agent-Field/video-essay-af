"""Phase 1 — multi-agent research.

Cognitive jobs (one reasoner each):

* ``plan_research``     — the research lead decomposes the brief into 3–6 lanes
                           and *writes each researcher's assignment* at runtime
                           (dynamism rung 4: meta-prompted children).
* ``research_lane``     — one researcher: a tool-using loop (web_search /
                           fetch_page) followed by ``extract_claims``.
* ``extract_claims``    — turns research notes into citable claims (schema).
* ``verify_claim``      — adversarial refuter for load-bearing claims
                           (verification rung 6 where being wrong is expensive).
* ``find_gaps``         — decides whether one more bounded round is needed
                           (dynamism rung 3: runtime fan-out width, cap 2).
* ``synthesize_dossier``— the research dossier every later phase reads.
* ``run_research``      — the orchestrator: plan → gather lanes → dedupe/key
                           (code) → verify top-K (gather) → gap round → dossier.

Rung-3 invariants enforced in code: a claim's source URL must have actually
been seen by that lane's tool calls (else it is demoted to ``weak``); claim
keys are assigned deterministically; every verified claim maps to a verdict.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from agentfield import AgentRouter, ToolCallConfig

from .helpers import (
    apply_verdicts,
    assign_claim_keys,
    claims_prose,
    fallback_dossier,
    known_keys,
    load_bearing_claims,
    normalize_url,
    run_dir,
    write_json,
)
from .models import (
    Claim,
    ClaimVerdict,
    Dossier,
    GapReport,
    LaneFindings,
    ResearchLane,
    ResearchPlan,
)

NODE_ID = os.getenv("AGENT_NODE_ID", "video-essay-af")
router = AgentRouter(prefix="", tags=["research"])

# ───── Budgets (the safety envelope) ─────────────────────────────────
MAX_LANES = int(os.getenv("VEA_MAX_LANES", "6"))
MIN_LANES = 3
LANE_MAX_TURNS = int(os.getenv("VEA_LANE_MAX_TURNS", "10"))
LANE_MAX_TOOL_CALLS = int(os.getenv("VEA_LANE_MAX_TOOL_CALLS", "12"))
VERIFY_TOP_K = int(os.getenv("VEA_VERIFY_CLAIMS", "6"))
VERIFY_MAX_TURNS = 5
VERIFY_MAX_TOOL_CALLS = 6
MAX_GAP_LANES = 2
MAX_CLAIMS_TOTAL = 40
LLM_TIMEOUT = float(os.getenv("VEA_LLM_TIMEOUT", "300"))


def _tool_config(max_turns: int, max_tool_calls: int) -> ToolCallConfig:
    return ToolCallConfig(
        agent_ids=[NODE_ID],
        tags=["research-tool"],
        max_turns=max_turns,
        max_tool_calls=max_tool_calls,
    )


def _urls_from_trace(resp: Any) -> set[str]:
    """Every URL a tool loop actually touched (search results + fetched pages)."""
    seen: set[str] = set()
    trace = getattr(resp, "trace", None)
    if trace is None:
        return seen
    for rec in getattr(trace, "calls", []) or []:
        res = rec.result
        if isinstance(res, dict):
            if isinstance(res.get("url"), str):
                seen.add(normalize_url(res["url"]))
            for item in res.get("results", []) or []:
                if isinstance(item, dict) and isinstance(item.get("url"), str):
                    seen.add(normalize_url(item["url"]))
        args = rec.arguments or {}
        if isinstance(args, dict) and isinstance(args.get("url"), str):
            seen.add(normalize_url(args["url"]))
    return seen


_LEAK_MARKERS = ("<｜DSML｜", "<|DSML|", "<tool_call>", "<function_call", "</invoke>", "<｜tool")


def _text_of(resp: Any) -> str:
    """The model's final prose.  Empty when the loop ended on a budget and the
    'answer' is really leaked tool-call markup (seen with DeepSeek when
    max_tool_calls is exhausted mid-thought)."""
    text = getattr(resp, "text", None)
    if not (isinstance(text, str) and text.strip()):
        try:
            text = str(resp.choices[0].message.content or "")
        except Exception:  # noqa: BLE001
            text = str(resp or "")
    if any(m in text for m in _LEAK_MARKERS):
        return ""
    return text


def _trace_digest(resp: Any, *, max_page_chars: int = 3000, max_total_chars: int = 26000) -> str:
    """Ground truth of what the loop actually read: search results + page text,
    in call order, each item carrying its URL.  Code-built, so it never
    hallucinates — the claim extractor always gets this alongside the notes."""
    trace = getattr(resp, "trace", None)
    if trace is None:
        return ""
    chunks: list[str] = []
    total = 0
    for rec in getattr(trace, "calls", []) or []:
        res = rec.result
        args = rec.arguments or {}
        block = ""
        if isinstance(res, dict) and isinstance(res.get("results"), list):
            lines = [f"SEARCH: {args.get('query', '')!s}"]
            for item in res["results"][:8]:
                if isinstance(item, dict):
                    lines.append(f"  - {item.get('title', '')} — {item.get('url', '')}\n    {str(item.get('snippet', ''))[:240]}")
            block = "\n".join(lines)
        elif isinstance(res, dict) and isinstance(res.get("text"), str) and res.get("text").strip():
            block = (
                f"PAGE: {res.get('title', '')} — {res.get('url', args.get('url', ''))}\n"
                f"{res['text'][:max_page_chars]}"
            )
        elif isinstance(res, dict) and res.get("error"):
            block = f"PAGE FAILED: {args.get('url', '')} — {res.get('error')}"
        if not block:
            continue
        if total + len(block) > max_total_chars:
            break
        chunks.append(block)
        total += len(block)
    return "\n\n".join(chunks)


# ───── Reasoners ─────────────────────────────────────────────────────


@router.reasoner()
async def plan_research(brief: str, model: str | None = None) -> ResearchPlan:
    """Research lead: split the brief into independent lanes and write each
    researcher's assignment.  The number of lanes is decided here, by the brief."""
    plan = await router.ai(
        system=(
            "You are the research lead for a documentary video essay. Decompose the "
            "brief into 3–6 INDEPENDENT research lanes a small team can run in parallel. "
            "Typical lanes: origins/history; mechanism/how-it-works; key figures & "
            "turning points; numbers & data; controversy/counterarguments/myths; "
            "present-day consequences. Pick only lanes the brief actually needs. For each "
            "lane write a 2–4 sentence assignment (what to find, what 'good' looks like, "
            "what to ignore) and 2–4 concrete seed search queries with names, dates, terms. "
            "Set confident=false if the brief is too vague to plan."
        ),
        user=f"BRIEF\n{brief}",
        schema=ResearchPlan,
        model=model,
        timeout=LLM_TIMEOUT,
    )
    lanes = plan.lanes[:MAX_LANES]
    if len(lanes) < MIN_LANES or not plan.confident:
        # Fallback: a standard three-lane spread still yields a usable dossier.
        topic = brief.splitlines()[0].replace("Topic:", "").strip() if brief else "the topic"
        base = [
            ResearchLane(name="origins & key figures", brief=f"Find the origin story of {topic}: who, when, where, and the first turning points. Prefer primary/secondary sources with dates.", queries=[f"{topic} history origin", f"who invented {topic}", f"{topic} timeline"]),
            ResearchLane(name="mechanism & how it works", brief=f"Explain how {topic} actually works or why it happened, concretely enough for a narrator to describe in plain language.", queries=[f"how does {topic} work", f"{topic} explained", f"{topic} mechanism"]),
            ResearchLane(name="numbers, impact & controversy", brief=f"Gather specific numbers, scale, consequences, and any myths or contested claims about {topic}.", queries=[f"{topic} statistics", f"{topic} impact", f"{topic} myth debunked"]),
        ]
        lanes = (lanes + [b for b in base if b.name not in {ln.name for ln in lanes}])[:max(MIN_LANES, len(lanes))]
        return ResearchPlan(lanes=lanes, confident=False)
    return ResearchPlan(lanes=lanes, confident=True)


@router.reasoner()
async def extract_claims(lane_name: str, notes: str, brief: str, model: str | None = None) -> LaneFindings:
    """Turn a researcher's notes into discrete, citable claims."""
    findings = await router.ai(
        system=(
            "You convert research notes into discrete, citable claims for a fact-checked "
            "video essay. Each claim: ONE self-contained sentence a narrator could say, with "
            "the exact source URL it came from (only URLs that appear in the notes), the "
            "source title, a short supporting quote/paraphrase (evidence), a kind "
            "(fact/number/anecdote/quote/counterpoint/context) and a strength (strong = "
            "stated plainly by a reputable source; moderate = plausible/secondary; weak = "
            "unsourced or inferred). Prefer specific numbers, names, dates and surprising "
            "facts; skip generic filler. 6–14 claims. Write 'notes' as an 80–200 word prose "
            "summary of what was learned. confident=false if the notes are thin."
        ),
        user=f"BRIEF (for relevance)\n{brief}\n\nLANE: {lane_name}\n\nRESEARCH NOTES\n{notes}",
        schema=LaneFindings,
        model=model,
        timeout=LLM_TIMEOUT,
    )
    return findings


@router.reasoner()
async def research_lane(lane: dict, brief: str, model: str | None = None) -> dict:
    """One researcher: search → read → take notes (tool loop), then extract claims.

    Returns LaneFindings as a dict, plus the URLs the loop actually touched.
    """
    lane_obj = ResearchLane(**lane)
    queries = "\n".join(f"  - {q}" for q in lane_obj.queries)
    resp = await router.ai(
        system=(
            "You are a meticulous documentary researcher with two tools: web_search and "
            "fetch_page. Work the assignment: run 2–4 searches (vary phrasing; include "
            "names/dates/numbers), then fetch_page the 2–4 most promising sources and READ "
            "them. Prefer primary sources, reputable outlets, encyclopedias and academic "
            f"pages. HARD BUDGET: at most {LANE_MAX_TOOL_CALLS} tool calls in total — after "
            f"about {max(3, LANE_MAX_TOOL_CALLS - 3)} calls STOP calling tools and write your "
            "notes. FINAL ANSWER (plain text, no tool calls): well-organised research notes "
            "(bullets ok) where EVERY fact carries the URL it came from in parentheses, with "
            "short verbatim quotes for numbers and surprising claims, plus a 'dead ends' line. "
            "Never invent URLs; only cite pages you actually searched or fetched."
        ),
        user=(
            f"BRIEF\n{brief}\n\nYOUR LANE: {lane_obj.name}\nASSIGNMENT: {lane_obj.brief}\n"
            f"Seed queries:\n{queries}\n\nBegin. Use the tools, then write the notes."
        ),
        tools=_tool_config(LANE_MAX_TURNS, LANE_MAX_TOOL_CALLS),
        model=model,
        timeout=LLM_TIMEOUT,
    )
    notes = _text_of(resp)
    seen_urls = _urls_from_trace(resp)
    trace = getattr(resp, "trace", None)
    n_tool_calls = getattr(trace, "total_tool_calls", 0) if trace else 0
    digest = _trace_digest(resp)
    if not notes.strip() and not digest.strip():
        notes = "(the researcher produced no notes and no tool results)"
    material = notes.strip()
    if digest:
        material += ("\n\n" if material else "") + "RAW TOOL RESULTS (ground truth — URLs are exact)\n" + digest

    f_raw = await router.call(
        f"{NODE_ID}.extract_claims", lane_name=lane_obj.name, notes=material, brief=brief, model=model
    )
    findings = LaneFindings(**f_raw)
    if not findings.claims and len(material) > 1500:
        # One retry with a firmer instruction: there is material, so there are claims.
        f_raw = await router.call(
            f"{NODE_ID}.extract_claims", lane_name=lane_obj.name,
            notes=material + "\n\nNOTE TO EXTRACTOR: a previous pass returned zero claims. The material above "
            "contains concrete facts with URLs — return at least 4 claims; when unsure which URL, use the "
            "most relevant PAGE/SEARCH URL listed above.",
            brief=brief, model=model,
        )
        findings = LaneFindings(**f_raw)

    # Rung 3 — a claim may only cite a URL this lane actually touched.
    checked: list[Claim] = []
    for c in findings.claims:
        if not c.source_url.strip():
            continue
        if seen_urls and normalize_url(c.source_url) not in seen_urls:
            c = c.model_copy(update={"strength": "weak"})
        elif not seen_urls:
            c = c.model_copy(update={"strength": "weak"})
        checked.append(c)
    findings = findings.model_copy(update={"claims": checked})
    return {
        "findings": findings.model_dump(),
        "seen_urls": sorted(seen_urls),
        "tool_calls": n_tool_calls,
        "notes": notes,
    }


@router.reasoner()
async def verify_claim(claim: dict, model: str | None = None) -> ClaimVerdict:
    """Adversarial refuter: try to break one load-bearing claim with fresh evidence."""
    c = Claim(**claim)
    resp = await router.ai(
        system=(
            "You are a sceptical fact-checker. Your ONLY job is to try to REFUTE the claim "
            f"below using web_search and fetch_page (1–3 searches, read 1–2 pages; hard budget "
            f"{VERIFY_MAX_TOOL_CALLS} tool calls, then STOP and write). Look for contradicting "
            "numbers, misattributions, outdated figures, or the claim being a popular myth. "
            "FINAL ANSWER (plain text): what you found, with URLs, and whether the original "
            "source supports the claim as stated."
        ),
        user=f"CLAIM [{c.key}]: {c.text}\nOriginal source: {c.source_url}\nEvidence offered: {c.evidence or '(none)'}",
        tools=_tool_config(VERIFY_MAX_TURNS, VERIFY_MAX_TOOL_CALLS),
        model=model,
        timeout=LLM_TIMEOUT,
    )
    evidence = _text_of(resp)
    digest = _trace_digest(resp, max_page_chars=1500, max_total_chars=9000)
    if digest:
        evidence = (evidence.strip() + "\n\nRAW TOOL RESULTS\n" + digest).strip()
    if not evidence.strip():
        evidence = "(the fact-checker found nothing: no notes, no tool results)"
    verdict = await router.ai(
        system=(
            "Given a claim and a fact-checker's notes, decide: supported (stands as written), "
            "contested (credible sources disagree or nuance changes the meaning), unsupported "
            "(evidence contradicts it or the source does not say it), or unverifiable (could "
            "not find anything either way). If a small correction would make it right, give "
            "corrected_text. confident=false if the notes are inconclusive."
        ),
        user=f"CLAIM [{c.key}]: {c.text}\n\nFACT-CHECKER NOTES\n{evidence}",
        schema=ClaimVerdict,
        model=model,
        timeout=LLM_TIMEOUT,
    )
    verdict = verdict.model_copy(update={"key": c.key})
    if not verdict.confident and verdict.verdict == "unsupported":
        # Never drop a claim on an unconfident refutation — downgrade instead.
        verdict = verdict.model_copy(update={"verdict": "contested"})
    return verdict


@router.reasoner()
async def find_gaps(brief: str, claims_text: str, lanes_run: list[str], model: str | None = None) -> GapReport:
    """Compare what the research questions asked against what was found."""
    report = await router.ai(
        system=(
            "You audit research coverage for a video essay. Compare the brief's research "
            "questions against the claims gathered. Identify the 1–2 most important holes that "
            "would embarrass the essay if left open (a missing mechanism, no numbers, no human "
            "story, no counterpoint). Only set needs_more=true if a hole is genuinely important; "
            "for each, write a lane assignment and seed queries. Do not repeat lanes already run."
        ),
        user=f"BRIEF\n{brief}\n\nLANES ALREADY RUN: {', '.join(lanes_run)}\n\nCLAIMS SO FAR\n{claims_text}",
        schema=GapReport,
        model=model,
        timeout=LLM_TIMEOUT,
    )
    return report.model_copy(update={"gaps": report.gaps[:MAX_GAP_LANES]})


@router.reasoner()
async def synthesize_dossier(brief: str, claims_text: str, caveats: list[str], model: str | None = None) -> Dossier:
    """Write the research dossier downstream writers will read."""
    cav = "\n".join(f"  - {c}" for c in caveats) or "  - (none)"
    dossier = await router.ai(
        system=(
            "You are the senior researcher writing the dossier a scriptwriter will work from. "
            "Summarise what is known (150–300 words), list 2–5 storylines worth telling "
            "(each one sentence, concrete), pick the citation keys of the load-bearing claims "
            "(8–15), and list tensions/caveats — contested points and things the evidence does "
            "NOT support. Only reference claim keys that exist. confident=false if the claims "
            "are too thin for a 3+ minute essay."
        ),
        user=f"BRIEF\n{brief}\n\nVERIFICATION CAVEATS\n{cav}\n\nCLAIMS\n{claims_text}",
        schema=Dossier,
        model=model,
        timeout=LLM_TIMEOUT,
    )
    return dossier


@router.reasoner()
async def run_research(brief: str, run_id: str, model: str | None = None) -> dict:
    """Orchestrator for the whole research phase.  Returns a dict:

    {dossier, claims, verdicts, caveats, lanes_run, tool_calls, confident}
    """
    out_dir = run_dir(run_id, "research")

    plan_raw = await router.call(f"{NODE_ID}.plan_research", brief=brief, model=model)
    plan = ResearchPlan(**plan_raw)
    write_json(out_dir / "plan.json", plan.model_dump())

    lane_results = await asyncio.gather(
        *[
            router.call(f"{NODE_ID}.research_lane", lane=lane.model_dump(), brief=brief, model=model)
            for lane in plan.lanes
        ],
        return_exceptions=True,
    )
    findings: list[LaneFindings] = []
    tool_calls = 0
    lane_errors: list[str] = []
    for lane, res in zip(plan.lanes, lane_results, strict=True):
        if isinstance(res, Exception):
            lane_errors.append(f"{lane.name}: {res!s}"[:300])
            continue
        findings.append(LaneFindings(**res["findings"]))
        tool_calls += int(res.get("tool_calls", 0))

    raw_claims = [c for f in findings for c in f.claims]
    claims = assign_claim_keys(raw_claims)
    claims = load_bearing_claims(claims, MAX_CLAIMS_TOTAL) if len(claims) > MAX_CLAIMS_TOTAL else claims
    claims = sorted(claims, key=lambda c: c.key)

    # Rung 6 — adversarial verification of the claims most likely to carry the essay.
    to_verify = load_bearing_claims(claims, VERIFY_TOP_K)
    verdict_results = await asyncio.gather(
        *[router.call(f"{NODE_ID}.verify_claim", claim=c.model_dump(), model=model) for c in to_verify],
        return_exceptions=True,
    )
    verdicts: list[ClaimVerdict] = [ClaimVerdict(**v) for v in verdict_results if not isinstance(v, Exception)]
    claims, caveats = apply_verdicts(claims, verdicts)

    # One bounded gap round (dynamism rung 3, cap MAX_GAP_LANES).
    lanes_run = [ln.name for ln in plan.lanes]
    gap_report: GapReport | None = None
    if claims:
        gap_raw = await router.call(
            f"{NODE_ID}.find_gaps", brief=brief, claims_text=claims_prose(claims, with_evidence=False),
            lanes_run=lanes_run, model=model,
        )
        gap_report = GapReport(**gap_raw)
        if gap_report.needs_more and gap_report.gaps and gap_report.confident:
            extra = await asyncio.gather(
                *[
                    router.call(f"{NODE_ID}.research_lane", lane=g.model_dump(), brief=brief, model=model)
                    for g in gap_report.gaps[:MAX_GAP_LANES]
                ],
                return_exceptions=True,
            )
            new_claims: list[Claim] = []
            for g, res in zip(gap_report.gaps[:MAX_GAP_LANES], extra, strict=True):
                if isinstance(res, Exception):
                    lane_errors.append(f"{g.name}: {res!s}"[:300])
                    continue
                lanes_run.append(g.name)
                f = LaneFindings(**res["findings"])
                findings.append(f)
                new_claims.extend(f.claims)
                tool_calls += int(res.get("tool_calls", 0))
            if new_claims:
                # Keys are sparse after verdict drops / the cap — continue from the highest in use.
                start = max((int(c.key[1:]) for c in claims if c.key[1:].isdigit()), default=0) + 1
                fresh = assign_claim_keys(new_claims, start=start)
                existing = known_keys(claims)
                claims = claims + [c for c in fresh if c.key not in existing]

    if not claims:
        dossier = fallback_dossier([], "no claims gathered — check research tools / network")
    else:
        d_raw = await router.call(
            f"{NODE_ID}.synthesize_dossier", brief=brief, claims_text=claims_prose(claims),
            caveats=caveats, model=model,
        )
        dossier = Dossier(**d_raw)
        valid = known_keys(claims)
        keep = [k for k in dossier.key_claim_keys if k in valid]
        if not keep:
            keep = [c.key for c in load_bearing_claims(claims, 12)]
        dossier = dossier.model_copy(update={"key_claim_keys": keep})
        if not dossier.confident and len(claims) < 6:
            dossier = fallback_dossier(claims, "synthesizer not confident and claims are thin")

    result = {
        "dossier": dossier.model_dump(),
        "claims": [c.model_dump() for c in claims],
        "verdicts": [v.model_dump() for v in verdicts],
        "caveats": caveats,
        "lanes_run": lanes_run,
        "lane_errors": lane_errors,
        "gap_report": gap_report.model_dump() if gap_report else None,
        "tool_calls": tool_calls,
        "confident": bool(claims) and dossier.confident,
    }
    write_json(out_dir / "claims.json", result["claims"])
    write_json(out_dir / "verdicts.json", result["verdicts"])
    write_json(out_dir / "dossier.json", result["dossier"])
    write_json(out_dir / "findings.json", [f.model_dump() for f in findings])
    return result


__all__ = ["router"]
