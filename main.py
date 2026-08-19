"""video-essay-af — prompt → researched, edited, narrated, stock-footage video essay.

One agent node, ~30 reasoners, five routers:

    create_video_essay (entry)
      ├─ interpret_brief                       (ai)
      ├─ run_research                          research router
      │    ├─ plan_research                    (ai, writes lane briefs at runtime)
      │    ├─ research_lane ×N (∥)             (tool loop: web_search / fetch_page)
      │    │    └─ extract_claims              (ai)
      │    ├─ verify_claim ×K (∥)              (adversarial tool loop + verdict)
      │    ├─ find_gaps → research_lane ×≤2    (bounded second round)
      │    └─ synthesize_dossier               (ai)
      ├─ write_script                          script router
      │    ├─ propose_angle ×3 (∥) → judge_angles
      │    ├─ build_outline
      │    ├─ write_section ×S (∥)            (word budget checked in code)
      │    └─ smooth_transitions
      ├─ run_editorial                         editorial router
      │    └─ ≤2 × [edit_facts ∥ edit_story ∥ edit_voice → gate_script → revise_draft]
      ├─ finalize_package
      └─ produce_video                         production router
           ├─ direct_visuals                   (N shots decided by the script)
           ├─ synthesize_narration ∥ source_shot_footage ×N
           │                                    └─ pick_footage
           └─ assemble_video                   (app.harness → AForge; ffprobe membrane; python fallback)

Every cross-reasoner call goes through ``app.call`` so the control plane records
the whole DAG and the verifiable-credential chain.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from agentfield import Agent, AIConfig  # noqa: E402

from reasoners import (  # noqa: E402
    editorial_router,
    production_router,
    research_router,
    script_router,
    tools_router,
)
from reasoners.helpers import (  # noqa: E402
    brief_prose,
    claims_prose,
    dossier_prose,
    fallback_brief,
    new_run_id,
    run_dir,
    slug,
    write_json,
)
from reasoners.models import Brief, Claim, Dossier, Package  # noqa: E402

NODE_ID = os.getenv("AGENT_NODE_ID", "video-essay-af")
DEFAULT_MODEL = os.getenv("AI_MODEL", "openrouter/deepseek/deepseek-v4-flash")

app = Agent(
    node_id=NODE_ID,
    agentfield_server=os.getenv("AGENTFIELD_SERVER", "http://localhost:8080"),
    version="1.0.0",
    description="Prompt → multi-agent research → script → editorial → narration + free stock footage → final.mp4",
    ai_config=AIConfig(model=DEFAULT_MODEL),
    dev_mode=True,
)
for r in (tools_router, research_router, script_router, editorial_router, production_router):
    app.include_router(r)

LLM_TIMEOUT = float(os.getenv("VEA_LLM_TIMEOUT", "300"))
MIN_MINUTES, MAX_MINUTES = 1.0, 12.0


# ───── Phase 0 — the brief ───────────────────────────────────────────


@app.reasoner()
async def interpret_brief(prompt: str, target_minutes: float, style: str, model: str | None = None) -> Brief:
    """Read the prompt the way a showrunner would: what is this essay about, for whom, in what register?"""
    brief = await app.ai(
        system=(
            "You are the showrunner of a documentary video-essay channel. Turn the user's prompt "
            "into a production brief: the topic as a noun phrase, the ONE central question the "
            "essay must answer, the audience and what they already know, the narrator's tone "
            f"(default to '{style}' unless the prompt implies otherwise), the target runtime, "
            "3–8 concrete research questions (specific enough to search), and things to avoid "
            "(clichés, angles the prompt rules out). confident=false if the prompt is too vague "
            "to plan an essay."
        ),
        user=f"PROMPT: {prompt}\nRequested runtime: {target_minutes:g} minutes\nRequested style: {style}",
        schema=Brief,
        model=model,
        timeout=LLM_TIMEOUT,
    )
    brief = brief.model_copy(update={"target_minutes": target_minutes})
    fb = fallback_brief(prompt, target_minutes, style)
    if not brief.confident:
        return fb.model_copy(update={"topic": brief.topic or fb.topic, "central_question": brief.central_question or fb.central_question})
    if len(brief.research_questions) < 3:
        # Keep the model's audience/tone/avoid-list; only top up the thin question list.
        extra = [q for q in fb.research_questions if q not in brief.research_questions]
        return brief.model_copy(update={"research_questions": (brief.research_questions + extra)[:6]})
    return brief


# ───── Entry ─────────────────────────────────────────────────────────


@app.reasoner(tags=["entry"])
async def create_video_essay(
    prompt: str,
    target_minutes: float = 4.0,
    style: str = "curious explainer",
    captions: bool = True,
    voice: str | None = None,
    model: str | None = None,
) -> dict:
    """Prompt → final.mp4.  Returns paths, the script, the package and a phase-by-phase report.

    ``model`` overrides the reasoning model for every call in this run
    (e.g. ``openrouter/deepseek/deepseek-v4-pro-0813``); the harness model is
    ``VEA_HARNESS_MODEL`` / AForge's own default.
    """
    t0 = time.monotonic()
    target_minutes = float(max(MIN_MINUTES, min(MAX_MINUTES, target_minutes)))
    run_id = new_run_id(slug(prompt, 28))
    out = run_dir(run_id)
    phases: dict[str, float] = {}

    # Phase 0 — brief
    b_raw = await app.call(f"{app.node_id}.interpret_brief", prompt=prompt, target_minutes=target_minutes, style=style, model=model)
    brief = Brief(**b_raw)
    b_text = brief_prose(brief)
    write_json(out / "brief.json", brief.model_dump())
    phases["brief"] = round(time.monotonic() - t0, 1)

    # Phase 1 — research (parallel lanes, verification, gap round, dossier)
    research = await app.call(f"{app.node_id}.run_research", brief=b_text, run_id=run_id, model=model)
    claims = [Claim(**c) for c in research["claims"]]
    dossier = Dossier(**research["dossier"])
    d_text = dossier_prose(dossier, claims)
    c_text = claims_prose(claims)
    phases["research"] = round(time.monotonic() - t0, 1)

    # Phase 2 — script (angles → judge → outline → sections → smooth)
    script = await app.call(
        f"{app.node_id}.write_script", dossier_text=d_text, claims=[c.model_dump() for c in claims],
        target_minutes=target_minutes, audience=brief.audience, tone=brief.tone, run_id=run_id, model=model,
    )
    angle = script["angle"]
    angle_text = f"Thesis: {angle['thesis']}\nHook: {angle['hook']}\nPromise: {angle['promise']}"
    phases["script"] = round(time.monotonic() - t0, 1)

    # Phase 3 — editorial room (≤2 rounds) → final draft
    edited = await app.call(
        f"{app.node_id}.run_editorial", sections=script["sections"], titles=script["titles"],
        claims_text=c_text, angle_text=angle_text, tone=brief.tone, run_id=run_id, model=model,
    )
    final_sections: list[str] = edited["sections"]
    titles: list[str] = edited["titles"]
    phases["editorial"] = round(time.monotonic() - t0, 1)

    # Packaging (title, description, chapters, tags)
    p_raw = await app.call(
        f"{app.node_id}.finalize_package", final_text="\n\n".join(final_sections), titles=titles, topic=brief.topic, model=model,
    )
    package = Package(**p_raw)
    chapter_titles = package.chapter_titles if len(package.chapter_titles) == len(titles) else titles
    write_json(out / "package.json", package.model_dump())
    phases["package"] = round(time.monotonic() - t0, 1)

    # Phase 4 — production (narration ∥ footage → assembly via harness → QC)
    video = await app.call(
        f"{app.node_id}.produce_video", sections=final_sections, titles=chapter_titles, title=package.title,
        tone=brief.tone, run_id=run_id, captions=captions, voice=voice, model=model,
    )
    phases["production"] = round(time.monotonic() - t0, 1)

    description = package.description.strip()
    if video.get("chapters"):
        description += "\n\nChapters:\n" + "\n".join(video["chapters"])
    if video.get("credits"):
        description += "\n\nFootage credits:\n" + "\n".join(video["credits"])
    sources = "\n".join(f"[{c.key}] {c.source_title or c.source_url} — {c.source_url}" for c in claims)
    description += "\n\nSources:\n" + sources

    result = {
        "run_id": run_id,
        "output_dir": str(out),
        "video_path": video["video_path"],
        "duration_s": video["duration_s"],
        "title": package.title,
        "description": description,
        "tags": package.tags,
        "chapters": video.get("chapters", []),
        "script": {"sections": final_sections, "titles": chapter_titles, "word_count": edited["word_count"]},
        "brief": brief.model_dump(),
        "angle": angle,
        "research": {
            "n_claims": len(claims),
            "lanes_run": research["lanes_run"],
            "verified": len(research["verdicts"]),
            "caveats": research["caveats"],
            "lane_errors": research.get("lane_errors", []),
            "tool_calls": research.get("tool_calls", 0),
            "dossier_confident": dossier.confident,
        },
        "editorial": {"rounds": edited["rounds"], "scrubbed": edited.get("scrubbed", False), "shipped": edited["shipped"], "blocking_left": edited["blocking_left"]},
        "production": {k: video.get(k) for k in ("n_sentences", "n_shots", "footage_shots", "card_shots", "providers", "assembly", "qc", "failed_tts_sentences", "voice", "narration_s")},
        "credits": video.get("credits", []),
        "model": model or DEFAULT_MODEL,
        "phase_times_s": phases,
        "confident": bool(research.get("confident")) and bool(script.get("confident", True)) and bool(edited.get("shipped")),
    }
    write_json(out / "result.json", result)
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8001")), auto_port=False)
