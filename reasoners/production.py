"""Phase 4 — production: visuals, footage, narration, assembly.

Cognitive jobs:

* ``direct_visuals``        — the visual director turns indexed sentences into
                               shots with concrete stock-footage queries
                               (dynamism rung 3: N shots decided by the script;
                               coverage repaired in code).
* ``pick_footage``          — relevance judge over candidate clips (cheap).
* ``source_shot_footage``   — per-shot: search ladder (Pexels → Pixabay →
                               Wikimedia Commons) → judge → download → ffprobe
                               (rung 3) → next candidate → generated text card.
* ``synthesize_narration``  — sentence-by-sentence TTS with measured timings.
* ``assemble_video``        — the delegated engineer: ``app.harness`` (AForge)
                               works in a prepared workspace and produces the
                               polished cut; verified at the membrane with
                               ffprobe (rung 3); deterministic Python fallback.
* ``produce_video``         — orchestrator: narration ∥ footage → timeline →
                               normalise → assemble → QC.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

from agentfield import AgentRouter

from media import assemble as asm
from media import footage as fz
from media import subtitles as subs
from media import tts

from .helpers import (
    estimate_shot_seconds,
    fallback_assets_credit_lines,
    fmt_timestamp,
    index_sentences,
    repair_shot_coverage,
    run_dir,
    sentences_prose,
    simplify_query,
    word_count,
    write_json,
    write_text,
)
from .models import (
    AssemblyReport,
    EDLEntry,
    FootagePick,
    Sentence,
    SentenceTiming,
    Shot,
    ShotAsset,
    ShotList,
)

NODE_ID = os.getenv("AGENT_NODE_ID", "video-essay-af")
router = AgentRouter(prefix="", tags=["production"])

LLM_TIMEOUT = float(os.getenv("VEA_LLM_TIMEOUT", "300"))
WIDTH, HEIGHT, FPS = asm.WIDTH, asm.HEIGHT, asm.FPS
TARGET_SHOT_SECONDS = float(os.getenv("VEA_SHOT_SECONDS", "7"))
MAX_SHOTS = int(os.getenv("VEA_MAX_SHOTS", "80"))
FOOTAGE_CANDIDATES = 8
FOOTAGE_DOWNLOAD_TRIES = 3
DOWNLOAD_CONCURRENCY = int(os.getenv("VEA_DOWNLOAD_CONCURRENCY", "4"))
RENDER_CONCURRENCY = int(os.getenv("VEA_RENDER_CONCURRENCY", "3"))
HARNESS_MAX_TURNS = int(os.getenv("VEA_HARNESS_MAX_TURNS", "40"))
USE_HARNESS = os.getenv("VEA_USE_HARNESS", "true").lower() not in {"0", "false", "no"}
HARNESS_PROVIDER = os.getenv("VEA_HARNESS_PROVIDER", "aforge")
HARNESS_MODEL = os.getenv("VEA_HARNESS_MODEL") or None

_download_sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)


def harness_available() -> bool:
    if not USE_HARNESS:
        return False
    if HARNESS_PROVIDER != "aforge":
        return shutil.which(HARNESS_PROVIDER.replace("claude-code", "claude")) is not None
    bin_path = os.getenv("AFORGE_BIN", "aforge")
    return shutil.which(bin_path) is not None or Path(bin_path).exists()


# ───── Visual direction ──────────────────────────────────────────────


@router.reasoner()
async def direct_visuals(sentences_text: str, n_sentences: int, tone: str, max_shots: int, model: str | None = None) -> ShotList:
    """The visual director: group sentences into shots and write footage queries."""
    shots = await router.ai(
        system=(
            "You are the visual director of a documentary video essay that will be cut from "
            "FREE STOCK FOOTAGE (Pexels, Pixabay, Wikimedia Commons). Group the numbered "
            f"sentences into at most {max_shots} consecutive shots (typically 1–3 sentences each, "
            "a new shot whenever the image should change). Every sentence index from 0 to "
            f"{n_sentences - 1} must be covered exactly once, in order. For each shot write 2–3 "
            "stock-footage search queries that are CONCRETE and VISUAL (things a camera can "
            "film: 'container ship unloading at port aerial', 'vintage 1950s dock workers "
            "archival', 'close-up hands typing keyboard'), never abstract ('economy', "
            "'innovation'); the first query is the best one; the last can be a safe generic "
            "fallback ('city skyline timelapse'). Give every shot a card_text: a 2–6 word "
            "headline (a name, a number, a place, a key phrase — NOT the sentence itself) that "
            "can stand alone on a text card if no footage fits. Give on_screen_text only for "
            "key moments (a name or number worth a lower-third), else null. Give a mood word. "
            "confident=false if the script is too short to cut."
        ),
        user=f"TONE: {tone}\n\nNUMBERED SENTENCES\n{sentences_text}",
        schema=ShotList,
        model=model,
        timeout=LLM_TIMEOUT,
    )
    return shots


@router.reasoner()
async def pick_footage(shot_text: str, candidates_text: str, model: str | None = None) -> FootagePick:
    """Relevance judge: which candidate clip best matches what is being said?"""
    return await router.ai(
        system=(
            "You are a footage researcher choosing B-roll for a narrated documentary. Given what "
            "the narrator says over this shot and a numbered list of candidate clips (title/tags, "
            "source, dimensions, duration), pick the ONE clip a good editor would use: a literal "
            "match or a tasteful metaphor, landscape, not embarrassing. REJECT (do not pick) clips "
            "whose title shows they are about a specific, unrelated named subject or event — e.g. "
            "a named telescope, mission, festival, product demo, or a place that has nothing to do "
            "with the narration — even if one keyword matches; and clips that are clearly a titled "
            "documentary/news segment (likely burned-in text). Prefer clips long enough (duration "
            "≥ needed); a shorter clip can be looped if clearly better. If NO candidate is a "
            "plausible match, return -1: the shot then gets a clean text card, which is better "
            "than a misleading clip. Explain in one sentence."
        ),
        user=f"{shot_text}\n\nCANDIDATES\n{candidates_text}",
        schema=FootagePick,
        model=model,
        timeout=LLM_TIMEOUT,
    )


def _cand_text(cands: list[fz.Candidate]) -> str:
    return "\n".join(
        f"[{i}] {c.title[:110]} — {c.provider}, {c.width}x{c.height}, "
        f"{c.duration_s:.1f}s" + (f", {c.license}" if c.license else "")
        for i, c in enumerate(cands)
    )


@router.reasoner()
async def source_shot_footage(shot: dict, spoken: str, needed_seconds: float, run_id: str, model: str | None = None) -> ShotAsset:
    """Find, judge, download and verify one shot's B-roll; fall back to a text card."""
    s = Shot(**shot)
    foot_dir = run_dir(run_id, "footage")
    cands: list[fz.Candidate] = []
    seen: set[str] = set()
    # Director queries first; then code-simplified 2–3 noun variants (keyless
    # providers such as Wikimedia Commons match short literal queries far better).
    attempts: list[str] = []
    for q in s.queries[:3]:
        attempts.append(q)
        short = simplify_query(q)
        if short and short.lower() != q.lower():
            attempts.append(short)
    for q in attempts:
        if len(cands) >= FOOTAGE_CANDIDATES:
            break
        try:
            found = await fz.search_footage(q, min_duration_s=needed_seconds, limit=FOOTAGE_CANDIDATES)
        except Exception:  # noqa: BLE001 — a provider hiccup must not kill the shot
            found = []
        for c in found:
            if c.download_url not in seen:
                seen.add(c.download_url)
                cands.append(c)
    cands = fz.rank_candidates(cands, needed_seconds)[:FOOTAGE_CANDIDATES]

    order: list[int] = list(range(len(cands)))
    query_used = s.queries[0] if s.queries else ""
    if len(cands) > 1:
        shot_text = f"Shot {s.index} (needs ~{needed_seconds:.0f}s, mood: {s.mood})\nNarration: {spoken}"
        try:
            p_raw = await router.call(f"{NODE_ID}.pick_footage", shot_text=shot_text, candidates_text=_cand_text(cands), model=model)
            pick = FootagePick(**p_raw)
            if pick.confident and pick.candidate_index == -1:
                order = []
            elif 0 <= pick.candidate_index < len(cands):
                order = [pick.candidate_index] + [i for i in order if i != pick.candidate_index]
        except Exception:  # noqa: BLE001 — judge failure → fall back to ranking order
            pass

    for i in order[:FOOTAGE_DOWNLOAD_TRIES]:
        c = cands[i]
        ext = ".mp4" if c.download_url.lower().split("?")[0].endswith(".mp4") else (
            ".webm" if ".webm" in c.download_url.lower() else ".ogv" if ".ogv" in c.download_url.lower() else ".mp4"
        )
        dest = foot_dir / f"shot-{s.index:02d}-{c.provider}{ext}"
        try:
            async with _download_sem:
                await fz.download(c, dest)
            info = await fz.probe(dest)
            if not info.get("ok"):
                dest.unlink(missing_ok=True)
                continue
            return ShotAsset(
                shot_index=s.index, path=str(dest), kind="footage", provider=c.provider,
                credit=c.credit, source_url=c.page_url, duration_s=float(info.get("duration", 0.0)),
                query_used=query_used,
            )
        except Exception:  # noqa: BLE001 — try the next candidate
            dest.unlink(missing_ok=True)
            continue

    # Nothing usable → a generated card (rendered to exact length later, once timings exist).
    return ShotAsset(shot_index=s.index, path="", kind="card", provider="generated", credit="", source_url="", duration_s=0.0, query_used=query_used)


# ───── Narration ─────────────────────────────────────────────────────


@router.reasoner()
async def synthesize_narration(sentences: list[dict], tone: str, run_id: str, voice: str | None = None) -> dict:
    """Sentence-by-sentence TTS; returns narration path + sentence timings."""
    sents = [Sentence(**s) for s in sentences]
    out_dir = run_dir(run_id, "audio")
    chosen_voice = voice or tts.voice_for_tone(tone)
    path, timings, failed = await tts.synthesize_sentences(sents, chosen_voice, out_dir)
    duration = tts.wav_duration(path)
    return {
        "audio_path": str(path),
        "voice": chosen_voice,
        "duration_s": duration,
        "timings": [t.model_dump() for t in timings],
        "failed_sentences": failed,
    }


# ───── Assembly (the harness slot) ───────────────────────────────────


def _brief_md(edl: list[EDLEntry], narration_s: float, captions: bool, title: str) -> str:
    rows = "\n".join(
        f"| {e.index:02d} | {e.clip} | {e.start_s:.2f} | {e.duration_s:.2f} | {e.on_screen_text or ''} |"
        for e in edl
    )
    cap = "captions.ass (burn it in with the subtitles filter; fonts in ./fonts)" if captions else "(no captions file — do not add captions)"
    return f"""# Finishing editor brief — {title}

You are the finishing editor. Everything you need is in THIS directory; work here and do not modify the inputs.

## Inputs
- `edl.json` — machine-readable edit decision list (same data as the table below).
- `clips/shot-NN.mp4` — one silent clip per shot, already normalised to {WIDTH}x{HEIGHT} @ {FPS} fps, yuv420p, EXACT duration per the EDL.
- `audio/narration.wav` — the complete narration, {narration_s:.2f} s, 24 kHz mono. It is the MASTER timeline.
- {cap}

| shot | clip | start_s | duration_s | on-screen text |
|---|---|---|---|---|
{rows}

## Deliverable
Produce `final.mp4` in this directory:
- {WIDTH}x{HEIGHT}, {FPS} fps, H.264 (libx264, yuv420p, crf ≤ 20, +faststart) with AAC 48 kHz stereo audio = narration.wav.
- The video track is the clips in EDL order. **Sync rule: every shot must start at its EDL `start_s` (±0.1 s) and the final duration must equal the narration duration ±0.5 s.**
- Make it look finished: a short fade-in at 0 s and fade-out at the end; transitions between shots — either hard cuts, or crossfades that keep sync (a correct way is `tpad=stop_mode=clone:stop_duration=0.5` on each clip then `xfade=duration=0.5:offset=<cumulative start of the next shot>`; if you are not certain your offsets keep sync, use hard cuts — sync beats style).
- If `captions.ass` exists, burn it in (`subtitles=captions.ass:fontsdir=fonts`).
- Optional, only if cheap and safe: for shots with on-screen text, a tasteful lower-third drawtext with the same font for 2–3 s at the shot start.
- Verify with ffprobe before you finish: duration, resolution, codecs, audio present. If something is off, fix it and re-check.

Write `qc.json` with {{"duration_s", "width", "height", "vcodec", "acodec", "notes"}} after verifying.
"""


@router.reasoner()
async def assemble_video(workspace: str, narration_s: float, title: str, captions: bool) -> dict:
    """Delegated engineer: AForge cuts the video in the workspace; we verify at the membrane.

    Returns {output_path, method, harness_used, harness_ok, issues, qc}.
    """
    ws = Path(workspace)
    final = ws / "final.mp4"
    issues: list[str] = []
    report: AssemblyReport | None = None
    harness_used = False
    harness_error = ""

    if harness_available():
        harness_used = True
        prompt = (
            "Read BRIEF.md and edl.json in the working directory, then produce final.mp4 exactly as "
            "specified (ffmpeg/ffprobe are installed; use them). Verify with ffprobe and write qc.json. "
            "Keep inputs untouched."
        )
        try:
            kwargs: dict[str, Any] = dict(
                schema=AssemblyReport,
                provider=HARNESS_PROVIDER,
                cwd=str(ws),
                max_turns=HARNESS_MAX_TURNS,
                system_prompt=(
                    "You are a senior video editor-engineer who finishes documentary cuts with ffmpeg. "
                    "Work only inside the given directory. Think in terms of the master audio timeline; "
                    "sync is non-negotiable. Verify outputs with ffprobe."
                ),
            )
            if HARNESS_MODEL:
                kwargs["model"] = HARNESS_MODEL
            result = await router.harness(prompt, **kwargs)
            if getattr(result, "is_error", False):
                harness_error = str(getattr(result, "error_message", "") or "harness error")
            parsed = getattr(result, "parsed", None)
            if isinstance(parsed, AssemblyReport):
                report = parsed
            elif isinstance(parsed, dict):
                try:
                    report = AssemblyReport(**parsed)
                except Exception:  # noqa: BLE001
                    report = None
        except Exception as exc:  # noqa: BLE001 — the membrane below decides what happens
            harness_error = f"{type(exc).__name__}: {exc}"

    # Membrane check (rung 3): the harness is judged only on the artifact.
    candidate = final
    if report and report.output_path:
        rp = Path(report.output_path)
        if rp.is_file():
            candidate = rp
    ok, problems, probe = await asm.verify_final(candidate, narration_s, tolerance_s=1.0)
    if ok:
        if candidate != final:
            shutil.copyfile(candidate, final)
        method = report.method if report else "harness (no structured report)"
        if report and not report.confident:
            issues.append("harness reported low confidence; artifact passed verification")
        return {"output_path": str(final), "method": method, "harness_used": harness_used, "harness_ok": True, "issues": issues, "qc": probe}

    if harness_used:
        issues.append("harness output failed verification: " + "; ".join(problems[:4]) + (f" ({harness_error})" if harness_error else ""))
    # Deterministic fallback cut.
    edl = [EDLEntry(**e) for e in json.loads((ws / "edl.json").read_text())["shots"]]
    clips = [ws / e.clip for e in edl]
    fonts_dir = ws / "fonts"
    ass = ws / "captions.ass"
    fb = ws / "final_fallback.mp4"
    await asm.assemble_simple(
        clips, ws / "audio" / "narration.wav", fb,
        captions_ass=ass if (captions and ass.exists()) else None,
        fonts_dir=fonts_dir if fonts_dir.exists() else None,
    )
    ok2, problems2, probe2 = await asm.verify_final(fb, narration_s, tolerance_s=1.5)
    if not ok2:
        raise RuntimeError("fallback assembly failed verification: " + "; ".join(problems2))
    shutil.copyfile(fb, final)
    return {
        "output_path": str(final),
        "method": "python fallback: single-pass ffmpeg concat + fade + subtitles + AAC mux",
        "harness_used": harness_used,
        "harness_ok": False,
        "issues": issues,
        "qc": probe2,
    }


# ───── Orchestrator ──────────────────────────────────────────────────


def _timeline(shots: list[Shot], timings: list[SentenceTiming], total_s: float) -> list[tuple[float, float]]:
    """Gap-free partition of [0, total_s] into one (start, duration) per shot."""
    by_idx = {t.index: t for t in timings}
    starts: list[float] = []
    for i, s in enumerate(shots):
        if i == 0:
            starts.append(0.0)
        else:
            t = by_idx.get(s.sentence_start)
            starts.append(max(starts[-1] + 0.5, (t.start_s - 0.15) if t else starts[-1] + 0.5))
    out: list[tuple[float, float]] = []
    for i, st in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else total_s
        out.append((st, max(0.5, end - st)))
    return out


@router.reasoner()
async def produce_video(
    sections: list[str],
    titles: list[str],
    title: str,
    tone: str,
    run_id: str,
    captions: bool = True,
    voice: str | None = None,
    model: str | None = None,
) -> dict:
    """Orchestrator for production.  Returns paths, shots, assets, QC and chapters."""
    t0 = time.monotonic()
    timings_log: dict[str, float] = {}
    sentences = index_sentences(sections)
    if not sentences:
        raise ValueError("produce_video: empty script")
    total_words = sum(word_count(s.text) for s in sentences)
    est_total_s = total_words / 150 * 60
    max_shots = max(1, min(MAX_SHOTS, math.ceil(est_total_s / TARGET_SHOT_SECONDS) + 2))

    # 1. Visual direction (N shots decided by the script).
    sl_raw = await router.call(
        f"{NODE_ID}.direct_visuals", sentences_text=sentences_prose(sentences),
        n_sentences=len(sentences), tone=tone, max_shots=max_shots, model=model,
    )
    shotlist = ShotList(**sl_raw)
    shots = repair_shot_coverage(shotlist.shots, len(sentences), max_shots)
    timings_log["direct"] = time.monotonic() - t0

    # 2. Narration ∥ footage for every shot.
    def _spoken(s: Shot) -> str:
        return " ".join(x.text for x in sentences if s.sentence_start <= x.index <= s.sentence_end)

    narration_task = router.call(
        f"{NODE_ID}.synthesize_narration", sentences=[s.model_dump() for s in sentences],
        tone=tone, run_id=run_id, voice=voice,
    )
    footage_tasks = [
        router.call(
            f"{NODE_ID}.source_shot_footage", shot=s.model_dump(), spoken=_spoken(s),
            needed_seconds=estimate_shot_seconds(s, sentences), run_id=run_id, model=model,
        )
        for s in shots
    ]
    results = await asyncio.gather(narration_task, *footage_tasks, return_exceptions=True)
    narration = results[0]
    if isinstance(narration, Exception):
        raise RuntimeError(f"narration failed: {narration}") from narration
    timings = [SentenceTiming(**t) for t in narration["timings"]]
    narration_s = float(narration["duration_s"])
    assets: list[ShotAsset] = []
    for s, r in zip(shots, results[1:], strict=True):
        if isinstance(r, Exception):
            assets.append(ShotAsset(shot_index=s.index, path="", kind="card", provider="generated", query_used=s.queries[0] if s.queries else ""))
        else:
            assets.append(ShotAsset(**r))
    timings_log["narration+footage"] = time.monotonic() - t0

    # 3. Timeline + normalised clips (exact durations) + cards for shots without footage.
    ws = run_dir(run_id, "workspace")
    clips_dir = run_dir(run_id, "workspace", "clips")
    (ws / "audio").mkdir(exist_ok=True)
    shutil.copyfile(narration["audio_path"], ws / "audio" / "narration.wav")
    timeline = _timeline(shots, timings, narration_s)
    edl: list[EDLEntry] = []
    font_path, font_family = subs.find_font()
    render_sem = asyncio.Semaphore(RENDER_CONCURRENCY)  # one cap for every ffmpeg encode in this step

    async def _render(shot: Shot, asset: ShotAsset, dur: float, clip: Path) -> None:
        """Normalise the shot's footage to its exact slot, or render its text card.
        A footage clip that fails to normalise becomes a card carrying the shot's
        headline — the run never dies on one bad download."""
        card_text = (shot.card_text or shot.on_screen_text or "").strip()
        async with render_sem:
            if asset.kind == "footage" and asset.path:
                try:
                    await asm.normalize_clip(Path(asset.path), clip, dur)
                    return
                except Exception as exc:  # noqa: BLE001
                    asset.kind, asset.provider, asset.credit, asset.source_url, asset.path = "card", "generated", "", "", ""
                    asset.query_used = f"{asset.query_used} (normalise failed: {str(exc)[:80]})"
            await fz.make_card(card_text, dur, clip, width=WIDTH, height=HEIGHT, fps=FPS, font_path=font_path)

    render_tasks = []
    for s, a, (st, dur) in zip(shots, assets, timeline, strict=True):
        clip = clips_dir / f"shot-{s.index:02d}.mp4"
        render_tasks.append(_render(s, a, dur, clip))
        edl.append(EDLEntry(index=s.index, clip=f"clips/{clip.name}", start_s=round(st, 3), duration_s=round(dur, 3), on_screen_text=s.on_screen_text, sentence_start=s.sentence_start, sentence_end=s.sentence_end))
    render_results = await asyncio.gather(*render_tasks, return_exceptions=True)
    for e, r in zip(edl, render_results, strict=True):
        if isinstance(r, Exception):
            raise RuntimeError(f"render failed for {e.clip}: {r}") from r
    timings_log["normalize"] = time.monotonic() - t0

    # 4. Captions + EDL + brief for the finishing editor.
    if captions:
        subs.write_captions_ass(sentences, timings, ws / "captions.ass", width=WIDTH, height=HEIGHT, font_name=font_family)
        fonts_dir = ws / "fonts"
        fonts_dir.mkdir(exist_ok=True)
        shutil.copyfile(font_path, fonts_dir / Path(font_path).name)
    write_json(ws / "edl.json", {
        "title": title, "width": WIDTH, "height": HEIGHT, "fps": FPS,
        "narration": "audio/narration.wav", "narration_duration_s": narration_s,
        "captions": "captions.ass" if captions else None, "output": "final.mp4",
        "shots": [e.model_dump() for e in edl],
    })
    write_text(ws / "BRIEF.md", _brief_md(edl, narration_s, captions, title))

    # 5. Assemble (harness → membrane check → fallback) and QC.
    asm_out = await router.call(f"{NODE_ID}.assemble_video", workspace=str(ws), narration_s=narration_s, title=title, captions=captions)
    final_path = Path(asm_out["output_path"])
    public_final = run_dir(run_id) / "final.mp4"
    if final_path.resolve() != public_final.resolve():
        shutil.copyfile(final_path, public_final)
    timings_log["assemble"] = time.monotonic() - t0

    # Chapters: first sentence of each section → timestamp.
    by_idx = {t.index: t for t in timings}
    chapters: list[str] = []
    for sec_idx, t_title in enumerate(titles):
        first = next((s for s in sentences if s.section_index == sec_idx), None)
        if first and first.index in by_idx:
            chapters.append(f"{fmt_timestamp(by_idx[first.index].start_s)} {t_title}")
    credits = fallback_assets_credit_lines(assets)
    write_json(run_dir(run_id) / "production.json", {
        "shots": [s.model_dump() for s in shots], "assets": [a.model_dump() for a in assets],
        "edl": [e.model_dump() for e in edl], "narration": narration, "assembly": asm_out,
        "chapters": chapters, "credits": credits, "timings_s": timings_log,
    })
    return {
        "video_path": str(public_final),
        "duration_s": float(asm_out["qc"].get("duration", narration_s)),
        "narration_s": narration_s,
        "voice": narration.get("voice"),
        "n_sentences": len(sentences),
        "n_shots": len(shots),
        "footage_shots": sum(1 for a in assets if a.kind == "footage"),
        "card_shots": sum(1 for a in assets if a.kind == "card"),
        "providers": sorted({a.provider for a in assets if a.kind == "footage"}),
        "assembly": {k: v for k, v in asm_out.items() if k != "qc"},
        "qc": asm_out["qc"],
        "chapters": chapters,
        "credits": credits,
        "failed_tts_sentences": narration.get("failed_sentences", []),
        "timings_s": timings_log,
        "workspace": str(ws),
    }


__all__ = ["router"]
