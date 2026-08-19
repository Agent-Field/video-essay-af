# CLAUDE.md — video-essay-af

## Mission

Turn a one-line prompt into a finished, narrated, fact-checked video essay cut from free stock footage — research → script → editorial → final draft → footage → narration → assembly — on the AgentField control plane, with every step visible in the workflow DAG.

External callers should hit `video-essay-af.create_video_essay` first.

## Architecture at a glance

- **Pattern(s) (post-hoc names, not templates):** a composition cascade (entry → phase orchestrators → specialists → atomic reasoners) in which research is *meta-prompted fan-out → verify → gap-find → bounded recurse*, scripting is *parallel perspectives → judge → parallel writers*, editorial is a *bounded control loop with a gate*, and production is *fan-out + a delegated-engineer harness slot verified at the membrane*. Derived via the `agentfield` skill's procedure (cognitive jobs → autonomy → verification rungs → dynamism rung + budgets).
- **Topology:** one AgentField node (`video-essay-af`) with 28 reasoners + 2 skills across five routers (`reasoners/tools.py`, `research.py`, `script.py`, `editorial.py`, `production.py`).
- **Depth from entry to leaf:** 4 layers (e.g. `create_video_essay → run_research → research_lane → extract_claims`, `create_video_essay → produce_video → source_shot_footage → pick_footage`).
- **Entry reasoner:** `create_video_essay(prompt, target_minutes=4, style, captions=True, voice=None, model=None)` — orchestrates the five phases and writes `output/<run_id>/result.json`.
- **Internal reasoners:**
  - `interpret_brief` (`.ai()`) — prompt → Brief (topic, central question, audience, tone, research questions).
  - `run_research` (orchestrator) — plan → lanes ∥ → dedupe/keys (code) → verify top-K ∥ → one gap round (≤2 lanes) → dossier.
  - `plan_research` (`.ai()`) — the research lead; writes each lane's assignment at runtime (meta-prompting).
  - `research_lane` (`.ai(tools=…)` + `app.call`) — tool loop over `web_search`/`fetch_page`, then `extract_claims`; URL invariant enforced in code.
  - `extract_claims` (`.ai()`) — notes → citable claims.
  - `verify_claim` (`.ai(tools=…)` + `.ai()`) — adversarial refuter for load-bearing claims (rung 6).
  - `find_gaps` (`.ai()`) — decides whether a bounded second research round runs.
  - `synthesize_dossier` (`.ai()`) — the dossier every writer reads.
  - `write_script` (orchestrator) — 3 `propose_angle` ∥ → `judge_angles` → `build_outline` → `write_section` ∥ → `smooth_transitions`.
  - `propose_angle` / `judge_angles` / `build_outline` / `write_section` / `smooth_transitions` (`.ai()`) — see docstrings; word budget checked in code with one retry.
  - `run_editorial` (orchestrator) — ≤2 × [`edit_facts` ∥ `edit_story` ∥ `edit_voice` → `gate_script` → `revise_draft`], then one bounded scrub pass (revise on blocking fact notes only → re-check) if still not shipped.
  - `finalize_package` (`.ai()`) — title, description, chapter titles, tags.
  - `produce_video` (orchestrator) — `direct_visuals` → (`synthesize_narration` ∥ `source_shot_footage` ×N) → timeline/normalise (code) → `assemble_video` → QC.
  - `direct_visuals` (`.ai()`) — shots over indexed sentences; coverage repaired in code.
  - `source_shot_footage` (code + `app.call`) — Pexels → Pixabay → Wikimedia Commons ladder → `pick_footage` → download → ffprobe → card fallback.
  - `pick_footage` (`.ai()`) — relevance judge over candidate clips.
  - `synthesize_narration` (TTS, no LLM) — sentence-by-sentence Gemini TTS via OpenRouter, measured timings.
  - `assemble_video` (`.harness()` provider `aforge`) — the finishing editor works in `output/<run>/workspace/`; verified with ffprobe at the membrane; deterministic `media/assemble.assemble_simple` fallback.
  - Skills: `web_search`, `fetch_page` (tag `research-tool`; discovered by the tool loops through the control plane).
- **Inter-reasoner traffic:** all internal calls go through `app.call(f"{NODE_ID}.X", ...)`. Never direct HTTP.

## Why this architecture (not a chain)

The graph's width is decided at runtime by the data: the research lead decides how many lanes to run and writes their briefs, the gap-finder decides whether a second round happens, the script decides how many sections and shots exist, and each shot decides its own footage ladder — none of that can be drawn as a static DAG. The expensive-to-be-wrong slots are priced accordingly: claims that will carry the essay get an adversarial refuter, the draft gets three independent editors and a gate, and the one delegated-engineer slot (AForge cutting the video) is judged purely on its artifact with ffprobe, with a Python cut behind it. Flattening this into "research → write → render" would lose the parallelism, the verification, and the per-run shape that the control-plane DAG and VC chain exist to record.

## Primitive selection rules (binding)

- `.ai()` is used at: `interpret_brief`, `plan_research`, `extract_claims`, the verdict step of `verify_claim`, `find_gaps`, `synthesize_dossier`, `propose_angle`, `judge_angles`, `build_outline`, `write_section`, `smooth_transitions`, `edit_*`, `gate_script`, `revise_draft`, `finalize_package`, `direct_visuals`, `pick_footage`. Every schema has `confident: bool` and every call site has a fallback (safe default, demotion, or ranking order).
- `.ai(tools=ToolCallConfig(agent_ids=[NODE_ID], tags=["research-tool"]))` is used at: `research_lane`, `verify_claim` — bounded by `max_turns` / `max_tool_calls`. Tools are control-plane capabilities, **not** Python callables.
- `@router.reasoner()` orchestrators: `create_video_essay`, `run_research`, `write_script`, `run_editorial`, `produce_video`, `source_shot_footage`, `assemble_video`.
- `@router.skill()`: `web_search`, `fetch_page`. Deterministic media work lives in `media/` as plain Python (`tts.py`, `subtitles.py`, `footage.py`, `assemble.py`, `websearch.py`).
- `.harness()` is used at: `assemble_video` only (provider `aforge`, `max_turns` cap, `AGENTFIELD_HARNESS_TIMEOUT_SECONDS` wall). The Dockerfile installs AForge via the official installer and `production.harness_available()` guards with `shutil.which`. `VEA_USE_HARNESS=false` skips it.

## Data-flow rules

- Structured JSON between code and reasoners (claims, shots, timings, EDL — code branches on them).
- Tool loops: `_text_of` discards leaked DeepSeek tool-call markup; `_trace_digest` (code-built search/page digest with URLs) is ALWAYS appended to the researcher's notes before claim extraction, so claims trace to pages actually read.
- Prose between reasoners that feed each other context (`helpers.brief_prose`, `claims_prose`, `dossier_prose`, `draft_prose`, `reports_prose`, `sentences_prose`).
- **Cross-`app.call` boundary:** every payload arrives as a dict — reconstruct via `Model(**payload)` on the receiver (see every orchestrator) or render to prose on the sender.
- Budgets: `VEA_MAX_LANES` (6), `VEA_VERIFY_CLAIMS` (6), gap lanes (2), `VEA_EDIT_ROUNDS` (2), `VEA_MAX_SHOTS` (80), footage candidates (8) / download tries (3), `VEA_HARNESS_MAX_TURNS` (40), lane tool loops (10 turns / 12 calls), `VEA_RENDER_CONCURRENCY` (3 concurrent ffmpeg encodes), `VEA_DOWNLOAD_CONCURRENCY` (4).

## Model selection

- Default model: `openrouter/deepseek/deepseek-v4-flash` via `AI_MODEL` env (cheap, fast; upgrade with `openrouter/deepseek/deepseek-v4-pro-0813` or `openrouter/google/gemini-3.7-flash` for better prose).
- The entry reasoner accepts an OPTIONAL `model` parameter; it propagates to every child via `app.call(..., model=model)`.
- Narration: `VEA_TTS_MODEL` (Gemini TTS via OpenRouter `/audio/speech`). Harness: `VEA_HARNESS_MODEL` / `AFORGE_MODEL` (AForge default otherwise).
- `OPENROUTER_API_KEY` is required (reasoning + TTS + AForge). `PEXELS_API_KEY` / `PIXABAY_API_KEY` / `TAVILY_API_KEY` are optional upgrades.

## Runtime contract

- Local runtime is `docker-compose.yml` in this directory: `agentfield/control-plane:latest` + this agent node (`Dockerfile`: python:3.11-slim + ffmpeg + fonts + AForge).
- Outputs land in `./output/<run_id>/` (`final.mp4`, `result.json`, `research/`, `script/`, `audio/`, `footage/`, `workspace/`).
- Default ports: control plane `8080`, agent node `8001` (`AGENTFIELD_HTTP_PORT` / `AGENT_NODE_PORT`).
- A 3-minute essay takes ~30–50 minutes end-to-end with the default model (research tool loops and editorial rounds dominate; cost ≈ $0.50–1.50 per essay on deepseek-v4-flash). Always use the **async** execute endpoint.

## Delivery contract — every change must preserve

- ✅ `docker compose up --build` runs (validated with `docker compose config`)
- ✅ `.env.example` lists every key the code reads
- ✅ `README.md` keeps the build checks (health → discovery/capabilities → async execute + poll → vc-chain)
- ✅ The canonical async curl in the README — body `{"input": {...}}` — returns a real `final.mp4`
- ✅ This `CLAUDE.md`

## Validation commands (run after every change)

```bash
python3 -m py_compile main.py reasoners/*.py media/*.py
docker compose config > /dev/null
docker compose up --build -d
sleep 10
curl -fsS http://localhost:8080/api/v1/health
curl -fsS http://localhost:8080/api/v1/discovery/capabilities | jq '.capabilities[] | select(.agent_id=="video-essay-af") | .reasoners | map(.id)'
# then the canonical async curl from README.md, poll until succeeded, ffprobe output/<run_id>/final.mp4
docker compose down
```

## Anti-patterns (reject these)

- ❌ Direct HTTP between reasoners. All internal traffic uses `app.call`.
- ❌ Passing Python functions as `tools=` — tools are control-plane skills discovered by tag.
- ❌ Replacing the harness slot with a fixed ffmpeg script *and deleting the fallback* — keep both; the membrane check decides.
- ❌ Letting an LLM sort/dedupe/normalise/timeline — that is code (`helpers.py`, `media/`).
- ❌ Hardcoding `node_id` or the model; read `AGENT_NODE_ID` / `AI_MODEL` and thread `model=`.
- ❌ Removing a `confident` field or a fallback; removing a cap (lanes, rounds, shots, turns).
- ❌ Feeding `model_dump()` JSON to another LLM as context — render prose first.
- ❌ Flattening the phase orchestrators into the entry reasoner.
- ❌ `app.serve()`; `input_schema=`/`output_schema=` on reasoners; `router.node_id` (does not exist — read env).

## Extension points (where to safely add work)

- New research lens: add a framing string to `script.FRAMINGS`, or a fixed lane the lead must always run in `plan_research`'s fallback list.
- New editor: write `edit_<lens>` with the `_editor(...)` factory in `editorial.py` and add it to the gather in `run_editorial`.
- New footage provider: implement `_search_<provider>` in `media/footage.py` and add it to the ladder; keep `Candidate` shape.
- Background music / voice cloning: extend `media/assemble.assemble_simple` (music ducking is stubbed via `music_path`) and the harness brief.
- Publishing: add a `publish_video` reasoner after `produce_video` behind a human approval gate (rung 7).

## Owner

This system was scaffolded by the `agentfield` skill. To rebuild, run that skill again with the same use case description. To extend, follow this CLAUDE.md. Live SDK docs: https://agentfield.ai/llms.txt (source of truth; the skill's snapshot may drift).
