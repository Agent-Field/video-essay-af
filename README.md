# video-essay-af

**Prompt → researched, fact-checked, edited, narrated video essay cut from free stock footage — one API call on [AgentField](https://agentfield.ai).**

```
"Why the humble shipping container quietly reshaped the global economy"
        │
        ▼
 interpret_brief ─▶ run_research ─▶ write_script ─▶ run_editorial ─▶ finalize_package ─▶ produce_video
                    (N lanes ∥,      (3 angles ∥ →    (3 editors ∥ →                      (shots → narration ∥
                     adversarial      judge → outline   gate → revise,                      footage ×N → AForge
                     fact-check,      → sections ∥)     ≤2 rounds)                          harness cut → ffprobe QC)
                     gap round)
        │
        ▼
 output/<run_id>/final.mp4   + result.json, research/, script/, audio/, footage/, workspace/
```

28 reasoners + 2 skills on one agent node. The graph's width is decided at runtime (how many research lanes, whether a second round runs, how many sections, how many shots, which footage ladder each shot climbs), every cross-reasoner call goes through the control plane, and the final cut is made by [AForge](https://agentfield.ai/docs/integrations/aforge) — AgentField's native coding harness — working in a prepared ffmpeg workspace, verified with ffprobe, with a deterministic Python cut behind it.

## What you get per run

| File | What |
|---|---|
| `final.mp4` | 1920×1080 30 fps H.264 + AAC, narrated, captioned, stock-footage B-roll, fades/transitions |
| `result.json` | title, description (with chapters, footage credits and every source), script, brief, angle, research stats, editorial rounds, QC |
| `research/` | `plan.json`, `findings.json`, `claims.json` (citation keys), `verdicts.json` (adversarial checks), `dossier.json` |
| `script/` | `angles.json`, `outline.json`, `draft_v0_sections.md` … `final.md`, `editorial.json` |
| `audio/` | `narration.wav` + per-sentence WAVs + `timings.json` |
| `footage/`, `workspace/` | downloaded clips; the exact directory AForge worked in (`edl.json`, `BRIEF.md`, `clips/`, `captions.ass`, `qc.json`) |

## 🚀 Run it

```bash
cp .env.example .env          # paste OPENROUTER_API_KEY (reasoning + Gemini TTS + AForge — one key)
docker compose up --build     # control plane :8080, agent :8001
```

For host runs, set `AFORGE_BIN` in `.env` to the absolute path of your `aforge` binary;
`main.py` loads `.env` before the SDK starts. Without it, agentfield 0.1.143 and later run
CodeAF for `aforge` requests. The Docker image already sets it. This product stays on
AForge on purpose until its own CodeAF migration.

Optional free keys that noticeably improve B-roll / research: `PEXELS_API_KEY`, `PIXABAY_API_KEY`, `TAVILY_API_KEY`. Without them footage comes from Wikimedia Commons (keyless) plus generated text cards, and search uses DuckDuckGo + Wikipedia.

## 🌐 Open the UI

- Workflow DAG / executions: http://localhost:8080/ui/
- Discovery: http://localhost:8080/api/v1/discovery/capabilities

## ✅ Verify registration

```bash
curl -fsS http://localhost:8080/api/v1/discovery/capabilities \
  | jq '.capabilities[] | select(.agent_id=="video-essay-af")
         | {agent_id, n_reasoners: (.reasoners|length),
            entry: [.reasoners[] | select(.tags[]? == "entry") | .id]}'
```

## 🎯 Try it (async — a 3-minute essay takes ~30–50 min with the default model)

```bash
EXEC_ID=$(curl -sS -X POST http://localhost:8080/api/v1/execute/async/video-essay-af.create_video_essay \
  -H 'Content-Type: application/json' \
  -d @sample_payload.json | jq -r '.execution_id')

while :; do
  R=$(curl -sS http://localhost:8080/api/v1/executions/$EXEC_ID)
  S=$(echo "$R" | jq -r '.status')
  case "$S" in
    succeeded) echo "$R" | jq '.result | {video_path, duration_s, title, chapters, research, editorial, production}'; break ;;
    failed)    echo "$R" | jq '.'; exit 1 ;;
    *)         sleep 10 ;;
  esac
done
```

`sample_payload.json`:

```json
{"input": {"prompt": "Why the humble shipping container quietly reshaped the global economy — and what it destroyed along the way",
           "target_minutes": 3, "style": "curious explainer", "captions": true}}
```

Inputs: `prompt` (required), `target_minutes` (1–12, default 4), `style` (narrator register), `captions` (burn sentence captions, default true), `voice` (Gemini voice name, default chosen from tone), `model` (per-request override, e.g. `openrouter/deepseek/deepseek-v4-pro-0813`).

Or with the `af` CLI: `af call video-essay-af.create_video_essay --in '{"prompt": "the placebo effect", "target_minutes": 3}'`.

## 🏆 Showpiece — the verifiable workflow chain

Every reasoner call in the run (lanes, fact-checks, editors, footage picks, the harness) is a signed credential:

```bash
WF=$(curl -s http://localhost:8080/api/v1/executions/$EXEC_ID | jq -r '.workflow_id')
curl -s http://localhost:8080/api/v1/did/workflow/$WF/vc-chain | jq '.[:3]'
```

## How it works

1. **Brief** — `interpret_brief` turns the prompt into topic, central question, audience, tone, research questions.
2. **Research** — `plan_research` (the research lead) decides 3–6 lanes and *writes each researcher's assignment*; `research_lane` ×N run in parallel as tool-using loops over the `web_search` / `fetch_page` skills and hand notes to `extract_claims`; code assigns citation keys and enforces that a claim may only cite a URL its lane actually touched; `verify_claim` adversarially tries to refute the load-bearing claims; `find_gaps` may trigger one bounded second round; `synthesize_dossier` writes the dossier.
3. **Script** — three `propose_angle` writers (reversal / human story / stakes) → `judge_angles` → `build_outline` (seconds normalised in code) → `write_section` ×S in parallel with a code-checked word budget → `smooth_transitions`.
4. **Editorial** — `edit_facts` ∥ `edit_story` ∥ `edit_voice` → `gate_script` → `revise_draft`, at most two rounds, plus one bounded scrub pass on remaining blocking fact notes; `result.editorial.shipped` tells you honestly whether the fact editor signed off. Then `finalize_package` (title, description, chapters, tags).
5. **Production** — `direct_visuals` groups sentences into shots with concrete footage queries; `synthesize_narration` (sentence-by-sentence Gemini TTS, measured timings) runs in parallel with `source_shot_footage` ×N (Pexels → Pixabay → Wikimedia Commons → `pick_footage` judge → download → ffprobe → text-card fallback); code builds the timeline and normalises every clip to its exact slot; `assemble_video` hands the workspace (`edl.json`, `BRIEF.md`, clips, narration, captions) to **AForge** via `app.harness()`, verifies the artifact with ffprobe, and falls back to a single-pass ffmpeg cut if needed.

## A real run (the sample payload, default model, no footage keys)

| Phase | What happened | Wall time |
|---|---|---|
| brief | topic, central question, 6 research questions | 15 s |
| research | 6 lanes in parallel, 72 tool calls, 55 claims (41 strong / 9 moderate / 5 weak), 6 adversarially verified | ~28 min |
| script | 3 angles → judge → 5-section outline → sections ∥ → smooth | ~4 min |
| editorial | 2 rounds, fact/story/voice editors, gate → `shipped: true`, 0 blocking | ~9 min |
| production | 38 sentences → 26 shots (8 Commons clips, 18 text cards), Gemini TTS, AForge cut passed ffprobe QC | ~11 min |
| **total** | `final.mp4` 1920×1080, 188 s, with chapters, credits and 55 sources in the description | **~52 min** |

With `PEXELS_API_KEY` set the footage/card ratio flips; with a stronger `model` the prose gets better; research time scales with `VEA_MAX_LANES` / `VEA_LANE_MAX_TOOL_CALLS`.

## Configuration

| Env | Default | What |
|---|---|---|
| `OPENROUTER_API_KEY` | — | required |
| `AI_MODEL` | `openrouter/deepseek/deepseek-v4-flash` | reasoning model; override per request with `model` |
| `VEA_TTS_MODEL` / `VEA_TTS_SPEED` | `google/gemini-3.1-flash-tts-preview` / `1.12` | narration |
| `VEA_HARNESS_MODEL` | AForge default | model the finishing editor uses |
| `VEA_USE_HARNESS` | `true` | `false` → always use the Python cut |
| `VEA_EDIT_ROUNDS` / `VEA_VERIFY_CLAIMS` / `VEA_MAX_LANES` | `2` / `6` / `6` | budgets |
| `PEXELS_API_KEY`, `PIXABAY_API_KEY`, `TAVILY_API_KEY`, `BRAVE_API_KEY`, `SERPER_API_KEY` | — | optional providers |
| `AGENTFIELD_HARNESS_TIMEOUT_SECONDS` | `1500` | wall clock for the AForge cut |

## Stop

```bash
docker compose down
```

## Extending

See [CLAUDE.md](CLAUDE.md) — architecture contract, binding primitive rules, anti-patterns, extension points.
