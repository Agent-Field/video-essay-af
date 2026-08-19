"""Sentence-by-sentence narration with sample-accurate timings.

Why not one big TTS call plus ASR alignment?  Because the engine already
knows where every sentence starts — we just have to stop throwing that
information away.  Chop the script into sentences, synthesize each one as
its own WAV, then concatenate the WAVs ourselves with the stdlib ``wave``
module.  Because we build the container, we know the exact frame offset of
every sentence: the timings are sample-accurate, not estimated, and they
survive the speed-up (``atempo`` is applied per sentence, *before* the
concat, so the measured length is the final length).

Transport is a bare ``httpx`` POST to OpenRouter's ``/audio/speech``.  The
endpoint answers with raw little-endian 16-bit PCM (mono, 24 kHz) — no
container — so we wrap it in a WAV header locally.

Gemini TTS also reads inline stage directions (``[warm]``, ``[curious]``)
without speaking them.  They stay in the text we send and are stripped
everywhere we count words or show text on screen.

No agentfield imports: this module is plain async Python so it can be
unit-tested and reused outside the agent runtime.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import wave
from pathlib import Path

import httpx

from reasoners.models import Sentence, SentenceTiming

# ───── Constants ─────────────────────────────────────────────────────

TTS_ENDPOINT = "https://openrouter.ai/api/v1/audio/speech"

# Gemini 3.1 Flash TTS: cheap, fast, and the only widely-available model
# that honours inline audio tags ([curious], [warm]) as stage directions.
DEFAULT_TTS_MODEL = os.getenv("VEA_TTS_MODEL", "google/gemini-3.1-flash-tts-preview")

# atempo factor applied to every sentence.  Gemini reads at ~110-130 wpm,
# which drags for an essay; 1.12 lands near documentary pace without the
# chipmunk artefacts that start creeping in past ~1.35.
DEFAULT_SPEED = float(os.getenv("VEA_TTS_SPEED", "1.12"))

# The endpoint always answers 24 kHz mono s16le for response_format=pcm.
TTS_SAMPLE_RATE = 24000
_CHANNELS = 1
_SAMPLE_WIDTH = 2  # bytes — 16-bit

# Words per minute used only to size the silence that stands in for a
# sentence whose synthesis failed, so the video stays in sync.
_FALLBACK_WPM = 150.0
_MIN_FALLBACK_S = 0.8

# Retry backoff between synthesis attempts, in seconds.
_BACKOFF = (0.8, 2.0, 4.0)

# Gemini voice per narrator register.  Keys are matched as substrings of
# the tone string, so "wry investigative explainer" resolves without an
# exact-match table.  Order matters — first key found in the tone wins.
VOICES_BY_TONE: dict[str, str] = {
    "wonder": "Kore",            # warm, curious — the default essayist
    "curious": "Kore",
    "warm": "Aoede",             # friendly, unhurried
    "documentary": "Aoede",
    "urgent": "Charon",          # deep, serious — gravitas
    "investigative": "Charon",
    "deadpan": "Schedar",        # neutral, measured
    "wry": "Schedar",
    "playful": "Puck",           # bright, conversational
}

DEFAULT_VOICE = "Kore"

_TAG_RE = re.compile(r"\[[^\]]*\]")
_WORD_RE = re.compile(r"[A-Za-z0-9'’\-]+")


# ───── Small pure helpers ────────────────────────────────────────────


def voice_for_tone(tone: str) -> str:
    """Resolve a free-text tone ("wry investigative") to a Gemini voice."""
    needle = (tone or "").lower()
    for key, voice in VOICES_BY_TONE.items():
        if key in needle:
            return voice
    return DEFAULT_VOICE


def strip_tags(text: str) -> str:
    """Drop ``[...]`` stage directions and collapse whitespace.

    The tags steer delivery but are never spoken, so anything that counts
    words or renders text on screen must look at the stripped form.
    """
    return " ".join(_TAG_RE.sub(" ", text or "").split())


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(strip_tags(text)))


def _estimated_seconds(text: str) -> float:
    """How long a sentence *would* have taken, for silence substitution."""
    return max(_MIN_FALLBACK_S, _word_count(text) / _FALLBACK_WPM * 60.0)


def wav_duration(path: Path) -> float:
    """Duration in seconds straight from the WAV header — no subprocess."""
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        if rate <= 0:
            raise RuntimeError(f"wav_duration: bad sample rate in {path}")
        return wf.getnframes() / float(rate)


async def probe_duration(path: Path) -> float:
    """Duration in seconds via ffprobe (works for any container)."""
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"probe_duration: ffprobe failed on {path} "
            f"({proc.returncode}): {stderr.decode(errors='replace')[-300:]}"
        )
    raw = stdout.decode(errors="replace").strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"probe_duration: unparseable ffprobe output {raw!r}") from exc


# ───── WAV plumbing ──────────────────────────────────────────────────


def _pcm_to_wav_bytes(pcm: bytes) -> bytes:
    """Wrap raw little-endian s16 PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(_SAMPLE_WIDTH)
        wf.setframerate(TTS_SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


def _silence_bytes(seconds: float) -> bytes:
    frames = max(0, int(round(max(0.0, seconds) * TTS_SAMPLE_RATE)))
    return b"\x00" * (frames * _CHANNELS * _SAMPLE_WIDTH)


def _write_silence_wav(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(_SAMPLE_WIDTH)
        wf.setframerate(TTS_SAMPLE_RATE)
        wf.writeframes(_silence_bytes(seconds))


def _atempo_chain(factor: float) -> str:
    """atempo is only well-behaved in [0.5, 2.0]; chain it for anything else."""
    parts: list[str] = []
    remaining = factor
    while remaining > 2.0:
        parts.append("atempo=2.000")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.500")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.4f}")
    return ",".join(parts)


async def _apply_speed(src: Path, dst: Path, factor: float) -> None:
    """Re-time ``src`` into ``dst`` with atempo, preserving pitch and PCM params.

    Output params must match the input exactly — the native concat later
    refuses to splice WAVs with different rate/width/channels.
    """
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-filter:a", _atempo_chain(factor),
        "-c:a", "pcm_s16le",
        "-ar", str(TTS_SAMPLE_RATE),
        "-ac", str(_CHANNELS),
        str(dst),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"_apply_speed: ffmpeg atempo failed ({proc.returncode}): "
            f"{stderr.decode(errors='replace')[-400:]}"
        )


def _concat_with_gaps(
    paths: list[Path],
    out_path: Path,
    *,
    lead_s: float,
    gap_s: float,
    tail_s: float,
) -> list[tuple[float, float]]:
    """Splice sentence WAVs into one file, padding with silence.

    Returns the ``(start_s, end_s)`` span of each sentence's own audio —
    lead-in and gaps excluded — computed from frame offsets, so the spans
    are exact rather than a running sum of floating-point durations.
    """
    if not paths:
        raise ValueError("_concat_with_gaps: no input WAVs")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    spans: list[tuple[float, float]] = []
    rate = float(TTS_SAMPLE_RATE)

    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(_CHANNELS)
        out.setsampwidth(_SAMPLE_WIDTH)
        out.setframerate(TTS_SAMPLE_RATE)

        cursor = 0  # frames written so far
        lead_frames = int(round(max(0.0, lead_s) * TTS_SAMPLE_RATE))
        out.writeframes(_silence_bytes(lead_s))
        cursor += lead_frames

        gap_frames = int(round(max(0.0, gap_s) * TTS_SAMPLE_RATE))
        last = len(paths) - 1
        for i, path in enumerate(paths):
            with wave.open(str(path), "rb") as src:
                if (
                    src.getnchannels(),
                    src.getsampwidth(),
                    src.getframerate(),
                ) != (_CHANNELS, _SAMPLE_WIDTH, TTS_SAMPLE_RATE):
                    raise RuntimeError(
                        f"_concat_with_gaps: PCM param mismatch on {path.name} "
                        f"— cannot splice"
                    )
                n_frames = src.getnframes()
                out.writeframes(src.readframes(n_frames))
            start = cursor / rate
            cursor += n_frames
            spans.append((start, cursor / rate))
            if i != last and gap_frames:
                out.writeframes(_silence_bytes(gap_s))
                cursor += gap_frames

        if tail_s > 0:
            out.writeframes(_silence_bytes(tail_s))

    return spans


# ───── Synthesis ─────────────────────────────────────────────────────


async def _post_speech(text: str, voice: str, model: str, timeout: float) -> bytes:
    """One POST to OpenRouter /audio/speech → raw PCM bytes."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            TTS_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": text,
                "voice": voice,
                "response_format": "pcm",
            },
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"TTS HTTP {resp.status_code}: {resp.text[:300]}"
        )
    content_type = resp.headers.get("content-type", "")
    if "json" in content_type.lower():
        raise RuntimeError(f"TTS returned JSON instead of audio: {resp.text[:300]}")
    pcm = resp.content
    if len(pcm) < 2 * TTS_SAMPLE_RATE // 100:  # < 10 ms of audio is not speech
        raise RuntimeError(f"TTS returned {len(pcm)} bytes — no usable audio")
    return pcm


async def synthesize_sentence(
    text: str,
    voice: str,
    out_wav: Path,
    *,
    model: str | None = None,
    speed: float | None = None,
    timeout: float = 120.0,
    retries: int = 3,
) -> float:
    """Synthesize one sentence to ``out_wav`` and return its duration.

    ``text`` keeps its inline ``[tag]`` stage directions — Gemini reads
    them as delivery notes and never speaks them.

    The result is 24 kHz mono s16le WAV, already sped up by ``speed`` so
    the measured duration is the duration the final film will use.  Raises
    ``RuntimeError`` if every attempt fails (or the key is missing).
    """
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tts_model = model or DEFAULT_TTS_MODEL
    factor = DEFAULT_SPEED if speed is None else float(speed)

    attempts = max(1, retries + 1)
    last_error: Exception | None = None
    pcm: bytes | None = None
    for attempt in range(attempts):
        try:
            pcm = await _post_speech(text, voice, tts_model, timeout)
            break
        except RuntimeError as exc:
            # A missing key will never fix itself — fail immediately.
            if "OPENROUTER_API_KEY not set" in str(exc):
                raise
            last_error = exc
        except (httpx.HTTPError, OSError) as exc:
            last_error = exc
        if attempt < attempts - 1:
            await asyncio.sleep(_BACKOFF[min(attempt, len(_BACKOFF) - 1)])

    if pcm is None:
        raise RuntimeError(
            f"synthesize_sentence: all {attempts} attempts failed for "
            f"{text[:60]!r} — last error: {last_error}"
        )

    if abs(factor - 1.0) < 1e-3:
        out_wav.write_bytes(_pcm_to_wav_bytes(pcm))
    else:
        raw_wav = out_wav.with_suffix(".raw.wav")
        raw_wav.write_bytes(_pcm_to_wav_bytes(pcm))
        try:
            await _apply_speed(raw_wav, out_wav, factor)
        finally:
            raw_wav.unlink(missing_ok=True)

    return await probe_duration(out_wav)


async def synthesize_sentences(
    sentences: list[Sentence],
    voice: str,
    out_dir: Path,
    *,
    model: str | None = None,
    speed: float | None = None,
    max_concurrency: int = 6,
    gap_s: float = 0.12,
) -> tuple[Path, list[SentenceTiming], list[int]]:
    """Narrate every sentence, splice them, and report exact timings.

    Sentences synthesize in parallel (bounded by ``max_concurrency``) into
    ``out_dir/sentences/s{index:03d}.wav``.  A sentence that fails every
    retry does **not** sink the run: it is replaced by silence the length
    the line would have taken, and its index comes back in the third
    element of the tuple so the caller can decide whether to care.

    The spliced ``out_dir/narration.wav`` gets a 0.35 s lead-in, ``gap_s``
    between sentences and a 0.8 s tail; the returned ``SentenceTiming``
    spans cover each sentence's own audio only, gaps excluded.
    """
    sentences = list(sentences)
    if not sentences:
        raise ValueError("synthesize_sentences: no sentences to narrate")

    out_dir = Path(out_dir)
    sent_dir = out_dir / "sentences"
    sent_dir.mkdir(parents=True, exist_ok=True)

    paths = [sent_dir / f"s{s.index:03d}.wav" for s in sentences]
    failed: list[int] = []
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def _one(sentence: Sentence, path: Path) -> None:
        async with semaphore:
            try:
                await synthesize_sentence(
                    sentence.text, voice, path, model=model, speed=speed,
                )
            except RuntimeError as exc:
                if "OPENROUTER_API_KEY" in str(exc):
                    raise  # not retryable, not a per-sentence problem — fail the run loudly
                # Keep the timeline intact: stand in silence of the length
                # this line would have taken, and flag it upstream.
                _write_silence_wav(path, _estimated_seconds(sentence.text))
                failed.append(sentence.index)
            except Exception:
                _write_silence_wav(path, _estimated_seconds(sentence.text))
                failed.append(sentence.index)

    await asyncio.gather(*(_one(s, p) for s, p in zip(sentences, paths, strict=True)))
    if sentences and len(failed) == len(sentences):
        raise RuntimeError(
            f"synthesize_sentences: every one of {len(sentences)} sentences failed — "
            "check OPENROUTER_API_KEY / VEA_TTS_MODEL (a fully silent narration is not a video essay)"
        )
    if sentences and len(failed) > max(2, len(sentences) // 2):
        raise RuntimeError(
            f"synthesize_sentences: {len(failed)}/{len(sentences)} sentences failed TTS — refusing to ship a mostly-silent narration"
        )

    narration_path = out_dir / "narration.wav"
    spans = _concat_with_gaps(
        paths, narration_path, lead_s=0.35, gap_s=gap_s, tail_s=0.8,
    )

    timings = [
        SentenceTiming(index=s.index, start_s=round(start, 4), end_s=round(end, 4))
        for s, (start, end) in zip(sentences, spans, strict=True)
    ]

    (out_dir / "timings.json").write_text(
        json.dumps(
            [
                {
                    "index": t.index,
                    "start_s": t.start_s,
                    "end_s": t.end_s,
                    "text": s.text,
                }
                for t, s in zip(timings, sentences, strict=True)
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return narration_path, timings, sorted(failed)


__all__ = [
    "DEFAULT_SPEED",
    "DEFAULT_TTS_MODEL",
    "DEFAULT_VOICE",
    "TTS_ENDPOINT",
    "TTS_SAMPLE_RATE",
    "VOICES_BY_TONE",
    "probe_duration",
    "strip_tags",
    "synthesize_sentence",
    "synthesize_sentences",
    "voice_for_tone",
    "wav_duration",
]
