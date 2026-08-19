"""Clip normalisation, single-pass assembly and QC probing (1920×1080 @ 30 fps).

The strategy is **normalise, then single-pass concat**:

1. ``normalize_clip`` re-encodes every source (stock footage, still image,
   title card) into a *silent* clip of exactly the length the EDL asks for,
   at exactly ``WIDTH×HEIGHT@FPS``, ``yuv420p``, ``setsar=1``, H.264.
   Sources shorter than the slot are looped; longer ones are trimmed from
   the head.  Nothing is ever letterboxed — we scale to fill and centre-crop.
2. ``assemble_simple`` then does **one** ffmpeg invocation that concatenates
   those clips with the ``concat`` *filter*, fades in/out, burns captions with
   libass, mixes narration (plus optional ducked music) and encodes once.

Why this shape:

* The ``concat`` **filter** (not the demuxer) is sample/frame-accurate at the
  boundary, but it demands that every input agree on resolution, SAR, pixel
  format and frame rate — which is precisely what step 1 guarantees.  Feeding
  it heterogeneous stock footage directly is the classic source of "only the
  first clip plays" bugs.
* One encode means one x264 rate-control run and one AAC priming delay for the
  whole essay, so audio never drifts against video across shot boundaries.
* Per-clip normalisation is embarrassingly parallel (``normalize_many``) while
  the final pass is serial, which is the cheapest split of the work.

This module is deliberately dependency-free: stdlib + ``asyncio`` subprocesses
driving ``ffmpeg``/``ffprobe``.  It imports nothing from agentfield so it can be
unit-tested and re-used as the deterministic *fallback* cut when the agentic
finishing-editor path is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps pydantic out of runtime
    from reasoners.models import EDLEntry

# ─────────────────────────────────────────────────────────────────────
# Canvas constants — 16:9 landscape video essay
# ─────────────────────────────────────────────────────────────────────

WIDTH, HEIGHT, FPS = 1920, 1080, 30

#: Sources with these extensions are stills: they need ``-loop 1`` and a Ken
#: Burns move, otherwise a single frame would produce a 1-frame "clip".
IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp"})

#: Ken Burns supersampling.  zoompan crops integer pixel rectangles, so zooming
#: a canvas-sized frame visibly stutters.  Stills are cheap to upscale, so they
#: get 2× headroom; video keeps 1× (motion masks the stepping and 4K zoompan on
#: hundreds of frames is needlessly slow).
_KB_SUPERSAMPLE_IMAGE = 2
_KB_SUPERSAMPLE_VIDEO = 1

_STDERR_TAIL = 1400

_BLANK_PROBE: dict[str, Any] = {
    "ok": False,
    "duration": 0.0,
    "width": 0,
    "height": 0,
    "has_video": False,
    "has_audio": False,
    "vcodec": "",
    "acodec": "",
    "fps": 0.0,
}


# ─────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────


def ffmpeg_escape_filter_path(path: Path | str) -> str:
    """Escape a filesystem path for use as a *filter argument value*.

    Filter-graph parsing treats ``:`` as the key/value separator and ``\\`` as
    the escape character, so both have to be escaped before a path can be
    handed to e.g. ``subtitles=`` or ``fontsdir=``.  Exported because the
    caption writer and any bespoke filter graphs need the identical rule.
    """
    return str(path).replace("\\", "\\\\").replace(":", r"\:")


async def _run(cmd: list[str]) -> tuple[int, bytes, bytes]:
    """Run a subprocess to completion, capturing stdout/stderr."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return int(proc.returncode or 0), out, err


def _fail(label: str, cmd: list[str], rc: int, stderr: bytes) -> RuntimeError:
    return RuntimeError(
        f"assemble: {label} failed (exit {rc}):\n"
        f"  cmd: {shlex.join(cmd)}\n"
        f"  stderr: {stderr.decode(errors='replace')[-_STDERR_TAIL:]}"
    )


def _parse_rate(value: Any) -> float:
    """Parse an ffprobe rational rate string such as ``30000/1001``."""
    if not value:
        return 0.0
    text = str(value)
    try:
        if "/" in text:
            num, _, den = text.partition("/")
            d = float(den)
            return float(num) / d if d else 0.0
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if f == f and f not in (float("inf"), float("-inf")) else default


# ─────────────────────────────────────────────────────────────────────
# Probing
# ─────────────────────────────────────────────────────────────────────


async def probe(path: Path) -> dict:
    """ffprobe a media file.  Never raises — ``ok`` reports success.

    ``ok`` is False when the file is missing, ffprobe errors, the JSON is
    unparseable, or the file carries no video stream.
    """
    p = Path(path)
    result = dict(_BLANK_PROBE)
    if not p.exists():
        return result

    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(p),
    ]
    rc, out, _err = await _run(cmd)
    if rc != 0:
        return result
    try:
        data = json.loads(out.decode(errors="replace") or "{}")
    except (ValueError, UnicodeDecodeError):
        return result

    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = _float((data.get("format") or {}).get("duration"))
    if duration <= 0.0:
        for s in (video, audio):
            if s is not None:
                duration = _float(s.get("duration"))
                if duration > 0.0:
                    break

    fps = 0.0
    if video is not None:
        fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(
            video.get("r_frame_rate")
        )

    result.update(
        {
            "ok": video is not None,
            "duration": duration,
            "width": int(video.get("width") or 0) if video else 0,
            "height": int(video.get("height") or 0) if video else 0,
            "has_video": video is not None,
            "has_audio": audio is not None,
            "vcodec": str(video.get("codec_name") or "") if video else "",
            "acodec": str(audio.get("codec_name") or "") if audio else "",
            "fps": fps,
        }
    )
    return result


# ─────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────


def _fill_chain(width: int, height: int) -> str:
    """Scale-to-fill + centre-crop.  Never letterbox, never distort."""
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}"
    )


async def normalize_clip(
    src: Path,
    dst: Path,
    duration_s: float,
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: int = FPS,
    crf: int = 20,
    preset: str = "fast",
    kenburns: bool = False,
) -> Path:
    """Re-encode ``src`` into a silent ``width×height@fps`` clip of exactly
    ``duration_s`` (±1 frame).

    * Shorter than the slot → the source is looped (``-stream_loop -1``).
    * Longer than the slot  → trimmed from the start.
    * Stills (``.png/.jpg/.jpeg/.webp``) are always ``-loop 1`` + Ken Burns,
      regardless of the ``kenburns`` flag — a static frame otherwise.
    * ``kenburns=True`` on a video applies the same slow push-in.

    Output is H.264 / yuv420p / SAR 1:1 / CFR, ``+faststart``, no audio, which
    is exactly what the ``concat`` filter in :func:`assemble_simple` requires.
    """
    src = Path(src)
    dst = Path(dst)
    duration_s = float(duration_s)
    if duration_s <= 0:
        raise ValueError(f"assemble: duration_s must be > 0, got {duration_s!r}")
    if not src.exists():
        raise RuntimeError(f"assemble: normalize source does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    is_image = src.suffix.lower() in IMAGE_SUFFIXES
    use_kenburns = bool(kenburns or is_image)
    frames = max(1, int(round(duration_s * fps)))

    info = await probe(src)
    src_duration = _float(info.get("duration"))
    # Loop only real media that is genuinely shorter than the slot; stills use
    # -loop 1 instead, and an unknown duration (0.0) is left alone so we don't
    # spin forever on a container ffprobe could not measure.
    needs_loop = (
        not is_image and src_duration > 0.0 and src_duration + 1e-3 < duration_s
    )

    if use_kenburns:
        ss = _KB_SUPERSAMPLE_IMAGE if is_image else _KB_SUPERSAMPLE_VIDEO
        pre_w, pre_h = width * ss, height * ss
        # zoompan `d` = output frames emitted per *input* frame.  A still is a
        # single input frame, so d must span the whole clip.  A video feeds one
        # frame per output frame, so d=1 and `zoom` accumulates frame to frame.
        d = frames if is_image else 1
        vf = (
            f"{_fill_chain(pre_w, pre_h)},"
            f"zoompan=z='min(zoom+0.0006,1.08)':d={d}"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":s={width}x{height}:fps={fps},"
            f"setsar=1,format=yuv420p"
        )
    else:
        vf = (
            f"{_fill_chain(width, height)},"
            f"setsar=1,fps={fps},format=yuv420p"
        )

    cmd: list[str] = ["ffmpeg", "-y", "-loglevel", "error"]
    if is_image:
        cmd += ["-loop", "1"]
    elif needs_loop:
        cmd += ["-stream_loop", "-1"]
    else:
        cmd += ["-ss", "0"]          # trim from the head
    cmd += ["-i", str(src)]
    cmd += [
        "-an",
        "-vf", vf,
        "-r", str(fps),
        "-t", f"{duration_s:.3f}",
        "-frames:v", str(frames),    # exact frame count → exact duration
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", preset,
        "-crf", str(crf),
        "-movflags", "+faststart",
        str(dst),
    ]

    rc, _out, err = await _run(cmd)
    if rc != 0 or not dst.exists():
        raise _fail(f"normalize_clip({src.name})", cmd, rc, err)

    # ffmpeg exits 0 even when the source's decodable content is shorter than
    # its container says (common with Ogg/Theora originals) — verify the
    # promise, and loop the source once if it came up short.
    got = _float((await probe(dst)).get("duration"))
    tolerance = 1.5 / fps
    if got > 0.0 and got + tolerance < duration_s and not is_image and "-stream_loop" not in cmd:
        loop_cmd = ["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "-1"] + cmd[cmd.index("-i"):]
        rc, _out, err = await _run(loop_cmd)
        if rc != 0 or not dst.exists():
            raise _fail(f"normalize_clip({src.name}) [loop retry]", loop_cmd, rc, err)
        got = _float((await probe(dst)).get("duration"))
    if got > 0.0 and abs(got - duration_s) > max(tolerance, 0.25):
        raise RuntimeError(
            f"normalize_clip({src.name}): produced {got:.3f}s, wanted {duration_s:.3f}s — source too short or undecodable"
        )
    return dst


async def image_to_clip(
    src_image: Path,
    dst: Path,
    duration_s: float,
    **kw: Any,
) -> Path:
    """Thin wrapper: a still → a Ken Burns clip of ``duration_s``."""
    kw.pop("kenburns", None)
    return await normalize_clip(src_image, dst, duration_s, kenburns=True, **kw)


async def normalize_many(
    jobs: list[tuple[Path, Path, float]],
    *,
    concurrency: int = 3,
    **kw: Any,
) -> list[Path]:
    """Normalise ``(src, dst, duration_s)`` triples, at most ``concurrency`` at
    a time.  Results come back in input order; the first failure is raised."""
    if not jobs:
        return []
    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def _one(src: Path, dst: Path, dur: float) -> Path:
        async with sem:
            return await normalize_clip(src, dst, dur, **kw)

    results = await asyncio.gather(
        *(_one(Path(s), Path(d), float(t)) for s, d, t in jobs),
        return_exceptions=True,
    )
    for r in results:
        if isinstance(r, BaseException):
            raise r
    return [r for r in results if isinstance(r, Path)]


def jobs_from_edl(
    edl: list[EDLEntry],
    out_dir: Path,
    *,
    prefix: str = "clip",
) -> list[tuple[Path, Path, float]]:
    """Convenience: turn an EDL into ``normalize_many`` jobs (source clip →
    normalised clip → slot duration), preserving EDL order."""
    out_dir = Path(out_dir)
    return [
        (Path(e.clip), out_dir / f"{prefix}-{e.index:03d}.mp4", float(e.duration_s))
        for e in edl
    ]


# ─────────────────────────────────────────────────────────────────────
# Single-pass assembly
# ─────────────────────────────────────────────────────────────────────


async def assemble_simple(
    clips: list[Path],
    narration_wav: Path,
    out_path: Path,
    *,
    captions_ass: Path | None = None,
    fonts_dir: Path | None = None,
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: int = FPS,
    fade_s: float = 0.5,
    music_path: Path | None = None,
    music_gain_db: float = -18.0,
    crf: int = 19,
    preset: str = "fast",
) -> Path:
    """The safe fallback cut: ONE ffmpeg call producing the final essay.

    concat filter → fade in/out → libass caption burn → narration (+ optional
    ducked music) → single H.264/AAC encode.  ``clips`` must already be
    normalised (see :func:`normalize_clip`); narration is the master clock.
    """
    clips = [Path(c) for c in clips]
    narration_wav = Path(narration_wav)
    out_path = Path(out_path)
    n = len(clips)
    if n == 0:
        raise RuntimeError("assemble: no clips to assemble.")
    missing = [str(c) for c in clips if not c.exists()]
    if missing:
        raise RuntimeError(f"assemble: missing clip(s): {missing}")
    if not narration_wav.exists():
        raise RuntimeError(f"assemble: narration not found: {narration_wav}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Total video length = sum of the probed (already exact) clip durations.
    probes = await asyncio.gather(*(probe(c) for c in clips))
    total_s = sum(_float(p.get("duration")) for p in probes)

    # ── video chain ────────────────────────────────────────────────
    concat_inputs = "".join(f"[{i}:v]" for i in range(n))
    parts = [f"{concat_inputs}concat=n={n}:v=1:a=0[cv]"]

    vops: list[str] = []
    fade_s = float(fade_s)
    if fade_s > 0 and total_s > 2 * fade_s:
        out_st = max(0.0, total_s - fade_s)
        vops.append(f"fade=t=in:st=0:d={fade_s:.3f}")
        vops.append(f"fade=t=out:st={out_st:.3f}:d={fade_s:.3f}")
    if captions_ass is not None:
        sub = f"subtitles={ffmpeg_escape_filter_path(Path(captions_ass))}"
        if fonts_dir is not None:
            sub += f":fontsdir={ffmpeg_escape_filter_path(Path(fonts_dir))}"
        vops.append(sub)
    vops.append("format=yuv420p")
    parts.append(f"[cv]{','.join(vops)}[v]")

    # ── audio chain ────────────────────────────────────────────────
    narration_idx = n
    music_idx = n + 1
    if music_path is not None:
        # Narration is the master: apad it out to (at least) the video length
        # so amix's `duration=first` is governed by narration, never by the
        # music bed.  `whole_dur` is essential — a bare `apad` produces an
        # INFINITE stream, and `-shortest` does not reliably terminate a mux
        # fed by an endless filter output (ffmpeg writes until killed).
        pad = (
            f"apad=whole_dur={total_s:.3f}" if total_s > 0 else "apad=pad_dur=0"
        )
        parts.append(f"[{narration_idx}:a]{pad}[narr]")
        parts.append(f"[{music_idx}:a]volume={music_gain_db:g}dB[mus]")
        parts.append(
            "[narr][mus]amix=inputs=2:duration=first:dropout_transition=2[a]"
        )
        audio_map = "[a]"
    else:
        audio_map = f"{narration_idx}:a"

    filter_complex = ";".join(parts)

    cmd: list[str] = ["ffmpeg", "-y", "-loglevel", "error"]
    for clip in clips:
        cmd += ["-i", str(clip)]
    cmd += ["-i", str(narration_wav)]
    if music_path is not None:
        cmd += ["-i", str(Path(music_path))]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", audio_map,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", preset,
        "-crf", str(crf),
        "-r", str(fps),
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-movflags", "+faststart",
        "-shortest",
        str(out_path),
    ]

    rc, _out, err = await _run(cmd)
    if rc != 0 or not out_path.exists():
        raise _fail("assemble_simple", cmd, rc, err)
    _ = (width, height)  # canvas is fixed by the normalised inputs
    return out_path


# ─────────────────────────────────────────────────────────────────────
# QC
# ─────────────────────────────────────────────────────────────────────


async def verify_final(
    path: Path,
    expected_duration_s: float,
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
    tolerance_s: float = 1.0,
) -> tuple[bool, list[str], dict]:
    """QC gate on the rendered file.  Returns ``(ok, issues, probe_dict)``."""
    p = Path(path)
    issues: list[str] = []

    if not p.exists():
        return False, [f"file does not exist: {p}"], dict(_BLANK_PROBE)

    info = await probe(p)
    if not info["ok"]:
        issues.append("ffprobe could not read the file (or it has no video stream)")

    if not info["has_video"]:
        issues.append("no video stream")
    if not info["has_audio"]:
        issues.append("no audio stream")

    actual = _float(info.get("duration"))
    drift = abs(actual - float(expected_duration_s))
    if drift > float(tolerance_s):
        issues.append(
            f"duration {actual:.2f}s differs from expected "
            f"{float(expected_duration_s):.2f}s by {drift:.2f}s "
            f"(tolerance {float(tolerance_s):.2f}s)"
        )

    if info["width"] != width or info["height"] != height:
        issues.append(
            f"resolution {info['width']}x{info['height']} != {width}x{height}"
        )

    if info["vcodec"] != "h264":
        issues.append(f"video codec {info['vcodec']!r} is not h264")

    try:
        size = p.stat().st_size
    except OSError:
        size = 0
    if size < 50 * 1024:
        issues.append(f"file is only {size} bytes (< 50 KB)")

    return (not issues), issues, info


__all__ = [
    "WIDTH",
    "HEIGHT",
    "FPS",
    "IMAGE_SUFFIXES",
    "ffmpeg_escape_filter_path",
    "probe",
    "normalize_clip",
    "image_to_clip",
    "normalize_many",
    "jobs_from_edl",
    "assemble_simple",
    "verify_final",
]
