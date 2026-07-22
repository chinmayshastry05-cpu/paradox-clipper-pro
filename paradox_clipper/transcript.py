"""Transcript extraction — WITHOUT ever downloading the full video.

Order of preference:
  1. Official / auto YouTube captions (no media download at all).
  2. Audio-only download (yt-dlp bestaudio) + Whisper.
  3. Local file -> extract audio + Whisper.

Result is cached by video id, so a transcript is never fetched twice.
The transcript here is phrase-level (enough for the LLM to pick clips); word-level
timing for burned captions is produced later, per short segment, in captions.py.
"""

import json
import sys
import tempfile
from pathlib import Path

from . import config, whisper_engine
from .logutil import get_logger
from .util import extract_wav16k, is_url, run, source_key, youtube_id

log = get_logger("transcript")


def _cache_path(key):
    return config.TRANSCRIPT_CACHE / f"transcript_{key}.json"


def get_transcript(source, whisper_model=config.DEFAULT_WHISPER, prefer_captions=True):
    """Return (segments, language, method). segments: [{start,end,text}].

    Cached by video id / source key — never re-fetched."""
    key = source_key(source)
    cache = _cache_path(key)
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        log.info("cache HIT — transcript for '%s' (%d segs, method=%s)",
                 key, len(data["segments"]), data.get("method"))
        return data["segments"], data.get("language"), data.get("method", "cache")

    log.info("cache MISS — building transcript for '%s'", key)
    segments, language, method = None, None, None

    if is_url(source) and prefer_captions:
        vid = youtube_id(source)
        if vid:
            segments, language = _youtube_captions(vid)
            if segments:
                method = "youtube-captions"

    if not segments:
        segments, language = _audio_whisper(source, whisper_model)
        method = "audio-whisper"

    if not segments:
        raise RuntimeError("could not obtain a transcript (no captions, no speech?)")

    config.TRANSCRIPT_CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(
        {"segments": segments, "language": language, "method": method},
        ensure_ascii=False), encoding="utf-8")
    log.info("transcript ready (%d segs, method=%s, lang=%s) -> cached",
             len(segments), method, language)
    return segments, language, method


def _youtube_captions(video_id):
    """Fetch YouTube captions via youtube-transcript-api. No media download.
    Returns (segments, language) or (None, None)."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except Exception:
        log.warning("youtube-transcript-api not installed; skipping caption fetch")
        return None, None
    try:
        api = YouTubeTranscriptApi()
        # Prefer manually-created; accept auto-generated; any language is fine
        # (the LLM prompt handles non-English transcripts).
        fetched = api.fetch(video_id, languages=["hi", "en", "en-US", "en-IN"],
                            preserve_formatting=False)
        raw = fetched.to_raw_data()
        lang = getattr(fetched, "language_code", None)
    except Exception as e:  # NoTranscriptFound / disabled / IP-blocked / API drift
        log.info("no usable YouTube captions (%s: %s)", type(e).__name__, e)
        return None, None
    segments = []
    for item in raw:
        start = float(item["start"])
        dur = float(item.get("duration", 0.0))
        text = (item.get("text") or "").replace("\n", " ").strip()
        if text:
            segments.append({"start": start, "end": start + dur, "text": text})
    if not segments:
        return None, None
    log.info("fetched %d caption lines from YouTube (lang=%s) — no video downloaded",
             len(segments), lang)
    return segments, lang


def _audio_whisper(source, whisper_model):
    """Audio-only path: download just the audio track (or use local file), transcribe."""
    workdir = Path(tempfile.mkdtemp(prefix="clipper_audio_"))
    try:
        if is_url(source):
            log.info("no captions — downloading AUDIO ONLY (not the video) with yt-dlp")
            out_tmpl = str(workdir / "audio.%(ext)s")
            run([sys.executable, "-m", "yt_dlp", "-f", "bestaudio/best",
                 "-x", "--audio-format", "wav", "--postprocessor-args",
                 "ffmpeg:-ac 1 -ar 16000", "-o", out_tmpl, str(source)])
            cands = sorted(workdir.glob("audio.*"))
            if not cands:
                raise RuntimeError("yt-dlp produced no audio file")
            wav = cands[0]
        else:
            log.info("local file — extracting audio for transcription")
            wav = extract_wav16k(source, workdir / "audio.wav")
        segments, _, language = whisper_engine.transcribe(
            wav, whisper_model, word_timestamps=False)
        return segments, language
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
