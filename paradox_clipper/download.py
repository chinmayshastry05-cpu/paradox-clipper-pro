"""Segment-only footage fetch — download ONLY the requested time range, never the
whole video. For a YouTube URL this uses yt-dlp --download-sections; for a local file
it slices with ffmpeg. Segments are cached, so a range is never fetched twice."""

import sys
import time
from pathlib import Path

from . import config
from .logutil import get_logger
from .util import is_url, run, source_key

log = get_logger("download")


def _seg_path(key, start, end):
    return config.SEGMENT_CACHE / f"{key}_{start:07.2f}-{end:07.2f}.mp4"


def get_segment(source, start, end, height=config.SEGMENT_MAX_HEIGHT):
    """Return a local mp4 containing only [start, end]. Cached by source+range."""
    key = source_key(source)
    out = _seg_path(key, start, end)
    if out.exists() and out.stat().st_size > 0:
        log.info("cache HIT — segment %.1f-%.1fs already downloaded (%s)",
                 start, end, out.name)
        return out
    config.SEGMENT_CACHE.mkdir(parents=True, exist_ok=True)

    if is_url(source):
        _download_yt_section(source, start, end, out, height)
    else:
        _slice_local(source, start, end, out)

    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"segment fetch produced no file for {start}-{end}")
    log.info("segment %.1f-%.1fs ready (%.1f MB) -> %s",
             start, end, out.stat().st_size / 1e6, out.name)
    return out


def _download_yt_section(url, start, end, out, height, attempts=4):
    """yt-dlp downloads ONLY this section of the stream (not the full video).

    Retries with backoff: YouTube/CDN throws transient DNS (getaddrinfo) and
    I/O errors, and concurrent workers make them more likely — a short wait and
    retry almost always succeeds."""
    section = f"*{start:.2f}-{end:.2f}"
    log.info("downloading ONLY %.1f-%.1fs via yt-dlp --download-sections (not full video)",
             start, end)
    last = None
    for attempt in range(1, attempts + 1):
        try:
            run([
                sys.executable, "-m", "yt_dlp",
                "-f", f"bv*[height<={height}]+ba/b[height<={height}]/bv*+ba/b",
                "--download-sections", section,
                "--force-keyframes-at-cuts",       # accurate in/out points
                "--merge-output-format", "mp4",
                "--retries", "10", "--fragment-retries", "10",
                "--socket-timeout", "30",
                "-o", str(out), str(url),
            ])
            return
        except Exception as e:                      # transient DNS / I/O / CDN
            last = e
            out.unlink(missing_ok=True)             # drop any partial file
            if attempt < attempts:
                wait = 4 * attempt
                log.warning("segment %.1f-%.1fs attempt %d/%d failed (%s); retry in %ds",
                            start, end, attempt, attempts,
                            str(e).splitlines()[0][:120], wait)
                time.sleep(wait)
    raise RuntimeError(f"segment {start:.1f}-{end:.1f}s failed after {attempts} attempts: {last}")


def get_full_video(source, height=config.SEGMENT_MAX_HEIGHT):
    """Fallback path: download the FULL video once (cached, shared across all
    clips) when ranged --download-sections fetches keep failing on a flaky CDN.
    yt-dlp's fragmented full download is far more resilient than ffmpeg pulling
    byte ranges. Still capped at `height` to limit bandwidth."""
    key = source_key(source)
    full_dir = config.CACHE_DIR / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    out = full_dir / f"{key}.mp4"
    if out.exists() and out.stat().st_size > 0:
        log.info("cache HIT — full video already downloaded (%s)", out.name)
        return out
    log.warning("ranged segment fetch unreliable — downloading FULL video ONCE "
                "(cached, shared across clips): %s", key)
    run([
        sys.executable, "-m", "yt_dlp",
        "-f", f"bv*[height<={height}]+ba/b[height<={height}]/bv*+ba/b",
        "--merge-output-format", "mp4",
        "--retries", "20", "--fragment-retries", "20", "--socket-timeout", "30",
        "-o", str(out), str(source),
    ])
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError("full-video fallback download produced no file")
    log.info("full video ready (%.1f MB) -> %s", out.stat().st_size / 1e6, out.name)
    return out


def slice_from_local(local_file, source, start, end):
    """Cut [start,end] out of an already-downloaded local file into the segment
    cache (same path get_segment would use), so downstream stages are identical."""
    out = _seg_path(source_key(source), start, end)
    if out.exists() and out.stat().st_size > 0:
        return out
    config.SEGMENT_CACHE.mkdir(parents=True, exist_ok=True)
    _slice_local(local_file, start, end, out)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"local slice produced no file for {start}-{end}")
    log.info("sliced %.1f-%.1fs from full video -> %s", start, end, out.name)
    return out


def _slice_local(source, start, end, out):
    """Local file: extract the range with ffmpeg (input seek -> 0-based output)."""
    log.info("slicing local file %.1f-%.1fs with ffmpeg", start, end)
    run([
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(source),
        "-t", f"{end - start:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "160k", str(out),
    ])
