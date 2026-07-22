"""Segment-only footage fetch — download ONLY the requested time range, never the
whole video. For a YouTube URL this uses yt-dlp --download-sections; for a local file
it slices with ffmpeg. Segments are cached, so a range is never fetched twice."""

import sys
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


def _download_yt_section(url, start, end, out, height):
    """yt-dlp downloads ONLY this section of the stream (not the full video)."""
    section = f"*{start:.2f}-{end:.2f}"
    log.info("downloading ONLY %.1f-%.1fs via yt-dlp --download-sections (not full video)",
             start, end)
    run([
        sys.executable, "-m", "yt_dlp",
        "-f", f"bv*[height<={height}]+ba/b[height<={height}]/bv*+ba/b",
        "--download-sections", section,
        "--force-keyframes-at-cuts",       # accurate in/out points
        "--merge-output-format", "mp4",
        "-o", str(out), str(url),
    ])


def _slice_local(source, start, end, out):
    """Local file: extract the range with ffmpeg (input seek -> 0-based output)."""
    log.info("slicing local file %.1f-%.1fs with ffmpeg", start, end)
    run([
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(source),
        "-t", f"{end - start:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "160k", str(out),
    ])
