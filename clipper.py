#!/usr/bin/env python3
"""
Paradox Clipper Pro — offline OpusClip-style short-form clipper.

Modular, bandwidth-minimal: it fetches a transcript WITHOUT downloading the full
video (YouTube captions first, else audio-only + Whisper), asks a local LLM for the
best clip ranges, then downloads ONLY those segments with yt-dlp --download-sections,
reframes to 9:16 with face tracking, burns cinematic captions, and exports.
Everything is cached; clips render concurrently.

See the `paradox_clipper` package for the individual stages.
"""

import argparse
import shutil
import sys

from paradox_clipper import config, pipeline
from paradox_clipper.logutil import get_logger, set_verbose

log = get_logger("cli")


def main():
    ap = argparse.ArgumentParser(
        prog="paradox-clipper",
        description="Paradox Clipper Pro — offline, bandwidth-minimal video clipper")
    ap.add_argument("--input", required=True, help="YouTube URL or local video path")
    ap.add_argument("--clips", type=int, default=5, help="number of clips (default 5)")
    ap.add_argument("--whisper-model", default=config.DEFAULT_WHISPER,
                    help="faster-whisper size (used only when captions are unavailable "
                         "and for per-segment caption timing)")
    ap.add_argument("--llm-model", default=config.DEFAULT_LLM, help="Ollama model")
    ap.add_argument("--output", default=None, help="output dir (default ./output)")
    v = ap.add_mutually_exclusive_group()
    v.add_argument("--vertical", dest="vertical", action="store_true")
    v.add_argument("--no-vertical", dest="vertical", action="store_false")
    ap.set_defaults(vertical=True)
    c = ap.add_mutually_exclusive_group()
    c.add_argument("--captions", dest="captions", action="store_true")
    c.add_argument("--no-captions", dest="captions", action="store_false")
    ap.set_defaults(captions=True)
    ap.add_argument("--caption-style", choices=["cinematic", "karaoke"],
                    default=config.DEFAULT_CAPTION_STYLE)
    ap.add_argument("--caption-script", choices=["hinglish", "native"],
                    default=config.DEFAULT_CAPTION_SCRIPT)
    ap.add_argument("--music", default=None, help="optional background music file")
    ap.add_argument("--broll", default=None, help="optional B-roll dir (overlay)")
    ap.add_argument("--dry-run", action="store_true",
                    help="select clips + write manifest, but download/render nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-render even if the output clip already exists")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    set_verbose(args.verbose)
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            log.error("%s not found on PATH", tool)
            sys.exit(1)

    try:
        pipeline.run(
            source=args.input, n_clips=args.clips, output_dir=args.output,
            whisper_model=args.whisper_model, llm_model=args.llm_model,
            vertical=args.vertical, captions_on=args.captions,
            caption_style=args.caption_style, caption_script=args.caption_script,
            music=args.music, broll=args.broll, dry_run=args.dry_run, force=args.force,
        )
    except (RuntimeError, KeyboardInterrupt) as e:
        log.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
