"""Pipeline orchestration: transcript -> AI analysis -> segment download -> edit ->
export. Caches at every stage, downloads only the needed ranges, processes clips
concurrently, skips work already done, and cleans up temp files."""

import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import analysis, captions, config, download, editing, export, transcript, whisper_engine
from .logutil import get_logger
from .util import fmt_time, ffprobe_duration, slugify, source_key

log = get_logger("pipeline")


def run(source, n_clips=5, output_dir=None, whisper_model=config.DEFAULT_WHISPER,
        llm_model=config.DEFAULT_LLM, vertical=True, captions_on=True,
        caption_style=config.DEFAULT_CAPTION_STYLE,
        caption_script=config.DEFAULT_CAPTION_SCRIPT,
        music=None, broll=None, dry_run=False, force=False):
    config.ensure_dirs()
    outdir = Path(output_dir).resolve() if output_dir else config.DEFAULT_OUTPUT
    outdir.mkdir(parents=True, exist_ok=True)
    key = source_key(source)
    log.info("=== pipeline start | source=%s | id=%s | clips=%d ===", source, key, n_clips)

    # ---- 1. transcript (no full-video download) --------------------------
    segments, language, method = transcript.get_transcript(source, whisper_model)
    duration = max((s["end"] for s in segments), default=0.0)
    log.info("transcript via %s | ~%.0fs | lang=%s", method, duration, language)

    # ---- 2. AI analysis: pick clip ranges --------------------------------
    clips = analysis.select_clips(segments, n_clips, duration, llm_model)
    log.info("selected %d clip range(s)", len(clips))
    for c in clips:
        c["clock"] = f"{fmt_time(c['start'])}-{fmt_time(c['end'])}"
        c["base"] = None  # filled below

    if dry_run:
        (outdir / "manifest_dryrun.json").write_text(
            json.dumps(clips, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("dry-run: wrote manifest_dryrun.json (no download/render)")
        return clips

    music = editing.resolve_music(music)
    broll = editing.pick_broll(broll)

    # assign output names + skip-existing
    todo = []
    for i, c in enumerate(clips, 1):
        base = f"{i:02d}-{slugify(c['title'])}"
        c["base"] = base
        c["file"] = f"{base}.mp4"
        if not force and (outdir / c["file"]).exists():
            log.info("skip clip %d — already exists (%s)", i, c["file"])
            c["_skipped"] = True
        else:
            todo.append(c)

    # ---- 3. download ONLY the needed segments (concurrent, cached) -------
    seg_paths = {}
    if todo:
        log.info("downloading %d segment(s) concurrently (workers=%d)...",
                 len(todo), config.DOWNLOAD_WORKERS)
        with ThreadPoolExecutor(max_workers=config.DOWNLOAD_WORKERS) as ex:
            futs = {ex.submit(download.get_segment, source, c["start"], c["end"]): c
                    for c in todo}
            for fut in as_completed(futs):
                c = futs[fut]
                try:
                    seg_paths[id(c)] = fut.result()
                except Exception as e:  # noqa: BLE001
                    log.error("download failed for clip '%s' (%s): %s",
                              c["title"], c["clock"], e)

    # ---- 4. prepare each clip (words + crop + captions) — GPU work serial -
    prepared = []
    for c in todo:
        seg = seg_paths.get(id(c))
        if seg is None:
            continue
        try:
            seg_dur = ffprobe_duration(seg)
            out_w, out_h, vf = editing.vertical_filters(seg, vertical)
            ass_name = None
            if captions_on:
                words = captions.segment_words(seg, source, c["start"], c["end"],
                                               whisper_model)  # serial GPU
                ass_name = f"{c['base']}.ass"
                captions.write_ass(words, seg_dur, out_w, out_h,
                                   outdir / ass_name, caption_style, caption_script)
            prepared.append((c, seg, vf, ass_name))
        except Exception as e:  # noqa: BLE001
            log.error("prep failed for clip '%s': %s", c["title"], e)
    whisper_engine.free_model()  # done with GPU

    # ---- 5. render (concurrent, CPU-bound ffmpeg) ------------------------
    if prepared:
        log.info("rendering %d clip(s) concurrently (workers=%d)...",
                 len(prepared), config.RENDER_WORKERS)
        with ThreadPoolExecutor(max_workers=config.RENDER_WORKERS) as ex:
            futs = {ex.submit(export.render, seg, outdir, c["file"], vf, ass_name,
                              music, broll): c
                    for (c, seg, vf, ass_name) in prepared}
            for fut in as_completed(futs):
                c = futs[fut]
                try:
                    fut.result()
                    c["_rendered"] = True
                except Exception as e:  # noqa: BLE001
                    log.error("render failed for clip '%s': %s", c["title"], e)

    # ---- 6. manifest + cleanup ------------------------------------------
    manifest = [{
        "file": c["file"], "title": c["title"], "hook": c["hook"],
        "reason": c.get("reason", ""), "clock": c["clock"],
        "start": c["start"], "end": c["end"],
        "duration": round(c["end"] - c["start"], 2),
    } for c in clips if (outdir / c["file"]).exists()]
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    _cleanup_temp()
    log.info("=== done | %d clip(s) in %s ===", len(manifest), outdir)
    for m in manifest:
        log.info("  %s | %s (%ss)", m["file"], m["title"], m["duration"])
    return manifest


def _cleanup_temp():
    """Remove stray temp working dirs (segment/word wavs). Caches are kept."""
    tmp = Path(tempfile.gettempdir())
    for d in tmp.glob("clipper_*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
    log.debug("temp working dirs cleaned")
