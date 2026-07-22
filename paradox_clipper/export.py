"""Export: render one already-downloaded segment into the final short.
Applies the 9:16 crop/scale, burns captions, and optionally overlays B-roll and
mixes background music. The segment is 0-based, so no seeking is needed and the
0-based .ass caption track lines up exactly."""

from pathlib import Path

from . import config
from .logutil import get_logger
from .util import has_audio, run

log = get_logger("export")


def render(segment_path, outdir, out_name, vf_parts, ass_name=None,
           music=None, broll=None):
    """Render segment_path -> outdir/out_name. Returns the output Path.
    vf_parts: video-filter chain (crop/scale); ass_name: bare .ass filename in outdir."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    vchain = list(vf_parts)
    if ass_name:
        vchain.append(f"ass={ass_name}")     # bare name; we run with cwd=outdir
    seg_has_audio = has_audio(segment_path)

    cmd = ["ffmpeg", "-y", "-i", str(segment_path)]
    idx = 1
    broll_idx = music_idx = None
    if broll:
        cmd += ["-i", str(broll)]
        broll_idx = idx
        idx += 1
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(music)]   # loop music to clip length
        music_idx = idx
        idx += 1

    if broll_idx is None and music_idx is None:
        # simple path: single input, -vf
        if vchain:
            cmd += ["-vf", ",".join(vchain)]
        _add_codecs(cmd, seg_has_audio)
    else:
        fc = []
        vlabel = "0:v"
        if vchain:
            fc.append(f"[0:v]{','.join(vchain)}[vbase]")
            vlabel = "vbase"
        if broll_idx is not None:
            fc.append(f"[{broll_idx}:v]scale={config.OUT_W // 3}:-2[bov]")
            fc.append(f"[{vlabel}][bov]overlay=(W-w)/2:70[vout]")
            vlabel = "vout"
        # audio
        if music_idx is not None and seg_has_audio:
            fc.append(f"[0:a]volume=1[a0]")
            fc.append(f"[{music_idx}:a]volume=0.16[a1]")
            fc.append("[a0][a1]amix=inputs=2:duration=first:dropout_transition=0[aout]")
            alabel = "aout"
        elif music_idx is not None:
            fc.append(f"[{music_idx}:a]volume=0.5[aout]")
            alabel = "aout"
        else:
            alabel = "0:a" if seg_has_audio else None
        cmd += ["-filter_complex", ";".join(fc), "-map", f"[{vlabel}]"]
        if alabel:
            cmd += ["-map", f"[{alabel}]" if alabel.endswith("out") else alabel]
        _add_codecs(cmd, alabel is not None)

    cmd.append(out_name)
    log.info("rendering -> %s", out_name)
    run(cmd, cwd=str(outdir))
    out = outdir / out_name
    log.info("done %s (%.1f MB)", out_name, out.stat().st_size / 1e6)
    return out


def _add_codecs(cmd, with_audio):
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if with_audio:
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    else:
        cmd += ["-an"]
