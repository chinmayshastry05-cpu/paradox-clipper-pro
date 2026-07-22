"""Editing: reframe a segment to vertical 9:16 with a face-anchored crop, and
optional B-roll / background music hooks. Operates on the short segment file."""

import os
from pathlib import Path

from . import config
from .logutil import get_logger
from .util import ffprobe_dims

log = get_logger("editing")


def face_center(video, samples=12):
    """Average detected face center across sampled frames of the whole segment.
    Returns (cx, cy) or None (fall back to a center crop)."""
    import cv2

    cascade_path = os.path.join(cv2.data.haarcascades,
                                "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        cascade_path = str(config.CASCADE_FALLBACK) if config.CASCADE_FALLBACK.exists() else ""
    cascade = cv2.CascadeClassifier(cascade_path) if cascade_path else None
    if cascade is None or cascade.empty():
        log.warning("face cascade unavailable — center crop")
        return None

    cap = cv2.VideoCapture(str(video))
    total_ms = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1.0, cap.get(cv2.CAP_PROP_FPS))) * 1000.0
    xs, ys = [], []
    for i in range(samples):
        cap.set(cv2.CAP_PROP_POS_MSEC, total_ms * (i + 0.5) / samples)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60)):
            xs.append(x + w / 2.0)
            ys.append(y + h / 2.0)
    cap.release()
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _crop_box(src_w, src_h, center):
    target = config.OUT_W / config.OUT_H
    if src_w / src_h > target:
        cw, ch = int(round(src_h * target)), src_h
    else:
        cw, ch = src_w, int(round(src_w / target))
    cw = min(cw, src_w) & ~1
    ch = min(ch, src_h) & ~1
    cx = center[0] if center else src_w / 2.0
    cy = center[1] if center else src_h / 2.0
    x = max(0, min(int(round(cx - cw / 2.0)), src_w - cw)) & ~1
    y = max(0, min(int(round(cy - ch / 2.0)), src_h - ch)) & ~1
    return cw, ch, x, y


def vertical_filters(segment_path, vertical=True):
    """Return (out_w, out_h, [vf strings]) for a face-anchored 9:16 crop, or the
    source untouched when vertical is False."""
    src_w, src_h = ffprobe_dims(segment_path)
    if not vertical:
        return src_w, src_h, []
    center = face_center(segment_path)
    if center is None:
        log.info("no face found — center crop")
    cw, ch, cx, cy = _crop_box(src_w, src_h, center)
    return (config.OUT_W, config.OUT_H,
            [f"crop={cw}:{ch}:{cx}:{cy}", f"scale={config.OUT_W}:{config.OUT_H}"])


def pick_broll(broll_dir):
    """Optional B-roll: return a clip path from a directory, or None.
    (Overlay/insertion is applied in export.py when a path is returned.)"""
    if not broll_dir:
        return None
    d = Path(broll_dir)
    clips = sorted([p for p in d.glob("*") if p.suffix.lower() in
                    (".mp4", ".mov", ".mkv", ".webm")]) if d.is_dir() else []
    if not clips:
        log.info("B-roll requested but none found in %s — skipping", broll_dir)
        return None
    log.info("B-roll available: %s", clips[0].name)
    return clips[0]


def resolve_music(music_path):
    """Optional background music: validate path or None."""
    if not music_path:
        return None
    p = Path(music_path)
    if p.exists():
        log.info("background music: %s", p.name)
        return p
    log.warning("music file not found: %s — skipping", music_path)
    return None
