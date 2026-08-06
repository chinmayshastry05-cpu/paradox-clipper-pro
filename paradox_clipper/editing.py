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
    xs, ys, read, frames_with_face = [], [], 0, 0
    for i in range(samples):
        cap.set(cv2.CAP_PROP_POS_MSEC, total_ms * (i + 0.5) / samples)
        ok, frame = cap.read()
        if not ok:
            continue
        read += 1
        fh = frame.shape[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # A real talking-head face is sizable; ignore tiny Haar false-positives.
        big = [(x, y, w, h) for (x, y, w, h)
               in cascade.detectMultiScale(gray, 1.1, 6, minSize=(int(fh * 0.10),) * 2)
               if h >= 0.10 * fh]
        if big:
            frames_with_face += 1
            x, y, w, h = max(big, key=lambda f: f[2] * f[3])   # largest face
            xs.append(x + w / 2.0)
            ys.append(y + h / 2.0)
    cap.release()
    # Only trust "face-anchored" when a sizable face is present in a solid share of
    # frames (a real speaker). Otherwise -> saliency crop.
    if xs and frames_with_face >= max(3, int(read * 0.34)):
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    return None


def _crop_size(src_w, src_h):
    """The 9:16 crop rectangle size (even ints) for a given source."""
    target = config.OUT_W / config.OUT_H
    if src_w / src_h > target:
        cw, ch = int(round(src_h * target)), src_h
    else:
        cw, ch = src_w, int(round(src_w / target))
    return (min(cw, src_w) & ~1), (min(ch, src_h) & ~1)


def _place(cx, cy, cw, ch, src_w, src_h):
    """Clamp a crop of size cw x ch centered at (cx, cy) into the frame."""
    x = max(0, min(int(round(cx - cw / 2.0)), src_w - cw)) & ~1
    y = max(0, min(int(round(cy - ch / 2.0)), src_h - ch)) & ~1
    return x, y


def salient_crop(video, src_w, src_h, cw, ch, samples=16):
    """Object/saliency-aware crop for footage with no clear face (screen recordings,
    product demos, top-down shots). Builds an energy map from edge/gradient detail
    plus frame-to-frame motion, then slides the 9:16 window to where the energy —
    i.e. the actual subject (a phone, a screen, moving hands) — is densest.
    Returns (x, y) or None."""
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    cap = cv2.VideoCapture(str(video))
    fps = max(1.0, cap.get(cv2.CAP_PROP_FPS))
    total_ms = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps) * 1000.0
    max_w = 360
    grays = []
    for i in range(samples):
        cap.set(cv2.CAP_PROP_POS_MSEC, total_ms * (i + 0.5) / samples)
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        if w > max_w:
            fr = cv2.resize(fr, (max_w, max(1, int(h * max_w / w))))
        grays.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype("float32"))
    cap.release()
    if not grays:
        return None
    H, W = grays[0].shape
    E = np.zeros((H, W), "float32")
    for g in grays:                       # edge/detail energy (busy regions)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        E += cv2.magnitude(gx, gy)
    for a, b in zip(grays, grays[1:]):    # motion energy (where the action is)
        E += np.abs(a - b) * 3.0
    sx, sy = src_w / W, src_h / H
    cwp, chp = max(1, int(round(cw / sx))), max(1, int(round(ch / sy)))

    def best_start(prof, win):
        n = len(prof)
        win = min(win, n)
        c = np.concatenate([[0.0], np.cumsum(prof)])
        sums = c[win:] - c[:-win]
        return int(np.argmax(sums)) if len(sums) else 0

    xi = best_start(E.sum(axis=0), cwp)
    yi = best_start(E.sum(axis=1), chp)
    # window-start -> window-center, back to source scale
    cx = (xi + cwp / 2.0) * sx
    cy = (yi + chp / 2.0) * sy
    return _place(cx, cy, cw, ch, src_w, src_h)


def vertical_filters(segment_path, vertical=True):
    """Return (out_w, out_h, [vf strings]) for a 9:16 crop that tracks the subject:
    face-anchored when a face is present, else saliency/motion-anchored, else center.
    Source is returned untouched when vertical is False."""
    src_w, src_h = ffprobe_dims(segment_path)
    if not vertical:
        return src_w, src_h, []
    cw, ch = _crop_size(src_w, src_h)
    center = face_center(segment_path)
    if center is not None:
        x, y = _place(center[0], center[1], cw, ch, src_w, src_h)
    else:
        box = salient_crop(segment_path, src_w, src_h, cw, ch)
        if box is not None:
            x, y = box
            log.info("no face — saliency/motion crop")
        else:
            x, y = _place(src_w / 2.0, src_h / 2.0, cw, ch, src_w, src_h)
            log.info("no face, no saliency — center crop")
    return (config.OUT_W, config.OUT_H,
            [f"crop={cw}:{ch}:{x}:{y}", f"scale={config.OUT_W}:{config.OUT_H}"])


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
