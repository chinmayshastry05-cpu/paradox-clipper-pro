"""Shared faster-whisper model — loaded once, reused across transcript + caption passes.

The GPU model is a lazy singleton. Caption passes must call these serially (the model
is not safe for concurrent GPU calls, and 6 GB VRAM has no room for two copies).
"""

import gc
import os
from pathlib import Path

from .logutil import get_logger

log = get_logger("whisper")

_model = None
_model_size = None


def _add_cuda_dll_dirs():
    """Make ctranslate2 find cuBLAS/cuDNN shipped as nvidia-*-cu12 wheels (Windows)."""
    if not hasattr(os, "add_dll_directory"):
        return
    import site
    roots = list(site.getsitepackages())
    if hasattr(site, "getusersitepackages"):
        roots.append(site.getusersitepackages())
    for root in roots:
        nvidia = Path(root) / "nvidia"
        if not nvidia.exists():
            continue
        for bindir in nvidia.glob("*/bin"):
            try:
                os.add_dll_directory(str(bindir))
                os.environ["PATH"] = str(bindir) + os.pathsep + os.environ.get("PATH", "")
            except OSError:
                pass


def get_model(size):
    """Return the shared WhisperModel, (re)loading if the requested size changed."""
    global _model, _model_size
    if _model is not None and _model_size == size:
        return _model
    if _model is not None:
        free_model()
    _add_cuda_dll_dirs()
    from faster_whisper import WhisperModel
    try:
        log.info("loading faster-whisper '%s' on cuda (float16)...", size)
        _model = WhisperModel(size, device="cuda", compute_type="float16")
    except Exception as e:  # noqa: BLE001 - want any CUDA failure to fall back
        log.warning("CUDA load failed (%s: %s); falling back to CPU/int8",
                    type(e).__name__, e)
        _model = WhisperModel(size, device="cpu", compute_type="int8")
    _model_size = size
    return _model


def transcribe(wav_path, size, word_timestamps):
    """Transcribe a wav. Returns (segments, words, language).
    segments: [{start,end,text}]; words: [{start,end,word}] (empty if not requested)."""
    model = get_model(size)
    seg_iter, info = model.transcribe(
        str(wav_path), word_timestamps=word_timestamps, vad_filter=True)
    segments, words = [], []
    for s in seg_iter:  # generator — iterate to actually run inference
        segments.append({"start": s.start, "end": s.end, "text": (s.text or "").strip()})
        if word_timestamps and s.words:
            for w in s.words:
                words.append({"start": w.start, "end": w.end, "word": w.word})
    log.info("transcribed %d segments, %d words (lang=%s)",
             len(segments), len(words), info.language)
    return segments, words, info.language


def free_model():
    """Release the model + VRAM (call before a different GPU stage, e.g. Ollama)."""
    global _model, _model_size
    if _model is None:
        return
    del _model
    _model = None
    _model_size = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    log.debug("whisper model freed")
