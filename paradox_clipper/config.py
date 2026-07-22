"""Central configuration: paths, defaults, constants."""

from pathlib import Path

# Project root = the folder containing the `paradox_clipper` package.
ROOT = Path(__file__).resolve().parent.parent

# Cache layout (never re-download / re-transcribe what we already have).
CACHE_DIR = ROOT / ".cache"
TRANSCRIPT_CACHE = CACHE_DIR / "transcripts"      # transcript_<videoid>.json
SEGMENT_CACHE = CACHE_DIR / "segments"            # <videoid>_<start>-<end>.mp4
SEGWORDS_CACHE = CACHE_DIR / "segwords"           # <videoid>_<start>-<end>.json

DEFAULT_OUTPUT = ROOT / "output"

# Bundled Haar cascade fallback (some opencv wheels don't ship it).
CASCADE_FALLBACK = ROOT / "haarcascade_frontalface_default.xml"

# Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_LLM = "qwen2.5:7b-instruct"

# Whisper
DEFAULT_WHISPER = "medium"

# Output video
OUT_W, OUT_H = 1080, 1920           # 9:16
SEGMENT_MAX_HEIGHT = 720            # download at most 720p (plenty for vertical)

# Clip length guardrails (seconds)
MIN_LEN = 18.0
MAX_LEN = 90.0

# Concurrency
DOWNLOAD_WORKERS = 4               # network-bound, safe to parallelize
RENDER_WORKERS = 2                # ffmpeg is CPU-heavy; keep modest
# Note: whisper caption passes run SERIALLY (single shared GPU model).

DEFAULT_CAPTION_STYLE = "cinematic"
DEFAULT_CAPTION_SCRIPT = "hinglish"


def ensure_dirs():
    for d in (CACHE_DIR, TRANSCRIPT_CACHE, SEGMENT_CACHE, SEGWORDS_CACHE):
        d.mkdir(parents=True, exist_ok=True)
