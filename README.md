# Paradox Clipper Pro

A personal, **fully offline** alternative to OpusClip. Give it a long video (YouTube
URL or local file) and it produces short, vertical (9:16), caption-burned highlight
clips ready for TikTok / Reels / Shorts. No cloud APIs, no subscription, no usage
limits — everything runs on your machine and GPU.

**Bandwidth-minimal:** it never downloads the full video. It reads the transcript,
picks the best moments, and downloads *only* those few seconds.

Built and tested on: Windows 11, RTX 4050 Laptop (6 GB VRAM), Python 3.11.

## Architecture (v2 — bandwidth-minimal, modular)

The pipeline **never downloads the full video**. It gets a transcript first, picks the
clip ranges, then downloads only those seconds. Code is split into independent modules
under `local_clipper/`:

| Module | Responsibility |
|--------|----------------|
| `transcript.py` | Get a transcript **without** a full download: YouTube captions first (`youtube-transcript-api`, no media at all), else audio-only download + Whisper. Cached by video id. |
| `analysis.py` | Send the transcript to a local **Ollama** LLM (JSON schema, dark-humor prompt); get back the best clip timestamp ranges. |
| `download.py` | Fetch **only** each selected range — `yt-dlp --download-sections "*start-end"` for URLs, ffmpeg slice for local files. Segments cached. |
| `captions.py` | Whisper the **short segment** (fast) for word-level timing (0-based), then author the `.ass` (Hinglish romanization + cinematic style). |
| `editing.py` | 9:16 face-anchored crop; optional B-roll overlay / background music resolution. |
| `export.py` | ffmpeg render of one segment → final short (crop, captions, optional music/B-roll). |
| `pipeline.py` | Orchestration: caching, concurrency, skip-existing, temp cleanup, per-stage logging. |
| `whisper_engine.py` | Shared faster-whisper model (lazy singleton; caption passes run serially on the GPU). |
| `config.py`, `util.py`, `logutil.py` | Config, ffmpeg/time/URL helpers, stage-prefixed logging. |

### Flow
1. Accept a YouTube URL (or local file).
2. **Transcript** — official captions preferred; audio-only + Whisper fallback. Never the full video.
3. **LLM** returns clip ranges as structured JSON (`start`/`end`, `title`, `hook`, `reason`).
4. **Download only** those ranges with `yt-dlp --download-sections` (multiple ranges → only those).
5. **Process each segment** — word-level captions (Hinglish, cinematic), 9:16 face-anchored reframe, optional B-roll + music, export.
6. **Cleanup** temp working files (caches are kept).

### Performance / caching
- Transcripts cached by **video id** (`.cache/transcripts/`) — never re-fetched.
- Downloaded **segments cached** by id+range (`.cache/segments/`) — never re-downloaded.
- Per-segment **word timings cached** (`.cache/segwords/`).
- **Skip** a clip whose output already exists (unless `--force`).
- Downloads run **concurrently**; renders run **concurrently** (Whisper stays serial on the 6 GB GPU).
- Detailed **per-stage logging** (`clipper.<stage>` loggers).

Other notes:
- LLM uses an Ollama **JSON schema** (structured outputs) + raised `num_ctx` so long transcripts don't overflow the default 2048-token context.
- Cinematic captions use **Segoe UI Black** (Hinglish/Latin) or **Nirmala UI** (native Devanagari).

## Prerequisites

These must be installed and on your PATH (they already were on the build machine):

- **Python 3.11** (3.9–3.12 fine)
- **ffmpeg / ffprobe** — https://www.gyan.dev/ffmpeg/builds/ (add `bin` to PATH)
- **NVIDIA GPU + recent driver** (for CUDA transcription; CPU fallback exists but is slow)
- **Ollama** — https://ollama.com/download , then pull a model:
  ```
  ollama pull qwen2.5:7b-instruct
  ```
  Ollama must be running (`ollama serve`, or just launch the Ollama app).

## Setup

```powershell
cd C:\Users\<you>\Documents\local-clipper
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` includes `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` — these ship the
CUDA runtime DLLs that `faster-whisper` (CTranslate2) needs on the GPU. Without them you
get `cublas64_12.dll is not found`. `clipper.py` adds these wheels' `bin` folders to the
DLL search path automatically at runtime, so no manual PATH editing is required.

> **Note:** `opencv-python` is pinned to `<5`. The `5.0.0.93` wheel currently on PyPI is
> broken (it has no `CascadeClassifier`). A copy of the Haar cascade
> (`haarcascade_frontalface_default.xml`) is also bundled in this folder as a fallback.

## Usage

```powershell
python clipper.py --input "<path or URL>" --clips 5 --whisper-model medium --llm-model qwen2.5:7b-instruct --vertical --captions
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--input` | *(required)* | Local video path **or** a URL (yt-dlp downloads it) |
| `--clips` | `5` | Number of highlight clips to produce |
| `--whisper-model` | `medium` | faster-whisper size (`tiny`/`base`/`small`/`medium`/`large-v3`) |
| `--llm-model` | `qwen2.5:7b-instruct` | Any pulled Ollama model |
| `--output` | `./output` | Output directory |
| `--vertical` / `--no-vertical` | vertical | 9:16 smart crop, or keep source aspect |
| `--captions` / `--no-captions` | captions | Burn animated captions, or none |
| `--keep-temp` | off | Keep the temp working dir (downloaded video, wav) for debugging |

### Examples

```powershell
# From a URL, 3 clips, defaults
python clipper.py --input "https://www.youtube.com/watch?v=..." --clips 3

# Local file, horizontal, no captions, faster tiny model
python clipper.py --input "C:\videos\talk.mp4" --no-vertical --no-captions --whisper-model small

# Bigger, higher-quality LLM for highlight selection
python clipper.py --input "talk.mp4" --llm-model mistral:7b
```

## Output

```
output/
  clip_1.mp4        1080x1920 H.264 + AAC, captions burned in
  clip_1.ass        the subtitle file used for clip 1
  clip_2.mp4
  ...
  manifest.json     [{ file, title, hook, start, end, duration }, ...]
```

## VRAM notes (6 GB cards)

- `medium` in float16 fits comfortably alongside nothing else on the GPU. The Whisper
  model is released before Ollama runs so the 7B LLM has room.
- If you ever hit an out-of-memory error, use a smaller `--whisper-model` (e.g. `small`)
  or a smaller `--llm-model` (e.g. `qwen3:4b`, `llama3.2:3b`).
- `faster-whisper` will automatically fall back to CPU/int8 if CUDA can't initialize
  (much slower, but it won't crash).

## Troubleshooting

- **`cublas64_12.dll is not found`** — the `nvidia-cublas-cu12` / `nvidia-cudnn-cu12`
  wheels aren't installed. `pip install -r requirements.txt`.
- **`Could not reach Ollama`** — start Ollama (app or `ollama serve`) and confirm the
  model is pulled (`ollama list`).
- **`module 'cv2' has no attribute 'CascadeClassifier'`** — you have the broken
  `opencv-python 5.0.0.93`. `pip install "opencv-python<5"`.
- **No faces detected / odd crop** — Haar detection is frontal-face only; it falls back
  to a center crop. Use `--no-vertical` if the source is already vertical.
