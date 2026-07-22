"""paradox_clipper — Paradox Clipper Pro's modular, bandwidth-minimal clip pipeline.

Modules (each independent, single responsibility):
  config          paths, defaults, constants
  logutil         stage-prefixed logging
  util            ffmpeg/ffprobe helpers, time + URL parsing
  whisper_engine  shared faster-whisper model (lazy singleton)
  transcript      get a transcript WITHOUT downloading the full video
  analysis        LLM highlight selection -> clip timestamp ranges
  download        segment-only fetch (yt-dlp --download-sections / local slice)
  captions        per-segment word timings + .ass caption authoring
  editing         9:16 face-anchored reframe, optional b-roll/music
  export          ffmpeg render of one segment -> final short
  pipeline        orchestration, caching, concurrency, cleanup
"""

__version__ = "2.0.0"
