"""AI analysis — send the transcript to a local Ollama LLM, get back the best clip
timestamp ranges as structured JSON. No media touched here."""

import json

import requests

from . import config
from .logutil import get_logger
from .util import fmt_time, parse_time

log = get_logger("analysis")

SYS_PROMPT = (
    "You are a viral short-form video editor (TikTok / Reels / YouTube Shorts) who "
    "clips long podcasts and panels into scroll-stopping moments. You have a proven "
    "instinct for what makes people stop, laugh, gasp, and share. You optimize for "
    "RETENTION and SHARES, not for summarizing."
)

CLIPS_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "title": {"type": "string"},
                    "hook": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["start", "end", "title", "hook", "reason"],
            },
        }
    },
    "required": ["clips"],
}


def _condense(segments, block=18.0, max_chars=9000):
    lines, cs, ce, ct = [], None, None, []
    for s in segments:
        if cs is None:
            cs = s["start"]
        ce = s["end"]
        ct.append(s["text"])
        if ce - cs >= block:
            lines.append(f"[{cs:.0f}-{ce:.0f}] {' '.join(ct)}")
            cs, ct = None, []
    if ct:
        lines.append(f"[{cs:.0f}-{ce:.0f}] {' '.join(ct)}")
    return "\n".join(lines)[:max_chars]


def _user_prompt(segments, n, duration):
    transcript = _condense(segments)
    return (
        f"The video is {duration:.0f} seconds long ({duration/60:.1f} minutes). "
        f"Below is its transcript; each line starts with its [start-end] time range "
        f"in SECONDS.\n\n"
        f"Find the {n} MOST VIRAL, non-overlapping moments — the clips most likely "
        f"to blow up as standalone Shorts/Reels.\n\n"
        f"WHAT MAKES A MOMENT VIRAL (pick for these — lean HARD into DARK HUMOR):\n"
        f"- DARK / edgy / savage humor: brutal roasts, morbid jokes, self-deprecating "
        f"pain played for laughs, offensive-but-funny, deadpan bleak takes, awkward cringe\n"
        f"- Jokes with a clear setup and a punchline that lands\n"
        f"- Emotional peaks: shocking confessions, vulnerability twisted into comedy, conflict, hot takes\n"
        f"- Taboo / spicy / personal topics people can't help clicking\n"
        f"- Surprising stories, bold claims, drama, someone getting destroyed in a roast\n"
        f"- A COMPLETE beat: setup + payoff, makes sense with zero context\n"
        f"Prioritize the funniest, darkest, most savage moments over wholesome ones.\n"
        f"AVOID: intros, greetings, logistics, generic praise, rambling, anything boring.\n\n"
        f"LENGTH: choose the natural length of each moment, anywhere from 18 to 75 "
        f"seconds. Start right before the setup; end right after the payoff. Don't pad.\n\n"
        f"TITLE — this is what earns the click. Use this exact two-part shape:\n"
        f"  [HOOK]: [SPECIFIC]!   (or end with ?)\n"
        f"  - [HOOK] = a 1-4 word provocative/curiosity teaser (the emotional angle).\n"
        f"  - [SPECIFIC] = who + the concrete thing that actually happens in THIS clip, "
        f"using the person's REAL NAME from the transcript (never 'he/someone/a guy').\n"
        f"  *** Build both parts ONLY from what is actually said in THIS segment. Do NOT "
        f"reuse wording from these instructions. Do NOT invent topics not in the clip. ***\n"
        f"  NEVER pick pure intros/greetings/'X appears on stage'.\n"
        f"  Clean English; no raw transcript words/numbers/timestamps in the title. Don't "
        f"sanitize spicy/taboo topics — name them bluntly.\n\n"
        f"HOOK: one scroll-stopping line. REASON: one short phrase on why it will pop.\n\n"
        f"*** LANGUAGE: the transcript may be Hindi or another language, but ALL of your "
        f"output — title, hook, reason — MUST be written in ENGLISH using the Latin "
        f"alphabet. Translate the meaning to English; transliterate people's names to "
        f"Latin letters. NEVER output Devanagari or any non-Latin script. ***\n\n"
        f"Return ONLY JSON: "
        f'{{"clips": [{{"start": <number>, "end": <number>, "title": "...", '
        f'"hook": "...", "reason": "..."}}]}}\n'
        f"start/end are PLAIN SECONDS (e.g. 1234.5), NOT clock timestamps, within "
        f"0 and {duration:.0f}.\n\nTRANSCRIPT:\n{transcript}"
    )


def select_clips(segments, n, duration, model=config.DEFAULT_LLM):
    """Ask the LLM for the n best clips. Returns validated list of clip dicts."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": _user_prompt(segments, n, duration)},
        ],
        "format": {**CLIPS_SCHEMA, "properties": {"clips": {
            **CLIPS_SCHEMA["properties"]["clips"], "minItems": n, "maxItems": n}}},
        "stream": False,
        "options": {"temperature": 0.5, "num_ctx": 12288},
    }
    for attempt in (1, 2):
        log.info("asking Ollama '%s' for %d clips (attempt %d)...", model, n, attempt)
        try:
            r = requests.post(config.OLLAMA_URL, json=payload, timeout=900)
            r.raise_for_status()
            content = r.json()["message"]["content"]
            clips = _validate(content, n, duration)
            if clips:
                for c in clips:
                    log.info("  picked %s-%s  %s",
                             fmt_time(c["start"]), fmt_time(c["end"]), c["title"])
                return clips
            log.warning("no valid clips parsed; head=%r", content[:200])
        except requests.RequestException as e:
            raise RuntimeError(
                f"could not reach Ollama at {config.OLLAMA_URL} ({e}). "
                f"Is `ollama serve` running and the model pulled?")
        except (KeyError, ValueError) as e:
            log.warning("parse error (%s); retrying", e)
    raise RuntimeError("Ollama did not return usable clips after 2 attempts")


import re as _re
_DEV = _re.compile(r"[ऀ-ॿ]")


def _ensure_latin(text):
    """If the model slipped and returned Devanagari, romanize it to Hinglish so
    titles/filenames stay Latin."""
    if not text or not _DEV.search(text):
        return text
    from .captions import to_hinglish
    out = " ".join(to_hinglish(w) if _DEV.search(w) else w for w in text.split())
    return out.strip()


def _clean_title(t):
    import re
    t = _ensure_latin(t)
    t = re.sub(r"\b\d{1,4}\s*[-:]\s*\d{1,4}\b", "", t)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s+([:!?.,])", r"\1", t)
    t = re.sub(r"[:\-–]\s*$", "", t).strip(" -–:,")
    t = re.sub(r"^[:\-–,\s]+", "", t)
    return t.strip() or "Untitled clip"


def _validate(content, n, duration):
    data = json.loads(content)
    raw = data if isinstance(data, list) else (
        data.get("clips") or data.get("segments") or data.get("highlights") or [])
    cleaned = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        try:
            start = max(0.0, parse_time(c["start"]))
            end = min(duration, parse_time(c["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        if end - start > config.MAX_LEN:
            end = start + config.MAX_LEN
        if end - start < config.MIN_LEN:
            end = min(duration, end + (config.MIN_LEN - (end - start)))
        if end - start < config.MIN_LEN:
            start = max(0.0, start - (config.MIN_LEN - (end - start)))
        if end - start < 8:
            continue
        cleaned.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "title": _clean_title(str(c.get("title", "")).strip()),
            "hook": _ensure_latin(str(c.get("hook", "")).strip()),
            "reason": _ensure_latin(str(c.get("reason", "")).strip()),
        })
    cleaned.sort(key=lambda x: x["start"])
    picked = []
    for c in cleaned:
        if picked and c["start"] < picked[-1]["end"]:
            continue
        picked.append(c)
        if len(picked) >= n:
            break
    return picked
