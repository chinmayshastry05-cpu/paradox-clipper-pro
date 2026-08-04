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

# Viral playbook — distilled patterns of top-performing short-form clips. This is the
# "trained-in" instinct: what actually makes people stop, watch, and share. Injected
# into every selection prompt so the model scores moments like a seasoned clipper.
VIRAL_PLAYBOOK = (
    "VIRAL PLAYBOOK (score every candidate moment against this, keep the highest):\n"
    "1. HOOK IN THE FIRST 2 SECONDS. The clip must OPEN on the punch — the wild line, "
    "the claim, the reaction — never on the wind-up or setup chatter. If the best line "
    "is 8s in, start the clip ~2s before it, not at the topic's beginning.\n"
    "2. HOOK TYPES that stop the scroll (a clip should hit at least one, hard):\n"
    "   - Curiosity gap / open loop: a question or tease the viewer needs resolved.\n"
    "   - Controversy / hot take: a bold, divisive, 'did they just say that' claim.\n"
    "   - Visceral emotion: shock, secondhand cringe, genuine vulnerability, outrage.\n"
    "   - High stakes / specificity: real names, real numbers, real consequences.\n"
    "   - Taboo / forbidden: sex, money, addiction, death, beef — named bluntly.\n"
    "   - Dark / savage humor: brutal roast, morbid punchline, deadpan bleak take.\n"
    "3. COMPLETE ARC: tension then release. Setup -> payoff, lands inside the clip, "
    "makes full sense with ZERO outside context. No clip that needs 'earlier they said'.\n"
    "4. SHARE TRIGGER: the moment makes a viewer FEEL something, pick a side, or tag a "
    "friend ('this is so you'). If it wouldn't get sent in a group chat, skip it.\n"
    "5. PACING: dense, no dead air. Cut rambles, throat-clearing, 'umm' runs, logistics.\n"
    "6. REJECT: intros, greetings, thank-yous, generic praise, context-only setup, "
    "anything that only makes sense if you watched the rest.\n"
    "For each of the N clips: internally rate hook-strength, emotion, and completeness, "
    "and only return moments that are strong on all three.\n\n"
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
        f"{VIRAL_PLAYBOOK}"
        f"When the content is comedy, lean into the funniest, darkest, most savage "
        f"beats (brutal roasts, morbid jokes, taboo confessions) over wholesome ones.\n\n"
        f"LENGTH: choose the natural length of each moment, anywhere from 18 to 75 "
        f"seconds. Start right before the setup; end right after the payoff. Don't pad.\n\n"
        f"TITLE — this is what earns the click. It is ONE plain English sentence "
        f"(a single string, NOT an object, NOT JSON, no braces, no field names). "
        f"Shape: a short provocative hook, then a colon, then the concrete specific "
        f"thing that happens, ending in ! or ?. Example shape only: "
        f"\"Brutal Roast: Abhijeet Destroys the Students!\".\n"
        f"  - Name the real person from the transcript (never 'he/someone/a guy').\n"
        f"  - Build it ONLY from what is actually said in THIS segment; don't invent "
        f"topics or reuse this instruction's wording.\n"
        f"  - NEVER pick pure intros/greetings/'X appears on stage'.\n"
        f"  - Clean English, no raw transcript words/numbers/timestamps. Don't sanitize "
        f"spicy/taboo topics — name them bluntly.\n\n"
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
    # Safety net: if the model stuffed JSON into the title field, recover the text.
    if "{" in t or '"hook"' in t or "'hook'" in t:
        vals = re.findall(r'["“]?(?:hook|specific|title)["”]?\s*[:=]\s*["“]([^"”]+)["”]', t)
        if vals:
            t = ": ".join(v.strip() for v in vals[:2])
        else:
            t = re.sub(r'[{}\[\]"“”]', " ", t)
            t = re.sub(r'\b(hook|specific|title)\b\s*[:=]?', " ", t, flags=re.I)
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
