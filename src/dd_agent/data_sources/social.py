"""Social signal gathering: LinkedIn snippets, Twitter long-form, podcast transcripts.

Free-tier strategy:
  - LinkedIn: site:linkedin.com/in/ via web search → public summaries only.
  - Twitter: Nitter mirrors with a fallback to web search of x.com.
  - Podcasts: yt-dlp + whisper for any YouTube interview the founder is on.

All return citation-ready (url, title, snippet) tuples.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .search import web_search, SearchResult, fetch_page_text


@dataclass(frozen=True)
class SocialSignal:
    platform: str        # "linkedin" | "twitter" | "podcast" | "interview" | "press"
    url: str
    title: str
    snippet: str


async def linkedin_summary(name: str, company: str | None = None) -> list[SocialSignal]:
    q = f'site:linkedin.com/in "{name}"'
    if company:
        q += f' "{company}"'
    results = await web_search(q, max_results=3)
    return [
        SocialSignal(platform="linkedin", url=r.url, title=r.title, snippet=r.snippet)
        for r in results
        if "linkedin.com/in/" in r.url
    ]


async def twitter_signals(handle: str | None, name: str | None = None) -> list[SocialSignal]:
    if handle:
        handle = handle.lstrip("@")
        q = f"site:x.com/{handle} OR site:twitter.com/{handle}"
    elif name:
        q = f'"{name}" twitter OR x.com'
    else:
        return []
    results = await web_search(q, max_results=5)
    return [
        SocialSignal(platform="twitter", url=r.url, title=r.title, snippet=r.snippet)
        for r in results
        if "twitter.com" in r.url or "x.com" in r.url or "nitter" in r.url
    ]


async def podcast_appearances(name: str) -> list[SocialSignal]:
    queries = [
        f'"{name}" podcast interview',
        f'"{name}" "Invest Like the Best" OR "20VC" OR "No Priors" OR "Acquired"',
    ]
    out: list[SocialSignal] = []
    seen: set[str] = set()
    for q in queries:
        for r in await web_search(q, max_results=5):
            if r.url in seen:
                continue
            if not any(k in r.url for k in ["youtube.com", "youtu.be", "spotify.com", "apple.com/podcast", "podcasts.apple.com", "substack.com"]):
                continue
            seen.add(r.url)
            out.append(SocialSignal(platform="podcast", url=r.url, title=r.title, snippet=r.snippet))
    return out[:8]


async def press_and_blogs(name: str, company: str | None = None) -> list[SocialSignal]:
    qs = [
        f'"{name}" interview',
        f'"{name}" essay OR blog OR post',
    ]
    if company:
        qs.append(f'"{name}" "{company}"')
    out: list[SocialSignal] = []
    seen: set[str] = set()
    for q in qs:
        for r in await web_search(q, max_results=5):
            if r.url in seen:
                continue
            seen.add(r.url)
            out.append(SocialSignal(platform="press", url=r.url, title=r.title, snippet=r.snippet))
    return out[:10]


async def gather_for_founder(
    *, name: str, company: str | None = None, twitter_handle: str | None = None
) -> list[SocialSignal]:
    li, tw, pc, pr = await asyncio.gather(
        linkedin_summary(name, company),
        twitter_signals(twitter_handle, name),
        podcast_appearances(name),
        press_and_blogs(name, company),
    )
    return li + tw + pc + pr


# --- Optional podcast transcription via yt-dlp + whisper ----------------------


async def transcribe_youtube(url: str, max_minutes: int = 60) -> str | None:
    """Best-effort: download audio with yt-dlp and transcribe with whisper.

    Returns None if either binary is missing or any step fails. Caches output
    under data/transcripts/.
    """
    if not shutil.which("yt-dlp"):
        return None
    try:
        import whisper  # noqa: F401
    except ImportError:
        return None

    cache_dir = Path(os.environ.get("DD_DATA_DIR", "./data")) / "transcripts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    vid = _video_id(url)
    if not vid:
        return None
    txt_path = cache_dir / f"{vid}.txt"
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8")

    audio_path = cache_dir / f"{vid}.m4a"
    if not audio_path.exists():
        proc = await asyncio.create_subprocess_exec(
            "yt-dlp", "-f", "bestaudio[ext=m4a]", "-o", str(audio_path), url,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        await proc.wait()
        if proc.returncode != 0 or not audio_path.exists():
            return None

    def _transcribe() -> str:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(str(audio_path), fp16=False)
        return result.get("text", "")

    text = await asyncio.to_thread(_transcribe)
    if text:
        txt_path.write_text(text, encoding="utf-8")
    return text or None


def _video_id(url: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_\-]{6,})", url)
    return m.group(1) if m else None


async def fetch_full_text(signal: SocialSignal, max_chars: int = 6000) -> str:
    """For deeper context, pull the full body of a signal's URL."""
    return await fetch_page_text(signal.url, max_chars=max_chars)
