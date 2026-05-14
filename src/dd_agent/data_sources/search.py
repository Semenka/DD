"""Web search with a four-tier cascade.

Priority order (highest signal first):

1. **Perplexity** Sonar — search + LLM synthesis in one call. Returns an
   answer string plus source citations. Best for "research" queries.
2. **Gemini grounding** — gemini-2.5-flash with the google_search tool.
   Returns text plus a `groundingMetadata.groundingChunks` list of URLs.
3. **Tavily** — classic search index.
4. **DuckDuckGo HTML** — anonymous fallback (frequently rate-limited).

`web_search()` returns a list of `SearchResult{url, title, snippet, source}`
from whichever tier responded first with non-empty results.

`ask_grounded()` is a higher-level helper for the market subagent: it sends a
single question and returns an answer string plus citations, suitable for TAM /
competitor reasoning where the LLM needs to read across pages, not just
collect URLs.

Configure via env (any one of these is enough):
    PERPLEXITY_API_KEY
    GEMINI_API_KEY
    TAVILY_API_KEY
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from urllib.parse import quote_plus, unquote, urlparse

import httpx
from selectolax.parser import HTMLParser


@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str
    snippet: str
    source: str               # "perplexity" | "gemini" | "tavily" | "duckduckgo"


@dataclass(frozen=True)
class GroundedAnswer:
    text: str
    sources: list[SearchResult]
    source: str               # which backend produced the answer


# --- main entrypoints --------------------------------------------------------


async def web_search(query: str, max_results: int = 8) -> list[SearchResult]:
    """Return ranked URLs/snippets. Tries backends in cascade order."""
    if os.environ.get("PERPLEXITY_API_KEY"):
        try:
            out = await _perplexity_search(query, max_results)
            if out:
                return out
        except Exception:
            pass
    if os.environ.get("GEMINI_API_KEY"):
        try:
            out = await _gemini_search(query, max_results)
            if out:
                return out
        except Exception:
            pass
    if os.environ.get("TAVILY_API_KEY"):
        try:
            out = await _tavily(query, max_results)
            if out:
                return out
        except Exception:
            pass
    return await _duckduckgo(query, max_results)


async def ask_grounded(question: str, max_sources: int = 10) -> GroundedAnswer | None:
    """Return a single grounded answer + citations. Used by the market subagent
    where we want the search backend to actually read pages and synthesize."""
    if os.environ.get("PERPLEXITY_API_KEY"):
        try:
            out = await _perplexity_ask(question, max_sources)
            if out:
                return out
        except Exception:
            pass
    if os.environ.get("GEMINI_API_KEY"):
        try:
            out = await _gemini_ask(question, max_sources)
            if out:
                return out
        except Exception:
            pass
    return None


async def search_many(queries: list[str], max_per: int = 5) -> dict[str, list[SearchResult]]:
    """Run several search queries in parallel."""
    results = await asyncio.gather(*(web_search(q, max_per) for q in queries))
    return dict(zip(queries, results))


# --- Perplexity --------------------------------------------------------------

PPLX_URL = "https://api.perplexity.ai/chat/completions"
PPLX_MODEL = os.environ.get("DD_PPLX_MODEL", "sonar")


async def _perplexity_search(query: str, max_results: int) -> list[SearchResult]:
    """Use Perplexity's search-only mode. We ask for a short answer and harvest
    the citations as our SearchResult list."""
    body = {
        "model": PPLX_MODEL,
        "messages": [
            {"role": "system", "content": "Search the web and answer concisely with sources."},
            {"role": "user", "content": query},
        ],
        "max_tokens": 400,
        "return_related_questions": False,
    }
    out: list[SearchResult] = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            PPLX_URL, json=body,
            headers={"Authorization": f"Bearer {os.environ['PERPLEXITY_API_KEY']}"},
        )
        r.raise_for_status()
        data = r.json()
    # Modern Perplexity returns structured search_results; legacy returns citations.
    for item in data.get("search_results", []) or []:
        url = item.get("url")
        if not url:
            continue
        out.append(SearchResult(
            url=url,
            title=(item.get("title") or url)[:200],
            snippet=(item.get("snippet") or item.get("text") or "")[:400],
            source="perplexity",
        ))
        if len(out) >= max_results:
            break
    if not out:
        for url in data.get("citations", []) or []:
            out.append(SearchResult(url=url, title=url, snippet="", source="perplexity"))
            if len(out) >= max_results:
                break
    return out


async def _perplexity_ask(question: str, max_sources: int) -> GroundedAnswer | None:
    body = {
        "model": PPLX_MODEL,
        "messages": [
            {"role": "system", "content": (
                "You are a market analyst. Answer with concise, cited facts. "
                "Prefer numbers and named sources. If sources disagree, present the range."
            )},
            {"role": "user", "content": question},
        ],
        "max_tokens": 1500,
    }
    async with httpx.AsyncClient(timeout=45.0) as client:
        r = await client.post(
            PPLX_URL, json=body,
            headers={"Authorization": f"Bearer {os.environ['PERPLEXITY_API_KEY']}"},
        )
        r.raise_for_status()
        data = r.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    sources: list[SearchResult] = []
    for item in data.get("search_results", []) or []:
        if not item.get("url"):
            continue
        sources.append(SearchResult(
            url=item["url"],
            title=(item.get("title") or item["url"])[:200],
            snippet=(item.get("snippet") or item.get("text") or "")[:400],
            source="perplexity",
        ))
        if len(sources) >= max_sources:
            break
    if not sources:
        for url in data.get("citations", []) or []:
            sources.append(SearchResult(url=url, title=url, snippet="", source="perplexity"))
            if len(sources) >= max_sources:
                break
    if not text:
        return None
    return GroundedAnswer(text=text, sources=sources, source="perplexity")


# --- Gemini grounding -------------------------------------------------------

GEMINI_MODEL = os.environ.get("DD_GEMINI_MODEL", "gemini-2.5-flash")


async def _gemini_call(prompt: str, *, timeout: float = 30.0) -> dict:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={os.environ['GEMINI_API_KEY']}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=body)
        r.raise_for_status()
        return r.json()


def _gemini_extract(data: dict) -> tuple[str, list[SearchResult]]:
    cand = (data.get("candidates") or [{}])[0]
    parts = cand.get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()

    sources: list[SearchResult] = []
    meta = cand.get("groundingMetadata") or {}
    chunks = meta.get("groundingChunks") or []
    for ch in chunks:
        web = ch.get("web") or {}
        url = web.get("uri")
        if not url:
            continue
        sources.append(SearchResult(
            url=url,
            title=web.get("title", url)[:200],
            snippet="",
            source="gemini",
        ))
    return text, sources


async def _gemini_search(query: str, max_results: int) -> list[SearchResult]:
    data = await _gemini_call(
        f"Search the web for: {query}\n\nList the most relevant 8 URLs with one-line summaries.",
        timeout=30.0,
    )
    _, sources = _gemini_extract(data)
    return sources[:max_results]


async def _gemini_ask(question: str, max_sources: int) -> GroundedAnswer | None:
    data = await _gemini_call(question, timeout=45.0)
    text, sources = _gemini_extract(data)
    if not text:
        return None
    return GroundedAnswer(text=text, sources=sources[:max_sources], source="gemini")


# --- Tavily ------------------------------------------------------------------


async def _tavily(query: str, max_results: int) -> list[SearchResult]:
    payload = {
        "api_key": os.environ["TAVILY_API_KEY"],
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post("https://api.tavily.com/search", json=payload)
        r.raise_for_status()
        data = r.json()
    return [
        SearchResult(
            url=item.get("url", ""),
            title=item.get("title", item.get("url", ""))[:200],
            snippet=(item.get("content") or "")[:400],
            source="tavily",
        )
        for item in data.get("results", [])
        if item.get("url")
    ]


# --- DuckDuckGo HTML (last-resort fallback) ---------------------------------


async def _duckduckgo(query: str, max_results: int) -> list[SearchResult]:
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (DD Agent research bot)"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
    except Exception:
        return []
    tree = HTMLParser(r.text)
    out: list[SearchResult] = []
    for result in tree.css(".result"):
        a = result.css_first("a.result__a")
        snippet_node = result.css_first(".result__snippet")
        if not a:
            continue
        href = a.attributes.get("href", "")
        title = a.text(strip=True)
        snippet = snippet_node.text(strip=True) if snippet_node else ""
        clean_url = _unwrap_ddg(href)
        if clean_url:
            out.append(SearchResult(url=clean_url, title=title, snippet=snippet, source="duckduckgo"))
        if len(out) >= max_results:
            break
    return out


def _unwrap_ddg(href: str) -> str | None:
    if not href:
        return None
    if href.startswith("//"):
        href = "https:" + href
    m = re.search(r"uddg=([^&]+)", href)
    if m:
        return unquote(m.group(1))
    parsed = urlparse(href)
    if parsed.scheme and parsed.netloc:
        return href
    return None


# --- page fetcher (unchanged, reused by other adapters) ---------------------


async def fetch_page_text(url: str, max_chars: int = 8000) -> str:
    """Fetch a URL and extract its body text. Used when a search result needs deeper context."""
    headers = {"User-Agent": "Mozilla/5.0 (DD Agent)"}
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
    except Exception:
        return ""
    tree = HTMLParser(r.text)
    for sel in ["script", "style", "nav", "footer", "noscript"]:
        for n in tree.css(sel):
            n.decompose()
    body = tree.body
    text = body.text(separator="\n", strip=True) if body else tree.text(strip=True)
    return text[:max_chars]
