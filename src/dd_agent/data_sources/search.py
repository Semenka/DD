"""Web search via Tavily (free tier ~1k/month) with a DuckDuckGo HTML fallback.

Returns a normalized list of SearchResult objects with url + title + snippet
ready for the citations module.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from urllib.parse import quote_plus, urlparse

import httpx
from selectolax.parser import HTMLParser


@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str
    snippet: str
    source: str               # "tavily" | "duckduckgo"


async def web_search(query: str, max_results: int = 8) -> list[SearchResult]:
    """Search the web. Tries Tavily first if API key is set, falls back to DDG."""
    key = os.environ.get("TAVILY_API_KEY")
    if key:
        try:
            return await _tavily(query, key, max_results)
        except Exception:
            pass
    return await _duckduckgo(query, max_results)


async def _tavily(query: str, api_key: str, max_results: int) -> list[SearchResult]:
    payload = {
        "api_key": api_key,
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
        from urllib.parse import unquote
        return unquote(m.group(1))
    parsed = urlparse(href)
    if parsed.scheme and parsed.netloc:
        return href
    return None


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


async def search_many(queries: list[str], max_per: int = 5) -> dict[str, list[SearchResult]]:
    """Run several queries in parallel."""
    results = await asyncio.gather(*(web_search(q, max_per) for q in queries))
    return dict(zip(queries, results))
