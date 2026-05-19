"""Web search with a five-tier cascade.

Priority order (configurable via `DD_SEARCH_PREFERRED`, default order below):

1. **OpenClaw infer-web** — shells out to `openclaw infer web search --provider
   {gemini|perplexity|tavily|duckduckgo|...}`. Uses OpenClaw's already-
   configured providers without dd-agent needing its own keys. Provider
   selectable via `DD_OPENCLAW_SEARCH_PROVIDER` (default: gemini).
2. **Perplexity** Sonar — direct Perplexity API call. Search + LLM synthesis
   in one call. Returns an answer string plus source citations.
3. **Gemini grounding** — direct Gemini API call with the google_search tool.
   Returns text plus a `groundingMetadata.groundingChunks` list of URLs.
4. **Tavily** — classic search index.
5. **DuckDuckGo HTML** — anonymous fallback (frequently rate-limited).

`web_search()` returns a list of `SearchResult{url, title, snippet, source}`
from whichever tier responded first with non-empty results.

`ask_grounded()` is a higher-level helper for synthesis queries: returns
an answer string plus citations.

Configure via env (any one is enough — the cascade falls through):
    DD_OPENCLAW_SEARCH_PROVIDER  (default: gemini, options: gemini|perplexity|tavily|duckduckgo|exa|firecrawl|ollama)
    DD_SEARCH_PREFERRED          (default: openclaw,perplexity,gemini,tavily,duckduckgo)
    PERPLEXITY_API_KEY
    GEMINI_API_KEY
    TAVILY_API_KEY
"""

from __future__ import annotations

import asyncio
import json as _json
import os
import re
import shutil
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


_DEFAULT_ORDER = ("openclaw", "perplexity", "gemini", "tavily", "duckduckgo")


def _backend_order() -> tuple[str, ...]:
    """User-overridable backend priority. DD_SEARCH_PREFERRED is a comma-
    separated list of backend ids."""
    raw = os.environ.get("DD_SEARCH_PREFERRED")
    if not raw:
        return _DEFAULT_ORDER
    parts = tuple(p.strip().lower() for p in raw.split(",") if p.strip())
    # Append any default backends the user omitted so we always fall back.
    seen = set(parts)
    rest = tuple(b for b in _DEFAULT_ORDER if b not in seen)
    return parts + rest


def _backend_available(backend: str) -> bool:
    """Cheap availability check — env keys, binaries, etc."""
    if backend == "openclaw":
        return shutil.which("openclaw") is not None
    if backend == "perplexity":
        return bool(os.environ.get("PERPLEXITY_API_KEY"))
    if backend == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY"))
    if backend == "tavily":
        return bool(os.environ.get("TAVILY_API_KEY"))
    if backend == "duckduckgo":
        return True
    return False


async def web_search(query: str, max_results: int = 8) -> list[SearchResult]:
    """Return ranked URLs/snippets. Tries backends in DD_SEARCH_PREFERRED order
    (default: openclaw → perplexity → gemini → tavily → duckduckgo)."""
    for backend in _backend_order():
        if not _backend_available(backend):
            continue
        try:
            if backend == "openclaw":
                out = await _openclaw_search(query, max_results)
            elif backend == "perplexity":
                out = await _perplexity_search(query, max_results)
            elif backend == "gemini":
                out = await _gemini_search(query, max_results)
            elif backend == "tavily":
                out = await _tavily(query, max_results)
            elif backend == "duckduckgo":
                out = await _duckduckgo(query, max_results)
            else:
                continue
        except Exception:
            continue
        if out:
            return out
    return []


async def ask_grounded(
    question: str,
    max_sources: int = 10,
    max_tokens: int = 1500,
) -> GroundedAnswer | None:
    """Return a single grounded answer + citations. Honors DD_SEARCH_PREFERRED."""
    for backend in _backend_order():
        if not _backend_available(backend):
            continue
        try:
            if backend == "openclaw":
                out = await _openclaw_ask(question, max_sources)
            elif backend == "perplexity":
                out = await _perplexity_ask(question, max_sources, max_tokens=max_tokens)
            elif backend == "gemini":
                out = await _gemini_ask(question, max_sources)
            else:
                continue
        except Exception:
            continue
        if out:
            return out
    return None


async def search_many(queries: list[str], max_per: int = 5) -> dict[str, list[SearchResult]]:
    """Run several search queries in parallel."""
    results = await asyncio.gather(*(web_search(q, max_per) for q in queries))
    return dict(zip(queries, results))


# --- OpenClaw infer-web -----------------------------------------------------

OPENCLAW_BIN = os.environ.get("DD_OPENCLAW_BIN", "openclaw")
# Default to gemini provider because: (a) it returns rich synthesized
# content + citations in a single call, (b) the user already has GEMINI_API_KEY
# configured, (c) OpenClaw's perplexity provider blows quotas fast since it's
# shared with the rest of the OpenClaw ecosystem.
OPENCLAW_PROVIDER = os.environ.get("DD_OPENCLAW_SEARCH_PROVIDER", "gemini")
_OPENCLAW_TIMEOUT = float(os.environ.get("DD_OPENCLAW_TIMEOUT", "60"))


async def _openclaw_run(args: list[str], timeout: float = _OPENCLAW_TIMEOUT) -> dict | None:
    """Shell out to openclaw, parse the --json stdout, return the payload dict."""
    proc = await asyncio.create_subprocess_exec(
        OPENCLAW_BIN, *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return None
    if proc.returncode != 0:
        return None
    try:
        return _json.loads(stdout.decode("utf-8", errors="replace"))
    except _json.JSONDecodeError:
        return None


def _openclaw_extract_result(payload: dict) -> dict | None:
    """Pull the first non-empty result dict from an `infer web` payload."""
    outputs = payload.get("outputs") or []
    for o in outputs:
        result = o.get("result") or {}
        if result:
            return result
    return None


async def _openclaw_search(query: str, max_results: int) -> list[SearchResult]:
    """`openclaw infer web search` — returns content + citations."""
    payload = await _openclaw_run([
        "infer", "web", "search",
        "--provider", OPENCLAW_PROVIDER,
        "--query", query,
        "--limit", str(max_results),
        "--json",
    ])
    if not payload or not payload.get("ok"):
        return []
    result = _openclaw_extract_result(payload)
    if not result:
        return []
    cits = result.get("citations") or []
    out: list[SearchResult] = []
    for c in cits[:max_results]:
        url = c.get("url")
        if not url:
            continue
        out.append(SearchResult(
            url=url,
            title=(c.get("title") or url)[:200],
            snippet=(c.get("snippet") or c.get("description") or "")[:400],
            source=f"openclaw/{result.get('provider', OPENCLAW_PROVIDER)}",
        ))
    return out


async def _openclaw_ask(question: str, max_sources: int) -> GroundedAnswer | None:
    """Same call as _openclaw_search, but we return the synthesized content
    as a GroundedAnswer instead of just the URLs."""
    payload = await _openclaw_run([
        "infer", "web", "search",
        "--provider", OPENCLAW_PROVIDER,
        "--query", question,
        "--limit", str(max_sources),
        "--json",
    ])
    if not payload or not payload.get("ok"):
        return None
    result = _openclaw_extract_result(payload)
    if not result:
        return None
    content = result.get("content") or ""
    # OpenClaw wraps content in <<<EXTERNAL_UNTRUSTED_CONTENT id="...">>> markers
    # for tool-output safety. Strip them — we trust the answer body itself.
    content = re.sub(
        r"<<<\s*EXTERNAL_UNTRUSTED_CONTENT[^>]*>>>\s*", "", content,
    )
    content = re.sub(r"<<<\s*END_EXTERNAL_UNTRUSTED_CONTENT[^>]*>>>\s*", "", content)
    content = content.strip()
    if not content:
        return None
    sources: list[SearchResult] = []
    for c in (result.get("citations") or [])[:max_sources]:
        url = c.get("url")
        if not url:
            continue
        sources.append(SearchResult(
            url=url, title=(c.get("title") or url)[:200], snippet="",
            source=f"openclaw/{result.get('provider', OPENCLAW_PROVIDER)}",
        ))
    return GroundedAnswer(
        text=content,
        sources=sources,
        source=f"openclaw/{result.get('provider', OPENCLAW_PROVIDER)}",
    )


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


async def _perplexity_ask(
    question: str, max_sources: int, max_tokens: int = 1500,
) -> GroundedAnswer | None:
    body = {
        "model": PPLX_MODEL,
        "messages": [
            {"role": "system", "content": (
                "You are a market analyst. Answer with concise, cited facts. "
                "Prefer numbers and named sources. If sources disagree, present the range."
            )},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
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
