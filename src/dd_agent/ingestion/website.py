"""Fetch and extract text content from a company website.

Strategy: fetch the landing page, plus a few high-signal subpaths (about, team,
pricing, customers, blog). Strip nav/footer chrome. Return concatenated text.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urljoin, urlparse

import httpx
from selectolax.parser import HTMLParser

_SUBPATHS = ["/about", "/team", "/company", "/pricing", "/customers", "/blog"]
_DROP_SELECTORS = ["nav", "footer", "script", "style", "noscript", "iframe", "svg"]
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    )
}


def _clean(html: str) -> str:
    tree = HTMLParser(html)
    for sel in _DROP_SELECTORS:
        for node in tree.css(sel):
            node.decompose()
    body = tree.body
    text = body.text(separator="\n", strip=True) if body else tree.text(strip=True)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)


async def fetch_one(client: httpx.AsyncClient, url: str) -> tuple[str, str]:
    try:
        r = await client.get(url, timeout=15.0, follow_redirects=True, headers=_HEADERS)
        if r.status_code >= 400:
            return url, ""
        return url, _clean(r.text)
    except Exception:
        return url, ""


async def fetch_site(url: str, max_chars: int = 40_000) -> str:
    """Fetch the site and a handful of common subpaths, return a combined text blob."""
    if not url.startswith("http"):
        url = "https://" + url
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    targets = [url] + [urljoin(base, p) for p in _SUBPATHS]

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(fetch_one(client, t) for t in targets))

    blocks: list[str] = []
    for u, text in results:
        if not text:
            continue
        blocks.append(f"--- {u} ---\n{text}")
    combined = "\n\n".join(blocks)
    return combined[:max_chars]
