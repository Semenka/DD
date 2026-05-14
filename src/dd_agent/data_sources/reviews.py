"""Independent-voice signal: G2 / Capterra / ProductHunt / Reddit / HN / Glassdoor.

Substitute for real customer reference calls. Pulls public reviews + mentions
and returns citation-ready summaries. Surfaced in the Traction section as
"Independent Voice".
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .search import web_search, SearchResult


@dataclass(frozen=True)
class ReviewSignal:
    source: str          # "g2" | "capterra" | "producthunt" | "reddit" | "hn" | "glassdoor"
    url: str
    title: str
    snippet: str


_SOURCES = {
    "g2": "site:g2.com",
    "capterra": "site:capterra.com",
    "producthunt": "site:producthunt.com",
    "reddit": "site:reddit.com",
    "hn": "site:news.ycombinator.com",
    "glassdoor": "site:glassdoor.com",
}


async def _search_source(company: str, source: str, op: str) -> list[ReviewSignal]:
    query = f'{op} "{company}"'
    results = await web_search(query, max_results=4)
    return [
        ReviewSignal(source=source, url=r.url, title=r.title, snippet=r.snippet)
        for r in results
    ]


async def gather_for_company(company: str) -> dict[str, list[ReviewSignal]]:
    """Run all source queries in parallel. Returns {source: [signals]}."""
    tasks = {s: _search_source(company, s, op) for s, op in _SOURCES.items()}
    results = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results))


def flatten(buckets: dict[str, list[ReviewSignal]]) -> list[ReviewSignal]:
    out: list[ReviewSignal] = []
    for items in buckets.values():
        out.extend(items)
    return out
