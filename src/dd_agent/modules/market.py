"""Market subagent: TAM/SAM/SOM + competitors + inflection thesis."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..citations import Citation, CitationBook
from ..context import DealContext
from ..data_sources.search import SearchResult, search_many
from ..retrieval import retrieve, format_for_prompt
from ._llm import load_prompt, render_section


@dataclass
class MarketResult:
    section_markdown: str
    citations: list[Citation]
    web_results: list[SearchResult]


async def run_market(ctx: DealContext, base_system: str) -> MarketResult:
    queries = _build_queries(ctx)
    search_buckets = await search_many(queries, max_per=5)

    web_results: list[SearchResult] = []
    seen: set[str] = set()
    for results in search_buckets.values():
        for r in results:
            if r.url in seen:
                continue
            seen.add(r.url)
            web_results.append(r)

    book = CitationBook()
    for r in web_results:
        book.add(Citation(key=r.url, title=r.title, url=r.url, snippet=r.snippet, source_type="web"))

    elad = retrieve(
        f"market sizing inflection competitor analysis {ctx.sector or ''}",
        k=4,
    )

    section_prompt = load_prompt("modules/market_prompt.md")
    system = f"{base_system}\n\n---\n\n{section_prompt}"
    user = _build_user(ctx, web_results, elad)
    text = await render_section(system=system, user=user, max_tokens=4500)

    return MarketResult(section_markdown=text, citations=book.citations, web_results=web_results)


def _build_queries(ctx: DealContext) -> list[str]:
    name = ctx.company_name
    sector = ctx.sector or ""
    one_liner = ctx.one_liner or ""
    qs = [
        f'"{name}" company',
        f'"{name}" competitors',
        f'"{name}" {sector} market size',
        f'{sector} market size billion 2024 2025',
        f'{sector} competitors landscape',
    ]
    if one_liner:
        qs.append(f'{one_liner} market growth')
    return [q.strip() for q in qs if q.strip()]


def _build_user(ctx: DealContext, web: list[SearchResult], elad) -> str:
    parts = [
        "# DealContext",
        f"Company: {ctx.company_name}",
        f"Sector: {ctx.sector or '(unknown)'}",
        f"Stage: {ctx.stage or '(unknown)'}",
        f"One-liner: {ctx.one_liner or '(unknown)'}",
        f"Website: {ctx.website or '(unknown)'}",
    ]
    if ctx.raw_memo:
        parts.append("\n## Memo\n" + ctx.raw_memo[:8000])
    if ctx.raw_deck_text:
        parts.append("\n## Deck excerpt\n" + ctx.raw_deck_text[:5000])
    parts.append("\n## Web search results (cite as [n], 1-indexed)\n")
    for i, r in enumerate(web, 1):
        parts.append(f"[{i}] {r.title} — {r.url}\n    {r.snippet}")
    refs = format_for_prompt(elad)
    if refs:
        parts.append("\n## Elad references\n" + refs)
    return "\n".join(parts)
