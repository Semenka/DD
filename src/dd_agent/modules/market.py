"""Market subagent: TAM/SAM/SOM + competitors + inflection thesis.

Strategy: combine breadth (URL list across multiple queries) with depth
(a grounded "ask" call that synthesizes a market-sizing answer with cited
analyst figures). Both go into the prompt; the LLM picks the best signal."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from ..citations import Citation, CitationBook
from ..context import DealContext
from ..data_sources.search import (
    GroundedAnswer,
    SearchResult,
    ask_grounded,
    search_many,
)
from ..retrieval import retrieve, format_for_prompt
from ._llm import load_prompt, render_section


@dataclass
class MarketResult:
    section_markdown: str
    citations: list[Citation]
    web_results: list[SearchResult]


async def run_market(ctx: DealContext, base_system: str) -> MarketResult:
    queries = _build_queries(ctx)
    grounded_questions = _build_grounded_questions(ctx)

    search_buckets, *grounded_answers = await asyncio.gather(
        search_many(queries, max_per=5),
        *(ask_grounded(q) for q in grounded_questions),
    )

    web_results: list[SearchResult] = []
    seen: set[str] = set()
    for results in search_buckets.values():
        for r in results:
            if r.url in seen:
                continue
            seen.add(r.url)
            web_results.append(r)
    for ans in grounded_answers:
        if ans is None:
            continue
        for r in ans.sources:
            if r.url in seen:
                continue
            seen.add(r.url)
            web_results.append(r)

    book = CitationBook()
    for r in web_results:
        book.add(Citation(
            key=r.url, title=r.title, url=r.url, snippet=r.snippet, source_type="web",
        ))

    elad = retrieve(
        f"market sizing inflection competitor analysis {ctx.sector or ''}",
        k=4,
    )

    section_prompt = load_prompt("modules/market_prompt.md")
    system = f"{base_system}\n\n---\n\n{section_prompt}"
    user = _build_user(ctx, web_results, grounded_answers, grounded_questions, elad)
    # v8: 4500 → 2400 tokens — analyst sections feed the synthesizer, not
    # the final memo. They no longer need to be exhaustive narratives.
    text = await render_section(system=system, user=user, max_tokens=2400)

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


def _build_grounded_questions(ctx: DealContext) -> list[str]:
    """Questions where we want a *synthesized* answer (analyst figures, named
    competitor list, inflection drivers) rather than just URLs."""
    name = ctx.company_name
    sector = ctx.sector or "this sector"
    one_liner = ctx.one_liner or f"the company {name}"
    return [
        (
            f"What is the current total addressable market (TAM) for {sector}, "
            f"and what is the projected size in 5 years? Cite specific analyst "
            f"reports (Markets and Markets, Insight Partners, Gartner, McKinsey, "
            f"BCG, etc.) with USD figures. If estimates conflict, present the range."
        ),
        (
            f"Who are the top 6-10 direct and adjacent competitors to {name} "
            f"({one_liner})? For each: stage (public/private), most recent "
            f"valuation or market cap, their specific wedge, and how they differ "
            f"from {name}. Be concrete with company names."
        ),
        (
            f"What is the specific market inflection happening right now that "
            f"enables {name} to exist? Look for: platform shifts, regulatory "
            f"unlocks, cost-frontier crossings, demographic shifts, or new "
            f"buyer behaviors. Cite recent (2024-2026) news or analyst notes."
        ),
    ]


def _build_user(
    ctx: DealContext,
    web: list[SearchResult],
    grounded_answers: list[GroundedAnswer | None],
    grounded_questions: list[str],
    elad,
) -> str:
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

    # The grounded answers are the high-signal block. They already include
    # analyst figures, named competitors, and inferred inflection drivers.
    has_grounded = any(a is not None for a in grounded_answers)
    if has_grounded:
        parts.append("\n## Grounded research (use these for analyst figures and competitor names)")
        for q, ans in zip(grounded_questions, grounded_answers):
            if ans is None:
                continue
            parts.append(f"\n### Q: {q}\n**Answer (via {ans.source}):**\n{ans.text}")

    parts.append("\n## Web search results — additional URLs (cite as [n], 1-indexed)\n")
    for i, r in enumerate(web, 1):
        snippet = (r.snippet or "")[:300]
        parts.append(f"[{i}] {r.title} — {r.url}\n    {snippet}")

    refs = format_for_prompt(elad)
    if refs:
        parts.append("\n## Elad references\n" + refs)
    return "\n".join(parts)
