"""Co-investors subagent: cap-table analysis + per-investor value-add."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from ..citations import Citation, CitationBook
from ..context import DealContext, Investor
from ..data_sources.search import web_search, SearchResult
from ..retrieval import retrieve, format_for_prompt
from ._llm import load_prompt, render_section


@dataclass
class CoinvestorsResult:
    section_markdown: str
    citations: list[Citation]


async def run_coinvestors(ctx: DealContext, base_system: str) -> CoinvestorsResult:
    if not ctx.existing_investors:
        return CoinvestorsResult(
            section_markdown=(
                "*No existing investors extracted from the inputs.* "
                "If this is a priced round, the cap table belongs in the memo."
            ),
            citations=[],
        )

    per_investor = await _research_investors(ctx.existing_investors)

    book = CitationBook()
    for inv, results in per_investor.items():
        for r in results:
            book.add(Citation(
                key=r.url, title=r.title, url=r.url, snippet=r.snippet, source_type="web",
            ))

    elad = retrieve("co-investor cap table lead investor smart money signaling round dynamics", k=4)

    section_prompt = load_prompt("modules/coinvestors_prompt.md")
    system = f"{base_system}\n\n---\n\n{section_prompt}"
    user = _build_user(ctx, per_investor, elad, book)
    text = await render_section(system=system, user=user, max_tokens=4500)

    return CoinvestorsResult(section_markdown=text, citations=book.citations)


async def _research_investors(investors: list[Investor]) -> dict[str, list[SearchResult]]:
    async def one(inv: Investor) -> tuple[str, list[SearchResult]]:
        q = f'"{inv.name}" venture portfolio recent investments partners'
        return inv.name, await web_search(q, max_results=5)
    pairs = await asyncio.gather(*(one(i) for i in investors[:8]))
    return dict(pairs)


def _build_user(
    ctx: DealContext,
    per_investor: dict[str, list[SearchResult]],
    elad,
    book: CitationBook,
) -> str:
    parts = [
        "# DealContext",
        f"Company: {ctx.company_name}",
        f"Stage: {ctx.stage or '(unknown)'}",
    ]
    parts.append("\n## Existing investors")
    for inv in ctx.existing_investors:
        parts.append(
            f"- **{inv.name}** — type: {inv.type or 'unknown'}, "
            f"round: {inv.round or 'unknown'}, lead: {inv.is_lead}"
        )
    parts.append("\n## Per-investor web research (cite by [n], 1-indexed)")
    for inv_name, results in per_investor.items():
        parts.append(f"\n### {inv_name}")
        for r in results:
            n = book.ref_for(r.url)
            if n is None:
                continue
            parts.append(f"[{n}] {r.title} — {r.url}\n    {r.snippet}")
    refs = format_for_prompt(elad)
    if refs:
        parts.append("\n## Elad references\n" + refs)
    return "\n".join(parts)
