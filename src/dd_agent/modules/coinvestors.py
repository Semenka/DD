"""Co-investors subagent: cap-table analysis + per-investor value-add +
detailed round-by-round funding history + notice.co secondary-market snapshot.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from ..citations import Citation, CitationBook
from ..context import DealContext, FundingRound, Investor, NoticeCoSnapshot
from ..data_sources import funding_rounds as fr_mod
from ..data_sources import notice_co as nco_mod
from ..data_sources.search import SearchResult, web_search
from ..retrieval import format_for_prompt, retrieve
from ._llm import load_prompt, render_section


@dataclass
class CoinvestorsResult:
    section_markdown: str
    citations: list[Citation]
    funding_rounds: list[FundingRound] = field(default_factory=list)
    notice_co: NoticeCoSnapshot | None = None


async def run_coinvestors(ctx: DealContext, base_system: str) -> CoinvestorsResult:
    # Gather everything in parallel: per-investor web search, funding history,
    # notice.co snapshot. None of these are blocking on each other.
    investor_task = _research_investors(ctx.existing_investors) if ctx.existing_investors else _empty_investors()
    funding_task = fr_mod.discover_rounds(ctx.company_name)
    notice_task = nco_mod.fetch_snapshot(ctx.company_name)

    per_investor, funding_pair, notice_snapshot = await asyncio.gather(
        investor_task,
        funding_task,
        notice_task,
    )
    rounds, funding_sources = funding_pair

    book = CitationBook()
    for inv, results in per_investor.items():
        for r in results:
            book.add(Citation(
                key=r.url, title=r.title, url=r.url, snippet=r.snippet, source_type="web",
            ))
    for r in funding_sources:
        book.add(Citation(
            key=r.url, title=r.title, url=r.url, snippet=r.snippet, source_type="web",
        ))
    if notice_snapshot.source_url:
        book.add(Citation(
            key=notice_snapshot.source_url,
            title=f"notice.co — {ctx.company_name}",
            url=notice_snapshot.source_url,
            snippet=(notice_snapshot.note or "notice.co secondary-market snapshot"),
            source_type="notice_co",
        ))

    elad = retrieve(
        "co-investor cap table lead investor smart money signaling round dynamics valuation",
        k=4,
    )

    section_prompt = load_prompt("modules/coinvestors_prompt.md")
    system = f"{base_system}\n\n---\n\n{section_prompt}"
    user = _build_user(ctx, per_investor, rounds, notice_snapshot, elad, book)
    # v8: 5500 → 2400 tokens
    text = await render_section(system=system, user=user, max_tokens=2400)

    return CoinvestorsResult(
        section_markdown=text,
        citations=book.citations,
        funding_rounds=rounds,
        notice_co=notice_snapshot,
    )


async def _empty_investors() -> dict[str, list[SearchResult]]:
    return {}


async def _research_investors(investors: list[Investor]) -> dict[str, list[SearchResult]]:
    async def one(inv: Investor) -> tuple[str, list[SearchResult]]:
        q = f'"{inv.name}" venture portfolio recent investments partners'
        return inv.name, await web_search(q, max_results=5)
    pairs = await asyncio.gather(*(one(i) for i in investors[:8]))
    return dict(pairs)


def _build_user(
    ctx: DealContext,
    per_investor: dict[str, list[SearchResult]],
    rounds: list[FundingRound],
    notice: NoticeCoSnapshot,
    elad,
    book: CitationBook,
) -> str:
    parts = [
        "# DealContext",
        f"Company: {ctx.company_name}",
        f"Stage: {ctx.stage or '(unknown)'}",
    ]

    if ctx.existing_investors:
        parts.append("\n## Existing investors (from memo)")
        for inv in ctx.existing_investors:
            parts.append(
                f"- **{inv.name}** — type: {inv.type or 'unknown'}, "
                f"round: {inv.round or 'unknown'}, lead: {inv.is_lead}"
            )

    if rounds:
        parts.append("\n## Funding rounds discovered (cite via the [n] you assign in the table)")
        parts.append(json.dumps(fr_mod.to_jsonable(rounds), indent=2, default=str))
    else:
        parts.append("\n## Funding rounds\n*(none found via free-tier search — say so explicitly in the section)*")

    parts.append("\n## notice.co secondary-market snapshot")
    parts.append(json.dumps(notice.__dict__, indent=2, default=str))

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
