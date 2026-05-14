"""Traction subagent: ARR vs public comps + reverse DCF + independent voice."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from ..citations import Citation, CitationBook
from ..context import DealContext
from ..data_sources.reviews import gather_for_company, flatten
from ..models import comps as comps_mod
from ..models import reverse_dcf as rdcf_mod
from ..retrieval import retrieve, format_for_prompt
from ._llm import load_prompt, render_section


@dataclass
class TractionResult:
    section_markdown: str
    citations: list[Citation]
    comp_distribution: dict | None
    reverse_dcf: dict | None
    sweep: list[dict]


async def run_traction(ctx: DealContext, base_system: str) -> TractionResult:
    review_buckets, dist = await asyncio.gather(
        gather_for_company(ctx.company_name),
        comps_mod.build_distribution(ctx.sector),
    )

    book = CitationBook()
    for r in flatten(review_buckets):
        book.add(Citation(
            key=r.url, title=r.title, url=r.url, snippet=r.snippet, source_type=r.source,
        ))

    rdcf_result = None
    sweep_rows: list[dict] = []
    if ctx.ask_valuation_usd and ctx.metrics.arr_usd:
        rdcf_result = rdcf_mod.run(
            ask_valuation_usd=ctx.ask_valuation_usd,
            current_arr_usd=ctx.metrics.arr_usd,
            growth_percentile_fn=lambda g: dist.growth_percentile(g),
        ).to_dict()
        sweep_rows = rdcf_mod.sweep(
            ask_valuation_usd=ctx.ask_valuation_usd,
            current_arr_usd=ctx.metrics.arr_usd,
        )

    elad = retrieve(f"rule of 40 magic number ARR growth multiples SaaS {ctx.sector or ''}", k=4)

    section_prompt = load_prompt("modules/traction_prompt.md")
    system = f"{base_system}\n\n---\n\n{section_prompt}"
    user = _build_user(ctx, dist, rdcf_result, sweep_rows, review_buckets, elad, book)
    text = await render_section(system=system, user=user, max_tokens=5000)

    return TractionResult(
        section_markdown=text,
        citations=book.citations,
        comp_distribution=dist.to_dict(),
        reverse_dcf=rdcf_result,
        sweep=sweep_rows,
    )


def _build_user(
    ctx: DealContext,
    dist,
    rdcf_result,
    sweep_rows,
    review_buckets,
    elad,
    book: CitationBook,
) -> str:
    parts = [
        "# DealContext",
        f"Company: {ctx.company_name}",
        f"Sector: {ctx.sector or '(unknown)'}",
        f"Ask: ${ctx.ask_amount_usd or 'unknown'} at ${ctx.ask_valuation_usd or 'unknown'} valuation",
        f"Metrics: {json.dumps(ctx.metrics.__dict__, default=str)}",
    ]
    parts.append("\n## Public comp distribution\n" + json.dumps(dist.to_dict(), indent=2, default=str))
    if rdcf_result is not None:
        parts.append("\n## Reverse DCF result\n" + json.dumps(rdcf_result, indent=2, default=str))
    else:
        parts.append("\n## Reverse DCF\n*Could not run — need both ask_valuation_usd and current ARR.*")
    if sweep_rows:
        parts.append("\n## Reverse DCF sweep\n" + json.dumps(sweep_rows, indent=2, default=str))

    parts.append("\n## Review/social signals (cite by [n], 1-indexed)")
    for c in book.citations:
        n = book.ref_for(c.key)
        parts.append(f"[{n}] ({c.source_type}) {c.title} — {c.url}\n    {c.snippet or ''}")

    refs = format_for_prompt(elad)
    if refs:
        parts.append("\n## Elad references\n" + refs)
    return "\n".join(parts)
