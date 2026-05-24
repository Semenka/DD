"""Founders subagent: track record, integrity, energy, founder/market fit + photo classifier."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from ..citations import Citation, CitationBook
from ..context import DealContext, Founder
from ..data_sources.github import fetch_founder as fetch_gh
from ..data_sources.social import gather_for_founder, SocialSignal
from ..retrieval import retrieve, format_for_prompt
from .photo_classifier import analyze_founder_photo, PhotoAnalysis
from ._llm import load_prompt, render_section


@dataclass
class FoundersResult:
    section_markdown: str
    citations: list[Citation]
    photo_analyses: list[PhotoAnalysis]


async def run_founders(ctx: DealContext, base_system: str) -> FoundersResult:
    if not ctx.founders:
        return FoundersResult(
            section_markdown=(
                "*No founder data extracted from the inputs.* "
                "Add LinkedIn URLs or names to the memo and re-submit."
            ),
            citations=[],
            photo_analyses=[],
        )

    gh_data, social_data, photo_data = await asyncio.gather(
        _gather_github(ctx.founders),
        _gather_social(ctx),
        _gather_photos(ctx.founders, deal_id=ctx.deal_id),
    )

    book = CitationBook()
    all_signals: list[SocialSignal] = []
    for sigs in social_data.values():
        all_signals.extend(sigs)
    for s in all_signals:
        book.add(Citation(
            key=s.url,
            title=s.title,
            url=s.url,
            snippet=s.snippet,
            source_type="web" if s.platform == "press" else s.platform,
        ))

    elad = retrieve("founder market fit relentless resourcefulness shipped artifacts integrity", k=4)

    section_prompt = load_prompt("modules/founders_prompt.md")
    system = f"{base_system}\n\n---\n\n{section_prompt}"
    user = _build_user(ctx, gh_data, social_data, photo_data, elad, book)
    # v8: 5000 → 2400 tokens
    text = await render_section(system=system, user=user, max_tokens=2400)

    return FoundersResult(
        section_markdown=text,
        citations=book.citations,
        photo_analyses=photo_data,
    )


async def _gather_github(founders: list[Founder]) -> dict[str, dict]:
    async def one(f: Founder) -> tuple[str, dict | None]:
        if not f.github_handle:
            return f.name, None
        data = await fetch_gh(f.github_handle)
        return f.name, (data.__dict__ if data else None)
    results = await asyncio.gather(*(one(f) for f in founders))
    return {n: d for n, d in results if d}


async def _gather_social(ctx: DealContext) -> dict[str, list[SocialSignal]]:
    async def one(f: Founder) -> tuple[str, list[SocialSignal]]:
        sigs = await gather_for_founder(
            name=f.name, company=ctx.company_name, twitter_handle=f.twitter_handle,
        )
        return f.name, sigs
    results = await asyncio.gather(*(one(f) for f in ctx.founders))
    return dict(results)


async def _gather_photos(
    founders: list[Founder],
    deal_id: str | None = None,
) -> list[PhotoAnalysis]:
    """Run the photo classifier per founder in parallel. Each photo is
    persisted under data/reports/photos/{deal_id}/ so the report can embed
    it via markdown image link + inline base64 HTML."""
    from pathlib import Path
    save_dir = None
    if deal_id:
        import os
        save_dir = Path(os.environ.get("DD_DATA_DIR", "./data")) / "reports" / "photos" / deal_id

    async def one(f: Founder) -> PhotoAnalysis:
        return await analyze_founder_photo(
            founder_name=f.name, photo_url=f.photo_url, save_dir=save_dir,
        )
    return list(await asyncio.gather(*(one(f) for f in founders)))


def _build_user(
    ctx: DealContext,
    gh_data: dict[str, dict],
    social_data: dict[str, list[SocialSignal]],
    photos: list[PhotoAnalysis],
    elad,
    book: CitationBook,
) -> str:
    parts = [
        "# DealContext",
        f"Company: {ctx.company_name}",
        f"Sector: {ctx.sector or '(unknown)'}",
        f"Stage: {ctx.stage or '(unknown)'}",
    ]
    parts.append("\n## Founders")
    for f in ctx.founders:
        parts.append(
            f"- **{f.name}** ({f.role or 'role unknown'}) — "
            f"LinkedIn: {f.linkedin_url or '(unknown)'}, "
            f"Twitter: @{f.twitter_handle or '(unknown)'}, "
            f"GitHub: {f.github_handle or '(unknown)'}"
        )
        if f.bio:
            parts.append(f"  Bio: {f.bio}")
        if f.prior_companies:
            parts.append(f"  Prior: {', '.join(f.prior_companies)}")

    parts.append("\n## GitHub data (raw)")
    parts.append(json.dumps(gh_data, indent=2, default=str))

    parts.append("\n## Social signals (cite by [n], 1-indexed)")
    for s in book.citations:
        n = book.ref_for(s.key)
        parts.append(f"[{n}] {s.title} ({s.source_type}) — {s.url}\n    {s.snippet or ''}")

    parts.append("\n## Photo analysis")
    for p in photos:
        parts.append(json.dumps(p.to_dict(), indent=2, default=str))

    refs = format_for_prompt(elad)
    if refs:
        parts.append("\n## Elad references\n" + refs)

    return "\n".join(parts)
