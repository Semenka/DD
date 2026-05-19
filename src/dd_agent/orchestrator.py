"""Top-level orchestrator.

Flow:
  1. Ingest inputs (memo / deck PDF / website URL) → raw text
  2. Normalize via LLM → typed DealContext
  3. Save context, update status
  4. asyncio.gather over the 4 subagents (each its own GPT-5.5 chat completion)
  5. Merge per-section citations into a single global numbering
  6. Synthesis call: GPT-5.5 reads all 4 sections and produces the exec summary +
     Beliefs Required to Invest + Kill Shot + 1-line bet
  7. Render markdown + HTML via report.renderer
  8. Save report into the DealStore
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .citations import Citation, CitationBook
from .context import DealContext
from .delivery import DeliverTo, deliver, extract_one_line_bet
from .ingestion import clipper as clipper_mod
from .ingestion import screenshot_deck as deck_mod
from .ingestion.normalize import normalize
from .ingestion.pdf import extract_text as extract_pdf
from .ingestion.website import fetch_site
from .modules._llm import load_prompt, render_section, rewrite_citations
from .modules.coinvestors import run_coinvestors
from .modules.founders import run_founders
from .modules.market import run_market
from .modules.traction import run_traction
from .report.renderer import render_markdown, render_html, render_pdf
from .state import DealStatus, DealStore

log = logging.getLogger("dd_agent.orchestrator")


@dataclass
class SubmittedDeal:
    deal_id: str
    company_name: str | None
    status: str


async def submit(
    *,
    store: DealStore,
    memo_text: str | None = None,
    memo_path: str | None = None,
    deck_path: str | None = None,
    company_url: str | None = None,
    founder_names: list[str] | None = None,
    deliver_to: dict[str, Any] | None = None,
) -> SubmittedDeal:
    """Create a deal record and kick off the DD pipeline in the background.

    If `deliver_to` is set, dd-agent itself sends the finished report via
    `openclaw message send` — caller doesn't need to poll."""
    record = await store.create()
    asyncio.create_task(_run_pipeline(
        store=store,
        deal_id=record.deal_id,
        memo_text=memo_text,
        memo_path=memo_path,
        deck_path=deck_path,
        company_url=company_url,
        founder_names=founder_names,
        deliver_to=DeliverTo.from_dict(deliver_to),
    ))
    return SubmittedDeal(deal_id=record.deal_id, company_name=None, status=record.status.value)


async def _run_pipeline(
    *,
    store: DealStore,
    deal_id: str,
    memo_text: str | None,
    memo_path: str | None,
    deck_path: str | None,
    company_url: str | None,
    founder_names: list[str] | None,
    deliver_to: DeliverTo | None = None,
) -> None:
    try:
        await store.update_status(deal_id, status=DealStatus.INGESTING,
                                  phase="ingesting inputs", progress_pct=5)

        # If memo came as a file path, read it. memo_text wins when both are set.
        # PDFs go through pypdf; .md / .txt / .markdown route through the
        # Obsidian Web Clipper parser first (which surfaces an embedded deck
        # URL if one is present).
        clipping: clipper_mod.ClippingContext | None = None
        if not memo_text and memo_path and Path(memo_path).exists():
            ext = Path(memo_path).suffix.lower()
            if ext == ".pdf":
                memo_text = await asyncio.to_thread(extract_pdf, memo_path)
            else:
                raw = await asyncio.to_thread(
                    Path(memo_path).read_text, encoding="utf-8", errors="ignore",
                )
                clipping = clipper_mod.parse(raw)
                if clipping is not None:
                    memo_text = clipping.body_text
                    # Promote the clipping's source URL when the caller didn't
                    # provide one.
                    if not company_url and clipping.source_url:
                        company_url = clipping.source_url
                else:
                    memo_text = raw

        deck_text = None
        deck_capture: deck_mod.DeckCapture | None = None
        # 1) Explicit deck_path PDF, if given.
        if deck_path and Path(deck_path).exists():
            deck_text = await asyncio.to_thread(extract_pdf, deck_path)
        # 2) Hosted deck URL discovered in a clipping → screenshot + OCR.
        elif clipping is not None and clipping.deck_url:
            await store.update_status(
                deal_id, phase=f"capturing deck from {clipping.deck_url}", progress_pct=10,
            )
            deck_capture = await deck_mod.capture(clipping.deck_url, deal_id=deal_id)
            if deck_capture.available:
                deck_text = deck_capture.text
                log.info("deck capture: %d slides, %d chars OCR'd",
                         deck_capture.slide_count, len(deck_capture.text))
            else:
                log.info("deck capture unavailable: %s", deck_capture.note)

        site_text = None
        if company_url:
            site_text = await fetch_site(company_url)

        ctx = await normalize(
            memo_text=memo_text,
            deck_text=deck_text,
            website_text=site_text,
            deal_id=deal_id,
        )
        if founder_names and not ctx.founders:
            from .context import Founder
            ctx.founders = [Founder(name=n.strip()) for n in founder_names if n.strip()]
        if not ctx.website and company_url:
            ctx.website = company_url

        await store.update_status(
            deal_id, company_name=ctx.company_name,
            status=DealStatus.RUNNING, phase="running subagents", progress_pct=20,
        )
        await store.save_context(deal_id, ctx.to_dict())

        base_system = load_prompt("prompts/orchestrator.md")

        market_t, founders_t, traction_t, coinvestors_t = await asyncio.gather(
            _safe(run_market(ctx, base_system), "market"),
            _safe(run_founders(ctx, base_system), "founders"),
            _safe(run_traction(ctx, base_system), "traction"),
            _safe(run_coinvestors(ctx, base_system), "co-investors"),
            return_exceptions=False,
        )

        await store.update_status(
            deal_id, phase="merging citations + synthesis", progress_pct=80,
        )

        merged = _merge_sections(market_t, founders_t, traction_t, coinvestors_t)

        synth = await _synthesize(ctx, merged, base_system)

        # 6th call: long-form Bessemer-style memo. Reads everything above
        # (subagent outputs + extras + synthesis) and produces the narrative
        # memo that appears at the top of the final report. Best-effort; if
        # this call fails the rest of the report still ships.
        try:
            bessemer_memo = await _synthesize_bessemer(ctx, merged, synth, base_system)
            merged["bessemer_memo"] = bessemer_memo
        except Exception as exc:  # noqa: BLE001
            log.warning("Bessemer memo synthesis failed: %s", exc)
            merged["bessemer_memo"] = None

        extras = merged.get("extras", {})
        if deck_capture is not None:
            extras["deck_capture"] = {
                "available": deck_capture.available,
                "deck_url": deck_capture.deck_url,
                "slide_count": deck_capture.slide_count,
                "gated": deck_capture.gated,
                "note": deck_capture.note,
                "screenshot_paths": deck_capture.screenshot_paths,
            }
        if clipping is not None:
            extras["clipping"] = {
                "source_url": clipping.source_url,
                "title": clipping.title,
                "author": clipping.author,
                "clipped_at": clipping.clipped_at,
                "deck_url": clipping.deck_url,
            }

        md = render_markdown(
            ctx=ctx,
            synthesis=synth,
            market=merged["market"],
            founders=merged["founders"],
            traction=merged["traction"],
            coinvestors=merged["coinvestors"],
            citations=merged["citations"],
            extras=extras,
            bessemer_memo=merged.get("bessemer_memo"),
        )
        html = render_html(markdown_text=md, deal_context=ctx)

        pdf_path = await _write_pdf(deal_id=deal_id, html=html)
        html_path = await _write_html(deal_id=deal_id, html=html)

        await store.save_report(
            deal_id=deal_id,
            markdown=md,
            html=html,
            citations=merged["citations"].to_list(),
            pdf_path=pdf_path,
        )

        if deliver_to is not None:
            await _deliver_async(
                deliver_to=deliver_to,
                deal_id=deal_id,
                company=ctx.company_name,
                markdown=md,
                html_path=html_path,
                pdf_path=pdf_path,
            )
    except Exception as exc:  # noqa: BLE001
        log.exception("pipeline failed: %s", exc)
        await store.update_status(deal_id, status=DealStatus.FAILED,
                                  phase="failed", error=str(exc))


async def _write_pdf(*, deal_id: str, html: str) -> str | None:
    """Render the HTML report to PDF on disk. Returns the absolute path or
    None on failure (PDF generation is best-effort — the report itself is
    already stored as markdown + html)."""
    out_dir = Path(os.environ.get("DD_DATA_DIR", "./data")) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{deal_id}.pdf"
    try:
        await asyncio.to_thread(render_pdf, html=html, out_path=str(out_path))
        return str(out_path.resolve())
    except Exception as exc:  # noqa: BLE001
        log.warning("PDF rendering failed for %s: %s", deal_id, exc)
        return None


async def _write_html(*, deal_id: str, html: str) -> str | None:
    """Mirror the HTML to disk so we can attach it to outbound deliveries."""
    out_dir = Path(os.environ.get("DD_DATA_DIR", "./data")) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{deal_id}.html"
    try:
        await asyncio.to_thread(out_path.write_text, html, "utf-8")
        return str(out_path.resolve())
    except Exception as exc:  # noqa: BLE001
        log.warning("HTML write failed for %s: %s", deal_id, exc)
        return None


async def _deliver_async(
    *,
    deliver_to: DeliverTo,
    deal_id: str,
    company: str | None,
    markdown: str,
    html_path: str | None,
    pdf_path: str | None,
) -> None:
    one_line = extract_one_line_bet(markdown)
    result = await deliver(
        deliver_to=deliver_to,
        deal_id=deal_id,
        company=company,
        markdown_path=None,
        html_path=html_path,
        pdf_path=pdf_path,
        one_line_bet=one_line,
    )
    if result.get("ok"):
        log.info("deal %s delivered via %s/%s: %s",
                 deal_id, deliver_to.channel, deliver_to.account, result)
    else:
        log.warning("deal %s delivery failed: %s", deal_id, result)


async def _safe(coro, label: str):
    """Wrap a subagent coro; on failure return a stub section instead of raising."""
    try:
        return await coro
    except Exception as exc:  # noqa: BLE001
        log.exception("%s subagent failed", label)
        return _StubResult(
            section_markdown=f"*Subagent failed: `{exc}`.*",
            citations=[],
        )


@dataclass
class _StubResult:
    section_markdown: str
    citations: list[Citation]


def _merge_sections(market, founders, traction, coinvestors) -> dict:
    """Merge per-section [n] citations into a single global numbering.

    Each subagent's `citations` is its local list. We re-add them to a global
    CitationBook in order and remap `[n]` markers in section_markdown via the
    local→global index mapping.
    """
    book = CitationBook()
    sections = {
        "market": market,
        "founders": founders,
        "traction": traction,
        "coinvestors": coinvestors,
    }
    remapped: dict[str, str] = {}
    for name, sec in sections.items():
        local_to_global: dict[int, int] = {}
        for i, c in enumerate(sec.citations, 1):
            global_n = book.add(c)
            local_to_global[i] = global_n
        remapped[name] = rewrite_citations(sec.section_markdown, local_to_global)

    extras = {}
    if hasattr(traction, "reverse_dcf"):
        extras["reverse_dcf"] = traction.reverse_dcf
        extras["sweep"] = traction.sweep
        extras["comp_distribution"] = traction.comp_distribution
    if hasattr(founders, "photo_analyses"):
        extras["photo_analyses"] = [p.to_dict() for p in founders.photo_analyses]
    if hasattr(coinvestors, "funding_rounds"):
        from dataclasses import asdict
        extras["funding_rounds"] = [asdict(r) for r in coinvestors.funding_rounds]
    if hasattr(coinvestors, "notice_co") and coinvestors.notice_co is not None:
        extras["notice_co"] = coinvestors.notice_co.__dict__

    return {
        "market": remapped["market"],
        "founders": remapped["founders"],
        "traction": remapped["traction"],
        "coinvestors": remapped["coinvestors"],
        "citations": book,
        "extras": extras,
    }


_SYNTH_PROMPT = """You are writing the synthesis page of an Elad-Gil-style DD report.

You will be given the 4 already-written sections (Market, Founders, Traction, Co-investors) for the deal. Your job is to produce a tight synthesis that a VC partner reads in 60 seconds. Output ONLY the markdown — do not restate the section content.

Required structure (in this order, with markdown headings):

### Exec summary
4-6 sentences. The whole deal in plain English.

### Beliefs Required to Invest
3-5 numbered propositions. Each is a single declarative sentence that **must be true** for this to be a fund-returner. Phrase as beliefs ("We must believe that X."), not aspirations. This is the Roger Martin / Elad frame.

### Kill Shot
1 paragraph. The single strongest specific reason this deal fails — cross-cutting across the 4 sections. Apply Principle 1.

### 1-line bet
≤ 20 words. The sentence you would say to the partner if they walked in the room right now.

### Recommendation
One of: "Pass", "Pass for now — revisit at X milestone", "Lean in — proceed to references", "Lead this round". One line of justification.

Do not invent new facts. Synthesize from the sections you are given.
"""


_BESSEMER_PROMPT_PATH = "modules/bessemer_prompt.md"


async def _synthesize_bessemer(
    ctx: DealContext, merged: dict, synth: str, base_system: str,
) -> str:
    """6th LLM call — produces the long-form Bessemer-style memo that
    appears at the top of the report (between Synthesis and the analyst
    sections)."""
    body_parts: list[str] = [
        f"# Deal: {ctx.company_name}",
        f"Sector: {ctx.sector or 'unknown'} | Stage: {ctx.stage or 'unknown'} | "
        f"Ask: ${ctx.ask_amount_usd or 'unknown'} at ${ctx.ask_valuation_usd or 'unknown'}",
    ]
    if ctx.founders:
        body_parts.append("Founders: " + ", ".join(f.name for f in ctx.founders))

    body_parts.append(f"\n---\n\n## Synthesis (already written)\n\n{synth}")
    body_parts.append(f"\n---\n\n## Market\n\n{merged.get('market', '')}")
    body_parts.append(f"\n---\n\n## Founders\n\n{merged.get('founders', '')}")
    body_parts.append(f"\n---\n\n## Traction\n\n{merged.get('traction', '')}")
    body_parts.append(f"\n---\n\n## Co-investors\n\n{merged.get('coinvestors', '')}")

    extras = merged.get("extras") or {}
    if extras:
        import json as _json
        # Trim very long arrays before serializing.
        serializable: dict = {}
        for k, v in extras.items():
            if isinstance(v, list) and len(v) > 20:
                serializable[k] = v[:20]
            else:
                serializable[k] = v
        body_parts.append(
            "\n---\n\n## Structured extras (numbers, rounds, photo cohort, etc.)\n\n"
            "```json\n" + _json.dumps(serializable, indent=2, default=str) + "\n```"
        )

    system_prompt = load_prompt(_BESSEMER_PROMPT_PATH)
    system = f"{base_system}\n\n---\n\n{system_prompt}"
    user = "\n".join(body_parts)
    return await render_section(system=system, user=user, max_tokens=5500)


async def _synthesize(ctx: DealContext, merged: dict, base_system: str) -> str:
    body = (
        f"# Deal: {ctx.company_name}\n\n"
        f"Sector: {ctx.sector or 'unknown'} • Stage: {ctx.stage or 'unknown'} • "
        f"Ask: ${ctx.ask_amount_usd or 'unknown'} at ${ctx.ask_valuation_usd or 'unknown'}\n\n"
        f"---\n\n## Market\n\n{merged['market']}\n\n"
        f"---\n\n## Founders\n\n{merged['founders']}\n\n"
        f"---\n\n## Traction\n\n{merged['traction']}\n\n"
        f"---\n\n## Co-investors\n\n{merged['coinvestors']}\n"
    )
    system = f"{base_system}\n\n---\n\n{_SYNTH_PROMPT}"
    return await render_section(system=system, user=body, max_tokens=2500)
