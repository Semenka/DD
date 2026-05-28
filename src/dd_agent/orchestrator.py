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
from .data_sources import exits as exits_mod
from .data_sources import founder_photo as founder_photo_mod
from .delivery import DeliverTo, deliver, extract_one_line_bet
from .ingestion import clipper as clipper_mod
from .ingestion import screenshot_deck as deck_mod
from .ingestion.identity import verify_company_identity
from .ingestion.normalize import normalize
from .ingestion.pdf import extract_text as extract_pdf
from .ingestion.pdf import extract_document
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
    """Create a deal record and kick off the DD pipeline in a DETACHED subprocess.

    Why a subprocess, not asyncio.create_task: the MCP server's parent process
    (OpenClaw) typically closes stdin to dd-agent right after the submit_deal
    call returns. When asyncio.run()'s loop sees EOF on stdin it tears down,
    cancelling any background tasks — including a pipeline that might still
    have 5 minutes of work to do. We saw this bug live with the Alfred deal
    (May 20 16:10): submit returned deal_id, the asyncio task was cancelled at
    16:10:14 in 'running subagents' phase, deal stuck forever.

    The detached subprocess survives OpenClaw closing the MCP stdio because
    it has its own session (start_new_session=True). It writes its result
    back to the same SQLite DB the MCP server reads from."""
    record = await store.create()
    await _spawn_pipeline_subprocess(
        deal_id=record.deal_id,
        memo_text=memo_text,
        memo_path=memo_path,
        deck_path=deck_path,
        company_url=company_url,
        founder_names=founder_names,
        deliver_to=deliver_to,
    )
    return SubmittedDeal(deal_id=record.deal_id, company_name=None, status=record.status.value)


async def _spawn_pipeline_subprocess(
    *,
    deal_id: str,
    memo_text: str | None,
    memo_path: str | None,
    deck_path: str | None,
    company_url: str | None,
    founder_names: list[str] | None,
    deliver_to: dict[str, Any] | None,
) -> None:
    """Spawn `dd-agent process-deal <deal_id>` as a detached subprocess.

    The payload (memo_text + paths + deliver_to) is passed via a temp JSON
    file because (a) memo_text can be many KB and (b) we don't want to leak
    the deliver_to channel token via process argv. The child reads + unlinks
    the file at startup.
    """
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    payload = {
        "memo_text": memo_text,
        "memo_path": memo_path,
        "deck_path": deck_path,
        "company_url": company_url,
        "founder_names": founder_names,
        "deliver_to": deliver_to,
    }
    queue_dir = Path(os.environ.get("DD_DATA_DIR", "./data")) / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    payload_path = queue_dir / f"{deal_id}.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False))

    # Prefer the dd-agent CLI on PATH. Fall back to invoking the same Python
    # interpreter that's running the server (lets us survive uv-managed venvs
    # without dd-agent being on the inherited PATH).
    bin_path = shutil.which("dd-agent")
    if bin_path:
        cmd = [bin_path, "process-deal", deal_id, str(payload_path)]
    else:
        cmd = [sys.executable, "-m", "dd_agent.server", "process-deal", deal_id, str(payload_path)]

    log_path = queue_dir / f"{deal_id}.log"
    log_fh = open(log_path, "ab")
    try:
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            close_fds=True,
            start_new_session=True,        # detach from MCP server's session
            env={**os.environ},            # inherit env so .env was loaded
        )
        log.info("spawned detached pipeline for deal %s (pid via Popen, log=%s)",
                 deal_id, log_path)
    finally:
        # Parent can close its handle — the child inherits its own fd.
        log_fh.close()


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

        # If memo came as a file path, dispatch by extension (.pdf, .docx,
        # .md, .txt, .markdown). DOCX support was added in v7 — before that
        # a .docx file fell through to read_text() and the ZIP magic bytes
        # PK\\x03\\x04 were extracted as the company name.
        clipping: clipper_mod.ClippingContext | None = None
        if not memo_text and memo_path and Path(memo_path).exists():
            ext = Path(memo_path).suffix.lower()
            if ext in (".md", ".markdown", ".txt"):
                # Markdown files route through the Obsidian Web Clipper parser
                # first (which surfaces an embedded deck URL if one is present).
                raw = await asyncio.to_thread(
                    Path(memo_path).read_text, encoding="utf-8", errors="ignore",
                )
                clipping = clipper_mod.parse(raw)
                if clipping is not None:
                    memo_text = clipping.body_text
                    if not company_url and clipping.source_url:
                        company_url = clipping.source_url
                else:
                    memo_text = raw
            else:
                # PDF / DOCX / other — let the dispatcher pick the extractor.
                try:
                    memo_text = await asyncio.to_thread(extract_document, memo_path)
                except NotImplementedError as exc:
                    log.warning("unsupported memo format: %s", exc)
                    memo_text = ""

        deck_text = None
        deck_capture: deck_mod.DeckCapture | None = None
        # 1) Explicit deck_path file (PDF, DOCX, or text), if given.
        if deck_path and Path(deck_path).exists():
            try:
                deck_text = await asyncio.to_thread(extract_document, deck_path)
            except NotImplementedError as exc:
                log.warning("unsupported deck format: %s", exc)
                deck_text = None
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

        # Identity verification (v7): the .docx dispatcher fixes the
        # ZIP-magic-bytes case, but the LLM can still pick up a section
        # header ("PK", "TERMS", "Round Seed") or a generic word
        # ("Investment") as the company name. Verify before burning ~14
        # minutes of pipeline time on the wrong company.
        await store.update_status(
            deal_id, phase="verifying company identity", progress_pct=15,
        )
        verify = await verify_company_identity(
            extracted_name=ctx.company_name,
            raw_memo=memo_text or "",
            raw_deck=deck_text or "",
            source_filename=memo_path or deck_path,
        )
        if not verify.verified:
            await store.update_status(
                deal_id,
                status=DealStatus.FAILED,
                phase="identity-mismatch — refused",
                error=verify.notes or "company identity could not be confirmed",
            )
            return
        if verify.company_name and verify.company_name != ctx.company_name:
            log.info(
                "identity verifier corrected company name: %r → %r (source=%s)",
                ctx.company_name, verify.company_name, verify.source,
            )
            await store.update_status(
                deal_id,
                phase=(
                    f"identity-corrected: {verify.original_name!r} → "
                    f"{verify.company_name!r} (via {verify.source})"
                ),
            )
            ctx.company_name = verify.company_name

        # v8: post-normalize founder photo cascade. Each founder gets a
        # 6-tier discovery (deck-slide face crop → Wikipedia → company /team
        # → LinkedIn og:image → grounded LLM → clipping embedded images).
        # Best-effort — pipeline continues regardless. Result is stored to
        # `founder.photo_url` for the downstream photo classifier to use.
        if ctx.founders:
            await store.update_status(
                deal_id, phase="discovering founder photos", progress_pct=18,
            )
            try:
                resolved = await founder_photo_mod.resolve_all_founder_photos(
                    ctx=ctx, deck_capture=deck_capture, clipping=clipping,
                )
                hits = sum(1 for v in resolved.values() if v)
                log.info("founder-photo cascade: %d/%d founders resolved",
                         hits, len(ctx.founders))
            except Exception as exc:  # noqa: BLE001
                log.warning("founder-photo cascade failed: %s", exc)

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

        # Series B+ only: pull comparable exits via grounded LLM. Adds ~10s
        # but gives the Bessemer memo a real "Comparable exits" table.
        if (ctx.stage or "") in ("series_b", "series_c_plus", "growth"):
            try:
                exits_result = await exits_mod.discover_exits(
                    ctx.company_name, sector=ctx.sector,
                )
                if exits_result.comps:
                    merged.setdefault("extras", {})["comparable_exits"] = exits_mod.to_jsonable(
                        exits_result,
                    )
                    log.info("discovered %d comparable exits for %s",
                             len(exits_result.comps), ctx.company_name)
            except Exception as exc:  # noqa: BLE001
                log.warning("comparable-exits lookup failed: %s", exc)

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

        # v8: build the inline-charts bundle (SVG percentile ruler, founder
        # trait bars, matplotlib heatmap + timeline). Best-effort — any chart
        # that fails to render returns '' and the template skips it.
        from .report import charts as _charts_mod
        try:
            charts = _charts_mod.build_chart_bundle(extras=extras)
        except Exception as exc:  # noqa: BLE001
            log.warning("chart bundle build failed: %s", exc)
            charts = {}

        def _assemble(synthesis_text: str) -> str:
            return render_markdown(
                ctx=ctx,
                synthesis=synthesis_text,
                market=merged["market"],
                founders=merged["founders"],
                traction=merged["traction"],
                coinvestors=merged["coinvestors"],
                citations=merged["citations"],
                extras=extras,
                bessemer_memo=merged.get("bessemer_memo"),
                charts=charts,
            )

        md = _assemble(synth)

        # v10: Quality Gate. Score the assembled report before it ships so
        # garbage (wrong company, empty sections) is never silently delivered.
        # Bounded: at most ONE synthesis retry, then ship with a confidence
        # banner. The gate can NEVER block shipping — any gate error falls
        # through to "ship with deterministic score only".
        quality_score: float | None = None
        quality_notes: str | None = None
        try:
            await store.update_status(
                deal_id, phase="scoring report quality", progress_pct=92,
            )
            md, quality_score, quality_notes = await _quality_gate(
                ctx=ctx, merged=merged, extras=extras, base_system=base_system,
                synth=synth, assemble=_assemble, store=store, deal_id=deal_id,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("quality gate errored (%s) — shipping ungated", exc)

        # v6: collect founder photo base64 strings so render_html can embed
        # them as inline data URLs (self-contained HTML for Telegram).
        photo_b64_by_path: dict[str, str] = {}
        try:
            if hasattr(founders_t, "photo_analyses"):
                for pa in founders_t.photo_analyses:
                    if pa.photo_path and pa.photo_base64:
                        photo_b64_by_path[pa.photo_path] = pa.photo_base64
        except Exception:
            pass

        html = render_html(
            markdown_text=md,
            deal_context=ctx,
            photo_b64_by_path=photo_b64_by_path or None,
        )

        pdf_path = await _write_pdf(deal_id=deal_id, html=html)
        html_path = await _write_html(deal_id=deal_id, html=html)

        await store.save_report(
            deal_id=deal_id,
            markdown=md,
            html=html,
            citations=merged["citations"].to_list(),
            pdf_path=pdf_path,
            quality_score=quality_score,
            quality_notes=quality_notes,
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


async def _quality_gate(
    *,
    ctx: DealContext,
    merged: dict,
    extras: dict,
    base_system: str,
    synth: str,
    assemble,                 # callable(synthesis_text) -> markdown
    store: DealStore,
    deal_id: str,
) -> tuple[str, float, str]:
    """v10 Quality Gate. Scores the assembled report, applies the
    ship/retry/flag decision, and returns (final_markdown, score, notes).

    Decision branches:
      - score >= PASS_THRESHOLD (7): ship, prepend a `> Quality: N/10` line.
      - RETRY_FLOOR (4) <= score < 7: re-run the synthesis ONCE with the
        reviewer critique injected (bounded — one extra LLM call), re-assemble,
        re-score once, ship with a `(after 1 retry)` note.
      - score < 4: ship but prepend a prominent LOW CONFIDENCE banner and set
        the status phase to `low-quality — needs review`.

    A company-identity failure short-circuits to the flag branch — no retry can
    fix the wrong company, so we don't waste an LLM call on it."""
    from .report.quality import score_report, PASS_THRESHOLD, RETRY_FLOOR

    md = assemble(synth)
    report = await score_report(ctx=ctx, markdown=md, merged=merged, extras=extras)
    log.info("quality gate: deal %s scored %.1f (det=%.1f, llm=%s) — %s",
             deal_id, report.score, report.deterministic_score,
             report.llm_score, report.verdict)

    identity_failed = "company_identity" in report.failed_checks

    # --- ship clean ---
    if report.score >= PASS_THRESHOLD:
        banner = f"> **Quality: {report.score:.0f}/10** — {report.verdict}\n\n"
        return banner + md, report.score, report.verdict

    # --- bounded retry (skip when identity is the problem) ---
    if report.score >= RETRY_FLOOR and not identity_failed:
        why = ""
        if report.weakest_sections:
            ws = report.weakest_sections[0]
            why = f"{ws.get('section', '')}: {ws.get('why', '')}".strip(": ")
        log.info("quality gate: deal %s in retry band — re-running synthesis "
                 "with critique: %s", deal_id, why or "(general)")
        await store.update_status(
            deal_id, phase="quality retry — re-synthesizing", progress_pct=94,
        )
        try:
            critique = (
                "\n\n## REVIEWER CRITIQUE (address this directly)\n"
                f"A partner flagged the previous draft: {why or report.verdict}. "
                "Tighten the weak area, make the call more decisive, and ensure "
                "every pillar is backed by a specific fact or a sharp diligence "
                "question — not a hedge."
            )
            synth2 = await _synthesize(ctx, merged, base_system + critique)
            md2 = assemble(synth2)
            report2 = await score_report(
                ctx=ctx, markdown=md2, merged=merged, extras=extras,
            )
            log.info("quality gate: deal %s rescored %.1f after retry (was %.1f)",
                     deal_id, report2.score, report.score)
            # Keep whichever draft scored higher.
            if report2.score >= report.score:
                banner = (f"> **Quality: {report2.score:.0f}/10** "
                          f"(after 1 retry) — {report2.verdict}\n\n")
                return banner + md2, report2.score, report2.verdict
        except Exception as exc:  # noqa: BLE001
            log.warning("quality retry failed (%s) — shipping original draft", exc)
        banner = (f"> **Quality: {report.score:.0f}/10** — {report.verdict}\n\n")
        return banner + md, report.score, report.verdict

    # --- flag (score < 4, or identity failed) ---
    top = ", ".join(report.failed_checks[:3]) or "multiple checks"
    await store.update_status(
        deal_id, phase="low-quality — needs review",
    )
    banner = (
        f"> ⚠️ **LOW CONFIDENCE — {report.score:.0f}/10.** {report.verdict}\n"
        f">\n"
        f"> This report did not pass quality checks (failed: {top}). "
        f"Treat its conclusions with caution and re-submit with a clearer "
        f"memo, the company name in the filename, or a deck PDF.\n\n"
    )
    notes = f"LOW CONFIDENCE: failed {top}. {report.verdict}"
    return banner + md, report.score, notes


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


_SYNTH_PROMPT = """You are writing the synthesis page of a Bessemer/Sequoia-grade DD report.

You will be given the 4 already-written sections (Market, Founders, Traction, Co-investors) for the deal, plus structured extras (founder photo profiles, ARR quality classification, existing investors). Your job is to produce a tight synthesis that a VC partner reads in 60 seconds. Output ONLY the markdown — do not restate the section content.

## OMISSION DISCIPLINE *(v8 — read before writing)*

If a fact is wholly unknown and cannot be inferred from disclosed data, **OMIT IT.** Do not write *"data is undisclosed"*, *"unknown"*, *"NRR is not disclosed"*, *"(speculation: …)"* or any absence-narrative. Either cite the number or skip the sentence. If an entire pillar has zero disclosed facts — skip that pillar entirely (its header included). Better to have 3 strong pillars than 4 pillars where one is hedge-padding.

The downstream Recommendation block is the only place where missing data can be referenced — and only via a sharp diligence question, never as a hedge.

## LENGTH BUDGET *(v8 — hard cap)*

The entire synthesis page (Exec summary + Beliefs + Kill Shot + 1-line bet + Recommendation) must be **≤ 600 words total**. At 11pt print this is ~1.5 pages. **Compress. Cut adjectives. Cut sentences that restate what their header already says.**

Required structure (in this order, with markdown headings):

### Exec summary

A factual, Bessemer-style 4-pillar scorecard. Each pillar is **one short labeled paragraph (2-3 sentences)**, not a bullet list. Cite `[n]` from the global References pool whenever you state a number, a partner name, or a fact from a source. Mark `(speculation)` ONLY for genuinely unsupported claims — facts from the memo, the deck, or cited references do NOT require the marker.

**Founders.** Name each founder. Quantify their prior reps ("scaled X from $Y to $Z ARR", "led team of N at Z", "shipped product used by W"). If a `<photo_profile>` block is provided below, integrate its character read in ONE sentence — name the dominant traits with their percentile ("intensity 92nd pct vs unicorn corpus, closest archetype: Technical visionary") and the nearest-archetype call. Immediately after this paragraph, embed the first founder's photo as standard markdown: `![{founder_name}]({photo_path})` using the `photo_path` from the photo profile block. If no photo is available, skip the image and say so in one short clause.

**Co-investors.** Name the specific VCs and angels already on the cap table. Prefer named partners where disclosed ("Sequoia (Roelof Botha) led the Series B"). Distinguish top-tier signals (Sequoia, Benchmark, Founders Fund, a16z, Accel, Greylock, Index, Lightspeed, Kleiner, Khosla) and super-angels (Naval Ravikant, Elad Gil, Lachy Groom, Balaji, Sam Altman, Garry Tan, Tobi Lutke, etc.) from generic participation. If no named lead and no named participants exist, state plainly: "Cap table is opaque — no named lead, no named participants disclosed."

**Growth metrics.** Lead with ARR + YoY growth, then state the revenue-quality call explicitly using the `arr_quality` taxonomy you are given (`recurring_subscription` / `annualized_contracts` / `annualized_pilots` / `annualized_transactions` / `gmv_or_take_rate` / `one_time_hardware` / `unclear`). Distinguish **stated** vs **implied** vs **unknown** for each number. Include NRR / Magic Number / burn multiple ONLY when disclosed; do not invent. If headline revenue is GMV or annualized pilots, say so — do not present it as ARR.

**Competitive position.** Name 2-4 specific competitors with their funding state. Close the paragraph with ONE explicit verdict on monopoly likelihood — pick exactly one label from this controlled vocabulary: **category winner** / **co-leader** / **challenger** / **commodity** / **uncertain**. Give the structural reason (data network effect / regulatory moat / switching cost / scale economy / brand / none). Reserve "uncertain" only when the evidence is genuinely thin.

### Beliefs Required to Invest
3-5 numbered propositions. Each is a single declarative sentence that **must be true** for this to be a fund-returner. Phrase as beliefs ("We must believe that X."), not aspirations. This is the Roger Martin / Elad frame.

### Kill Shot
1 paragraph. The single strongest specific reason this deal fails — cross-cutting across the 4 sections. Apply Principle 1.

### 1-line bet
≤ 20 words. The sentence you would say to the partner if they walked in the room right now.

### Recommendation
One of: "Pass", "Pass for now — revisit at X milestone", "Lean in — proceed to references", "Lead this round". One line of justification.

Do not invent new facts. Synthesize from the sections and the structured extras you are given.
"""


_BESSEMER_PROMPT_PATH = "modules/bessemer_prompt.md"


async def _synthesize_bessemer(
    ctx: DealContext, merged: dict, synth: str, base_system: str,
) -> str:
    """6th LLM call — produces the long-form Bessemer-style memo that
    appears at the top of the report (between Synthesis and the analyst
    sections)."""
    stage = ctx.stage or "series_a"  # default depth when undisclosed
    body_parts: list[str] = [
        f"STAGE: {stage}",
        "",
        f"# Deal: {ctx.company_name}",
        f"Sector: {ctx.sector or 'unknown'} | Stage: {stage} | "
        f"Ask: ${ctx.ask_amount_usd or 'unknown'} at ${ctx.ask_valuation_usd or 'unknown'}",
    ]
    if ctx.founders:
        body_parts.append("Founders: " + ", ".join(f.name for f in ctx.founders))

    # Surface photo profile prose summaries up front so the prompt can find
    # them easily (the prompt's Team section specifically looks for these).
    photo_analyses = (merged.get("extras") or {}).get("photo_analyses") or []
    if photo_analyses:
        body_parts.append("\n## Photo profiles (use summary_for_prompt in Team section)")
        for p in photo_analyses:
            summary = p.get("summary_for_prompt") if isinstance(p, dict) else None
            if summary:
                body_parts.append(f"\n<photo_profile>\n{summary}\n</photo_profile>")

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
    # v8: 5500 → 2200 tokens. The Bessemer memo must fit in ~3.5 print pages
    # with the synthesis page above it landing the deliverable at 5 pages.
    return await render_section(system=system, user=user, max_tokens=2200)


async def _synthesize(ctx: DealContext, merged: dict, base_system: str) -> str:
    parts: list[str] = [
        f"# Deal: {ctx.company_name}",
        "",
        f"Sector: {ctx.sector or 'unknown'} • Stage: {ctx.stage or 'unknown'} • "
        f"Ask: ${ctx.ask_amount_usd or 'unknown'} at ${ctx.ask_valuation_usd or 'unknown'}",
    ]

    # --- Structured extras that the 4-pillar Exec Summary leans on ---
    extras = merged.get("extras") or {}

    # Founder photo profiles for the Founders pillar (character read + image embed).
    photo_analyses = extras.get("photo_analyses") or []
    if photo_analyses:
        parts.append("\n## Founder photo profiles")
        parts.append(
            "Use these in the **Founders.** pillar. Embed the first founder's photo "
            "via `![{name}]({photo_path})` immediately after that paragraph."
        )
        for p in photo_analyses:
            if not isinstance(p, dict):
                continue
            summary = p.get("summary_for_prompt") or ""
            photo_path = p.get("photo_path")
            name = p.get("founder_name") or "founder"
            parts.append(
                f"\n<photo_profile>\nfounder: {name}\nphoto_path: {photo_path or '—'}\n{summary}\n</photo_profile>"
            )

    # ARR quality classification — feeds the **Growth metrics.** pillar.
    m = ctx.metrics
    arr_lines: list[str] = []
    if m.arr_usd is not None:
        arr_lines.append(f"arr_usd: ${m.arr_usd:,.0f}")
    if m.mrr_usd is not None:
        arr_lines.append(f"mrr_usd: ${m.mrr_usd:,.0f}")
    if m.growth_rate_yoy is not None:
        arr_lines.append(f"growth_rate_yoy: {m.growth_rate_yoy:.2f} (i.e. {int(m.growth_rate_yoy * 100)}%)")
    if m.gmv_usd is not None:
        arr_lines.append(f"gmv_usd: ${m.gmv_usd:,.0f}")
    if m.net_retention is not None:
        arr_lines.append(f"net_retention: {m.net_retention:.2f}")
    if m.burn_usd_monthly is not None:
        arr_lines.append(f"burn_usd_monthly: ${m.burn_usd_monthly:,.0f}")
    if m.arr_quality:
        arr_lines.append(f"arr_quality: {m.arr_quality}")
    if m.arr_quality_notes:
        arr_lines.append(f"arr_quality_notes: {m.arr_quality_notes}")
    if arr_lines:
        parts.append("\n## Growth metrics + ARR quality (use in Growth metrics pillar)")
        parts.append("```\n" + "\n".join(arr_lines) + "\n```")

    # Existing investors — feeds the **Co-investors.** pillar.
    if ctx.existing_investors:
        parts.append("\n## Existing investors (use in Co-investors pillar)")
        inv_lines = []
        for inv in ctx.existing_investors:
            tag = []
            if inv.is_lead:
                tag.append("LEAD")
            if inv.type:
                tag.append(inv.type)
            if inv.round:
                tag.append(inv.round)
            tag_str = f" [{', '.join(tag)}]" if tag else ""
            inv_lines.append(f"- {inv.name}{tag_str}")
        parts.append("\n".join(inv_lines))

    # Funding rounds (often the cleanest source for named lead investors).
    funding_rounds = extras.get("funding_rounds") or []
    if funding_rounds:
        import json as _json
        parts.append("\n## Funding rounds discovered (named leads + participants)")
        # Trim to the most useful fields so the prompt stays cheap.
        slim = []
        for r in funding_rounds[:10]:
            if isinstance(r, dict):
                slim.append({
                    "round_type": r.get("round_type"),
                    "date": r.get("date"),
                    "amount_usd": r.get("amount_usd"),
                    "post_money_valuation_usd": r.get("post_money_valuation_usd"),
                    "lead_investors": r.get("lead_investors"),
                    "participants": r.get("participants"),
                })
        parts.append("```json\n" + _json.dumps(slim, indent=2, default=str) + "\n```")

    parts.append(f"\n---\n\n## Market\n\n{merged['market']}")
    parts.append(f"\n---\n\n## Founders\n\n{merged['founders']}")
    parts.append(f"\n---\n\n## Traction\n\n{merged['traction']}")
    parts.append(f"\n---\n\n## Co-investors\n\n{merged['coinvestors']}")

    body = "\n".join(parts)
    system = f"{base_system}\n\n---\n\n{_SYNTH_PROMPT}"
    return await render_section(system=system, user=body, max_tokens=2800)
