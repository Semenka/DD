"""MCP server entry point.

Exposes four tools over stdio:
  - submit_deal(memo_text?, deck_path?, company_url?, founder_names?[])
  - get_report_status(deal_id)
  - get_report(deal_id)
  - list_deals()

`dd-agent serve` runs the stdio server. Connect from OpenClaw / Claude Code /
Cursor by adding the standard MCP config block (see README).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from .orchestrator import submit
from .state import DealStore

load_dotenv()

logging.basicConfig(
    level=os.environ.get("DD_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("dd_agent.server")

mcp = FastMCP("dd-agent")
_store = DealStore()


@mcp.tool()
async def submit_deal(
    memo_text: str | None = None,
    memo_path: str | None = None,
    deck_path: str | None = None,
    company_url: str | None = None,
    founder_names: list[str] | None = None,
    deliver_to: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Submit a deal for due diligence.

    Provide at least one input:
      - memo_text: the deal memo as plain text
      - memo_path: absolute path to a deal memo (PDF, text, or markdown). Used
        when the memo came in as an attachment (e.g. a PDF posted in Telegram).
      - deck_path: absolute path to a pitch deck PDF
      - company_url: the company website URL
      - founder_names: list of founder names (used when not extractable from inputs)

    Optional `deliver_to` makes dd-agent send the finished report itself when
    the pipeline completes (so the caller doesn't have to poll):
        {"channel": "telegram", "account": "cosmo", "target": "148594943",
         "format": "html", "summary_line": "<optional>"}
    channel is required; account defaults to "default"; format defaults to
    "html" with active citation links (also accepts "pdf" or "markdown").

    Returns immediately with a `deal_id`. The DD pipeline runs in the background.
    If deliver_to was set, dd-agent will send the report on completion via
    `openclaw message send`. Without deliver_to, poll get_report_status(deal_id)
    until phase=='done', then call get_report.
    """
    await _store.init()
    if not any([memo_text, memo_path, deck_path, company_url]):
        return {"error": "provide at least one of memo_text, memo_path, deck_path, company_url"}
    submitted = await submit(
        store=_store,
        memo_text=memo_text,
        memo_path=memo_path,
        deck_path=deck_path,
        company_url=company_url,
        founder_names=founder_names,
        deliver_to=deliver_to,
    )
    return {
        "deal_id": submitted.deal_id,
        "status": submitted.status,
        "auto_delivery": bool(deliver_to),
    }


@mcp.tool()
async def get_report_status(deal_id: str) -> dict[str, Any]:
    """Get the current status of a submitted deal.

    Returns: {status, phase, progress_pct, company, error?}
    Status is one of: queued | ingesting | running | done | failed.
    """
    await _store.init()
    record = await _store.get(deal_id)
    if record is None:
        return {"error": f"deal_id {deal_id} not found"}
    return record.to_summary()


@mcp.tool()
async def get_report(
    deal_id: str,
    include_pdf_base64: bool = False,
    include_markdown: bool = True,
    include_html: bool = False,
) -> dict[str, Any]:
    """Retrieve the final DD report for a completed deal.

    Returns: {markdown, html, pdf_path, citations, company}.
    Optionally include `pdf_base64` for callers (e.g. Telegram bots) that need
    to attach the PDF directly. Suppress `html` for size if you only need to
    forward the PDF; suppress `markdown` if the caller only needs the PDF path.

    If the deal is not yet complete, returns {status, phase, progress_pct}.
    """
    import base64
    await _store.init()
    record = await _store.get(deal_id)
    if record is None:
        return {"error": f"deal_id {deal_id} not found"}
    if record.status.value != "done":
        return {
            "status": record.status.value,
            "phase": record.phase,
            "progress_pct": record.progress_pct,
            "error": record.error,
        }
    result: dict[str, Any] = {
        "deal_id": record.deal_id,
        "company": record.company_name,
        "pdf_path": record.report_pdf_path,
        "citations": json.loads(record.citations_json) if record.citations_json else [],
    }
    if include_markdown:
        result["markdown"] = record.report_markdown
    if include_html:
        result["html"] = record.report_html
    if include_pdf_base64 and record.report_pdf_path:
        try:
            with open(record.report_pdf_path, "rb") as f:
                result["pdf_base64"] = base64.b64encode(f.read()).decode("ascii")
        except OSError as exc:
            result["pdf_base64_error"] = str(exc)
    return result


@mcp.tool()
async def list_deals(limit: int = 50) -> list[dict[str, Any]]:
    """List recent deals (most recent first)."""
    await _store.init()
    records = await _store.list_recent(limit=limit)
    return [r.to_summary() for r in records]


def main() -> None:
    """CLI entry. `dd-agent serve` starts the stdio MCP server."""
    args = sys.argv[1:]
    if args and args[0] == "serve":
        mcp.run()
    else:
        print(
            "dd-agent — Elad-Gil-style DD agent\n\n"
            "Usage:\n"
            "  dd-agent serve            # run as stdio MCP server\n\n"
            "Test without MCP:\n"
            "  python scripts/smoke_test.py --memo path/to/memo.md --deck path/to/deck.pdf",
            file=sys.stderr,
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
