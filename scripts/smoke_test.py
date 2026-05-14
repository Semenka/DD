"""End-to-end smoke test for the DD agent.

Runs the full pipeline on a sample deal and writes report.md + report.html to
the current directory. Useful for verifying the install and watching the
agent work without the MCP layer in between.

Usage:
  python scripts/smoke_test.py                          # uses examples/sample_deal/
  python scripts/smoke_test.py --memo path/to/memo.md \
      --deck path/to/deck.pdf --url https://example.com
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=os.environ.get("DD_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("smoke_test")


async def _run(args) -> int:
    from dd_agent.orchestrator import submit, _run_pipeline  # noqa: F401
    from dd_agent.state import DealStore

    import shutil
    if not shutil.which(os.environ.get("DD_CODEX_BIN", "codex")):
        log.error(
            "codex CLI not on PATH. Install it: `npm install -g @openai/codex && codex login`"
        )
        return 2

    memo_text = None
    if args.memo:
        memo_text = Path(args.memo).read_text(encoding="utf-8")

    store = DealStore(db_path=ROOT / "data" / "smoke_test.db")
    await store.init()
    submitted = await submit(
        store=store,
        memo_text=memo_text,
        deck_path=args.deck,
        company_url=args.url,
    )
    log.info("submitted deal_id=%s — pipeline running", submitted.deal_id)

    while True:
        await asyncio.sleep(2.0)
        record = await store.get(submitted.deal_id)
        if record is None:
            log.error("deal record disappeared")
            return 3
        log.info("status=%s phase=%s progress=%d%%",
                 record.status.value, record.phase, record.progress_pct)
        if record.status.value in ("done", "failed"):
            break

    if record.status.value == "failed":
        log.error("pipeline failed: %s", record.error)
        return 4

    out_md = Path("report.md")
    out_html = Path("report.html")
    out_md.write_text(record.report_markdown or "", encoding="utf-8")
    out_html.write_text(record.report_html or "", encoding="utf-8")
    log.info("wrote %s (%d bytes), %s (%d bytes)",
             out_md, out_md.stat().st_size, out_html, out_html.stat().st_size)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sample = ROOT / "examples" / "sample_deal"
    ap.add_argument("--memo", default=str(sample / "memo.md"))
    ap.add_argument("--deck", default=None)
    ap.add_argument("--url", default="https://linear.app")
    args = ap.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
