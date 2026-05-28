"""v10 Quality Gate — backtest over the real historical reports.

This is the "money test": it proves the deterministic gate cleanly separates
the historical garbage reports (PK / Round Seed / Investment — all analyzed a
section header instead of a company) from the legitimate ones (Alfred, Rivan).

Runs against the live data/deals.db. Skipped automatically when the DB or the
expected deals aren't present (fresh checkouts, CI), so it never blocks the
suite — but on the dev machine it's the guard that keeps the gate honest.

Observed scores at v10 ship time (deterministic layer, use_llm=False):
    PK            2.0   <- garbage (wrong company)
    Round Seed    3.0   <- garbage (section header)
    Investment    3.0   <- garbage (section header)
    Linear        5.0   <- legit but pre-v7 (no 4-pillar exec summary) → retry band
    Alfred (old)  6.0   <- legit
    Alfred (new)  8.0   <- legit
    Rivan         8.0   <- legit

The contract: every garbage deal scores < RETRY_FLOOR (4.0); every legit deal
scores strictly above the garbage ceiling. The 4.0 gate flags garbage as LOW
CONFIDENCE while letting legit reports ship or retry.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dd_agent.citations import Citation, CitationBook
from dd_agent.context import DealContext, Founder
from dd_agent.report import quality

_DB_PATH = os.environ.get("DD_DB_PATH", "/Users/andrey/DD/data/deals.db")

# Lowercased company-name substrings we expect to be garbage vs legit.
_GARBAGE = {"pk", "round seed", "investment"}
_LEGIT = {"alfred", "rivan", "linear"}


def _db_available() -> bool:
    return Path(_DB_PATH).exists()


pytestmark = pytest.mark.skipif(
    not _db_available(),
    reason=f"historical deals.db not present at {_DB_PATH}",
)


def _ctx_from_record(d) -> DealContext:
    cj = json.loads(d.context_json or "{}")
    ctx = DealContext(
        deal_id=d.deal_id,
        company_name=cj.get("company_name") or (d.company_name or ""),
    )
    ctx.raw_memo = cj.get("raw_memo")
    ctx.raw_deck_text = cj.get("raw_deck_text")
    for f in cj.get("founders", []):
        ctx.founders.append(Founder(
            name=f.get("name", ""), bio=f.get("bio"),
            photo_url=f.get("photo_url"),
            prior_companies=f.get("prior_companies") or [],
        ))
    return ctx


async def _score_all() -> dict[str, float]:
    from dd_agent.state import DealStore
    store = DealStore(_DB_PATH)
    deals = await store.list_recent(limit=40)
    scores: dict[str, float] = {}
    for d in deals:
        if d.status.value != "done" or not d.report_markdown:
            continue
        ctx = _ctx_from_record(d)
        book = CitationBook()
        for c in json.loads(d.citations_json or "[]"):
            book.add(Citation(
                key=c.get("url", c.get("key", "")),
                title=c.get("title", ""), url=c.get("url", ""),
            ))
        rep = await quality.score_report(
            ctx=ctx, markdown=d.report_markdown,
            merged={"citations": book}, extras={}, use_llm=False,
        )
        # Keep the lowest score seen per company name (worst case).
        name = (ctx.company_name or "?").lower()
        scores[name] = min(scores.get(name, 99.0), rep.deterministic_score)
    return scores


@pytest.mark.asyncio
async def test_garbage_reports_flagged_low():
    scores = await _score_all()
    seen_garbage = {k: v for k, v in scores.items() if k in _GARBAGE}
    if not seen_garbage:
        pytest.skip("no historical garbage deals in DB to backtest")
    for name, score in seen_garbage.items():
        assert score < quality.RETRY_FLOOR, (
            f"{name!r} scored {score} — should be < {quality.RETRY_FLOOR} (garbage)"
        )


@pytest.mark.asyncio
async def test_legit_reports_clear_garbage_tier():
    scores = await _score_all()
    garbage = [v for k, v in scores.items() if k in _GARBAGE]
    legit = {k: v for k, v in scores.items() if k in _LEGIT}
    if not legit:
        pytest.skip("no legit deals in DB to backtest")
    # Every legit deal must clear the garbage ceiling.
    ceiling = max(garbage) if garbage else quality.RETRY_FLOOR
    for name, score in legit.items():
        assert score > ceiling, (
            f"{name!r} scored {score} — should exceed garbage ceiling {ceiling}"
        )


@pytest.mark.asyncio
async def test_at_least_one_legit_ships_clean():
    """At least one modern legit report should clear the 6.0 'near-ship' bar,
    proving the gate isn't so strict that nothing passes."""
    scores = await _score_all()
    legit = [v for k, v in scores.items() if k in _LEGIT]
    if not legit:
        pytest.skip("no legit deals to backtest")
    assert max(legit) >= 6.0, f"best legit report only scored {max(legit)}"
