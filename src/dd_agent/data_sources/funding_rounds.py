"""Funding-history adapter.

Discovers a company's private funding rounds via free search, fetches the
most informative pages (Crunchbase /funding-rounds, PitchBook profile, TC /
SEC press releases), and asks GPT-5.5 (via codex) to normalize them into
structured FundingRound records. Returns the rounds plus the source URLs
used for citation.

Used by `modules/coinvestors.py` to render the detailed round-by-round
history table the user requested.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict

from ..context import FundingRound
from .search import web_search, fetch_page_text, SearchResult

log = logging.getLogger("dd_agent.funding_rounds")


_PROMPT = """You are extracting a private company's complete funding-round history \
from the supplied web search snippets and page excerpts. Output ONLY a JSON object with \
shape: {"rounds": [...]} where each round has:

{
  "round_type": "pre_seed" | "seed" | "series_a" | "series_b" | "series_c" | ... | "secondary" | null,
  "date": "YYYY-MM-DD" or "YYYY-MM" or null,
  "amount_usd": float | null,           // raised in this round
  "post_money_valuation_usd": float | null,
  "pre_money_valuation_usd": float | null,
  "lead_investors": [str],              // names; empty list if unknown
  "participants": [str],                // non-lead investors; empty list if unknown
  "source_url": str | null,             // the most authoritative URL backing this row
  "source_title": str | null,
  "notes": str | null                   // brief context like "led by partner X" if relevant
}

CRITICAL: Only emit rounds you can substantiate from the supplied snippets/excerpts. \
Order from earliest to latest. If a field is not stated, set it to null — do not invent. \
Output ONLY the JSON object, no preamble, no markdown fences. Start with `{`.

---
"""


async def discover_rounds(
    company_name: str,
    *,
    max_search_results: int = 12,
    max_pages_to_fetch: int = 4,
) -> tuple[list[FundingRound], list[SearchResult]]:
    """Return (rounds, source results). Either may be empty.

    Steps:
      1. Search the web for the company's funding history (Crunchbase, PitchBook,
         press releases).
      2. Fetch the top few result pages for richer text.
      3. Ask GPT-5.5 to normalize into FundingRound[].
    """
    queries = [
        f'"{company_name}" funding rounds crunchbase',
        f'"{company_name}" series investors valuation',
        f'"{company_name}" raised funding round announcement',
    ]
    results: list[SearchResult] = []
    seen: set[str] = set()
    for q in queries:
        for r in await web_search(q, max_results=6):
            if r.url in seen:
                continue
            seen.add(r.url)
            results.append(r)
            if len(results) >= max_search_results:
                break
        if len(results) >= max_search_results:
            break

    if not results:
        return [], []

    # Fetch deeper text for the most authoritative-looking hits.
    fetch_targets = _rank_for_fetch(results)[:max_pages_to_fetch]
    page_texts: dict[str, str] = {}
    for r in fetch_targets:
        text = await fetch_page_text(r.url, max_chars=6000)
        if text:
            page_texts[r.url] = text

    rounds = await _extract_via_llm(company_name, results, page_texts)
    return rounds, results


def _rank_for_fetch(results: list[SearchResult]) -> list[SearchResult]:
    """Prefer Crunchbase, PitchBook, TechCrunch, SEC, official press releases."""
    priority_hosts = (
        "crunchbase.com", "pitchbook.com", "techcrunch.com", "sec.gov",
        "businesswire.com", "prnewswire.com", "axios.com", "bloomberg.com",
        "wsj.com", "ft.com", "theinformation.com", "tracxn.com",
    )
    def score(r: SearchResult) -> int:
        return next((10 - i for i, h in enumerate(priority_hosts) if h in r.url), 0)
    return sorted(results, key=score, reverse=True)


async def _extract_via_llm(
    company_name: str,
    results: list[SearchResult],
    page_texts: dict[str, str],
) -> list[FundingRound]:
    from ..modules._llm import codex_exec, CodexUnavailableError, FAST_MODEL

    user = [_PROMPT, f"Company: {company_name}\n", "Web search results:"]
    for i, r in enumerate(results, 1):
        user.append(f"[{i}] {r.title} — {r.url}\n    {r.snippet}")
    if page_texts:
        user.append("\nPage excerpts:")
        for url, txt in page_texts.items():
            user.append(f"\n=== {url} ===\n{txt[:5500]}")

    try:
        text = await codex_exec("\n".join(user), model=FAST_MODEL, timeout=180.0)
    except CodexUnavailableError:
        log.warning("codex unavailable; skipping funding-round extraction")
        return []
    except Exception as exc:
        log.warning("funding-round LLM call failed: %s", exc)
        return []

    data = _parse_json(text)
    if not data:
        return []
    rows = data.get("rounds", []) if isinstance(data, dict) else []
    out: list[FundingRound] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            out.append(FundingRound(
                round_type=row.get("round_type"),
                date=row.get("date"),
                amount_usd=_to_float(row.get("amount_usd")),
                post_money_valuation_usd=_to_float(row.get("post_money_valuation_usd")),
                pre_money_valuation_usd=_to_float(row.get("pre_money_valuation_usd")),
                lead_investors=list(row.get("lead_investors") or []),
                participants=list(row.get("participants") or []),
                source_url=row.get("source_url"),
                source_title=row.get("source_title"),
                notes=row.get("notes"),
            ))
        except Exception:
            continue
    return out


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


def _to_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_jsonable(rounds: list[FundingRound]) -> list[dict]:
    return [asdict(r) for r in rounds]
