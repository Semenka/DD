"""Funding-history adapter.

Asks Perplexity / Gemini for a structured funding-round history in JSON,
returning the rounds plus the cited source URLs. This is far more reliable
than scraping Crunchbase + asking codex to normalize: the grounded LLM does
the search and the extraction in one round-trip.

Falls back to a legacy "search + codex extract" path if grounded search isn't
configured (no PERPLEXITY_API_KEY / GEMINI_API_KEY).

Used by `modules/coinvestors.py` to render the detailed round-by-round
history table.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict

from ..context import FundingRound
from .search import ask_grounded, fetch_page_text, SearchResult, web_search

log = logging.getLogger("dd_agent.funding_rounds")


_GROUNDED_PROMPT_TEMPLATE = """List every publicly-disclosed private funding \
round for the company "{name}". Order from earliest to latest. For each round \
include: round type (pre-seed / seed / series A / series B / etc.), date \
(YYYY-MM or YYYY), amount raised in USD, post-money valuation if disclosed, \
lead investor(s), and other named participants.

Output ONLY a single JSON object with this exact shape, nothing else:

{{
  "rounds": [
    {{
      "round_type": "seed" | "series_a" | "series_b" | ... | null,
      "date": "YYYY-MM-DD" | "YYYY-MM" | "YYYY" | null,
      "amount_usd": number_in_usd | null,
      "post_money_valuation_usd": number_in_usd | null,
      "pre_money_valuation_usd": number_in_usd | null,
      "lead_investors": ["Name", ...],
      "participants": ["Name", ...],
      "notes": "free-text context" | null
    }}
  ]
}}

CRITICAL RULES:
- All dollar values must be numbers in raw USD (e.g. 5000000 for $5M, not "5M").
- Set fields to null when not publicly reported. NEVER invent figures.
- If the company has no publicly-reported rounds, output: {{"rounds": []}}.
- Output ONLY the JSON object. No preamble, no markdown fences, no commentary.
"""


async def discover_rounds(
    company_name: str,
    *,
    max_search_results: int = 12,
    max_pages_to_fetch: int = 4,
) -> tuple[list[FundingRound], list[SearchResult]]:
    """Return (rounds, source results). Either may be empty."""
    # Preferred path: ask Perplexity/Gemini for the structured answer directly.
    if os.environ.get("PERPLEXITY_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        rounds, sources = await _discover_grounded(company_name)
        if rounds or sources:
            return rounds, sources
    # Legacy fallback: search → fetch pages → codex extract.
    return await _discover_legacy(company_name, max_search_results, max_pages_to_fetch)


async def _discover_grounded(
    company_name: str,
) -> tuple[list[FundingRound], list[SearchResult]]:
    """Use ask_grounded() so the LLM searches AND extracts in one call."""
    prompt = _GROUNDED_PROMPT_TEMPLATE.format(name=company_name)
    ans = await ask_grounded(prompt, max_sources=15)
    if ans is None:
        return [], []
    rounds = _parse_rounds(ans.text)
    # Attach the cited source URL to each round when we can match.
    _attach_sources(rounds, ans.sources)
    return rounds, ans.sources


def _attach_sources(rounds: list[FundingRound], sources: list[SearchResult]) -> None:
    """Best-effort: if a round lacks source_url, point it at the most relevant
    citation. We pick by simple title-substring heuristics."""
    if not sources:
        return
    fallback = sources[0]
    for r in rounds:
        if r.source_url:
            continue
        chosen = fallback
        if r.round_type:
            for s in sources:
                if r.round_type.replace("_", " ").lower() in s.title.lower():
                    chosen = s
                    break
        r.source_url = chosen.url
        r.source_title = chosen.title


def _parse_rounds(text: str) -> list[FundingRound]:
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


# --- legacy fallback (search + codex extract) -------------------------------


async def _discover_legacy(
    company_name: str,
    max_search_results: int,
    max_pages_to_fetch: int,
) -> tuple[list[FundingRound], list[SearchResult]]:
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
    fetch_targets = _rank_for_fetch(results)[:max_pages_to_fetch]
    page_texts: dict[str, str] = {}
    for r in fetch_targets:
        text = await fetch_page_text(r.url, max_chars=6000)
        if text:
            page_texts[r.url] = text
    rounds = await _extract_via_codex(company_name, results, page_texts)
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


async def _extract_via_codex(
    company_name: str,
    results: list[SearchResult],
    page_texts: dict[str, str],
) -> list[FundingRound]:
    from ..modules._llm import codex_exec, CodexUnavailableError, FAST_MODEL

    legacy_prompt = (
        "You are extracting a private company's funding-round history from the "
        "supplied web snippets and page excerpts. Output ONLY a JSON object "
        '{"rounds":[{round_type, date, amount_usd, post_money_valuation_usd, '
        "pre_money_valuation_usd, lead_investors, participants, source_url, "
        "source_title, notes}]}. Use null for unknown fields. Never invent.\n\n"
    )
    user = [legacy_prompt, f"Company: {company_name}\n", "Web search results:"]
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
