"""Comparable-exits adapter.

For Series B+ deals, ground a list of "named M&A or IPO comps in {sector}
with exit multiples" via `ask_grounded`. Returns a structured list the
Bessemer memo's "Comparable exits" section can render verbatim.

Schema per comp: {company, exit_type, year, value_usd, multiple, multiple_basis, acquirer_or_ipo, source_url}

Used by the orchestrator for series_b / series_c_plus / growth deals only.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from .search import ask_grounded, SearchResult

log = logging.getLogger("dd_agent.exits")


@dataclass
class ComparableExit:
    company: str
    exit_type: str                          # "M&A" | "IPO"
    year: int | None = None
    value_usd: float | None = None          # total exit value
    multiple: float | None = None           # revenue or ARR multiple at exit
    multiple_basis: str | None = None       # "revenue" | "arr" | "ebitda"
    acquirer_or_ipo: str | None = None      # named acquirer or "IPO"
    source_url: str | None = None
    notes: str | None = None


@dataclass
class ExitsResult:
    comps: list[ComparableExit] = field(default_factory=list)
    sources: list[SearchResult] = field(default_factory=list)


_PROMPT_TEMPLATE = """List 3-5 notable {sector_clause} M&A or IPO exits in the past 5-10 years that \
would be useful comparable exits for evaluating {company}. For each one, output a JSON object \
with these exact fields. Use null for unknown values. NEVER invent numbers.

Return ONLY a single JSON object, no preamble, no markdown fences:

{{
  "exits": [
    {{
      "company": "Looker",
      "exit_type": "M&A",
      "year": 2019,
      "value_usd": 2600000000,
      "multiple": 22.0,
      "multiple_basis": "revenue",
      "acquirer_or_ipo": "Google",
      "notes": "all-cash, expanded GCP data stack"
    }}
  ]
}}

Rules:
- value_usd in raw USD (e.g. 2600000000 for $2.6B, not "$2.6B").
- multiple is a number (e.g. 22.0 for "22x revenue").
- multiple_basis is one of: "revenue" | "arr" | "ebitda" | null.
- exit_type is "M&A" or "IPO".
- Pick exits that are GENUINELY analogous (same sector, similar customer profile, similar maturity at exit).
- If fewer than 3 genuine comps exist, return fewer rather than reach.
- Output ONLY the JSON. No preamble."""


async def discover_exits(
    company: str, sector: str | None = None,
) -> ExitsResult:
    """Ask Perplexity/Gemini for named comparable exits in the same sector.
    Returns ExitsResult with parsed comps + the citation sources used."""
    sector_clause = f"{sector} sector" if sector else "tech / SaaS"
    prompt = _PROMPT_TEMPLATE.format(sector_clause=sector_clause, company=company)
    try:
        ans = await ask_grounded(prompt, max_sources=10, max_tokens=2000)
    except Exception as exc:
        log.warning("ask_grounded failed for exits: %s", exc)
        return ExitsResult()
    if ans is None:
        return ExitsResult()

    text = ans.text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            log.warning("could not parse exits JSON from response: %s", text[:200])
            return ExitsResult(sources=ans.sources)
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return ExitsResult(sources=ans.sources)

    rows = data.get("exits", []) if isinstance(data, dict) else []
    comps: list[ComparableExit] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("company"):
            continue
        try:
            comps.append(ComparableExit(
                company=str(row["company"]),
                exit_type=str(row.get("exit_type", "M&A")),
                year=_to_int(row.get("year")),
                value_usd=_to_float(row.get("value_usd")),
                multiple=_to_float(row.get("multiple")),
                multiple_basis=row.get("multiple_basis"),
                acquirer_or_ipo=row.get("acquirer_or_ipo"),
                notes=row.get("notes"),
            ))
        except Exception:
            continue
    return ExitsResult(comps=comps, sources=ans.sources)


def _to_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def to_jsonable(result: ExitsResult) -> dict:
    from dataclasses import asdict
    return {
        "comps": [asdict(c) for c in result.comps],
        "sources": [{"url": s.url, "title": s.title} for s in result.sources],
    }
