"""Normalize raw ingested text into a typed DealContext.

Uses OpenAI GPT-5.5 via the `codex` CLI to extract structured fields from
free-form deal memo, deck text, and scraped website. Falls back to regex
heuristics if the LLM call fails or codex is unavailable.
"""

from __future__ import annotations

import json
import os
import re
import uuid

from ..context import DealContext, Founder, Investor, Metrics

_EXTRACT_PROMPT = """You are an extractor. Given raw text from a deal memo, pitch deck, and \
company website, extract a JSON object with the schema below. Use null for unknown values. \
Do not invent values. Output ONLY the JSON object, no preamble.

SCHEMA:
{
  "company_name": str,
  "one_liner": str | null,
  "sector": str | null,
  "stage": str | null,           // seed | series_a | series_b | etc
  "founded_year": int | null,
  "hq_location": str | null,
  "website": str | null,
  "founders": [
    {"name": str, "role": str | null, "linkedin_url": str | null,
     "twitter_handle": str | null, "github_handle": str | null,
     "photo_url": str | null, "bio": str | null, "prior_companies": [str]}
  ],
  "metrics": {
    "arr_usd": float | null, "mrr_usd": float | null,
    "growth_rate_yoy": float | null,  // 2.5 = 250%
    "burn_usd_monthly": float | null, "runway_months": float | null,
    "gross_margin": float | null, "customer_count": int | null,
    "nps": float | null, "churn_monthly": float | null
  },
  "existing_investors": [
    {"name": str, "type": str | null, "round": str | null, "is_lead": bool}
  ],
  "ask_amount_usd": float | null,
  "ask_valuation_usd": float | null,
  "pre_money_usd": float | null,
  "round_type": str | null
}

INPUTS:
=== MEMO ===
<<MEMO>>

=== DECK ===
<<DECK>>

=== WEBSITE ===
<<WEBSITE>>
"""


async def normalize(
    *,
    memo_text: str | None,
    deck_text: str | None,
    website_text: str | None,
    deal_id: str | None = None,
) -> DealContext:
    """Normalize raw inputs into a DealContext."""
    raw_memo = memo_text or ""
    raw_deck = deck_text or ""
    raw_site = website_text or ""

    data = await _extract_with_llm(raw_memo, raw_deck, raw_site)
    if data is None:
        data = _extract_heuristic(raw_memo, raw_deck, raw_site)

    ctx = DealContext(
        deal_id=deal_id or uuid.uuid4().hex[:12],
        company_name=data.get("company_name") or "Unknown",
        one_liner=data.get("one_liner"),
        sector=data.get("sector"),
        stage=data.get("stage"),
        founded_year=data.get("founded_year"),
        hq_location=data.get("hq_location"),
        website=data.get("website"),
        founders=[Founder(**f) for f in (data.get("founders") or []) if f.get("name")],
        metrics=Metrics(**(data.get("metrics") or {})),
        existing_investors=[
            Investor(**i) for i in (data.get("existing_investors") or []) if i.get("name")
        ],
        ask_amount_usd=data.get("ask_amount_usd"),
        ask_valuation_usd=data.get("ask_valuation_usd"),
        pre_money_usd=data.get("pre_money_usd"),
        round_type=data.get("round_type"),
        raw_memo=raw_memo or None,
        raw_deck_text=raw_deck or None,
        raw_website_text=raw_site or None,
    )
    return ctx


async def _extract_with_llm(memo: str, deck: str, site: str) -> dict | None:
    """Call GPT-5.5 via codex CLI to extract structured fields. Returns None on failure."""
    from ..modules._llm import codex_exec, CodexUnavailableError, FAST_MODEL
    prompt = (
        _EXTRACT_PROMPT
        .replace("<<MEMO>>", _trim(memo, 30_000))
        .replace("<<DECK>>", _trim(deck, 30_000))
        .replace("<<WEBSITE>>", _trim(site, 20_000))
    )
    try:
        text = await codex_exec(prompt, model=FAST_MODEL, timeout=180.0)
    except CodexUnavailableError:
        return None
    except Exception:
        return None
    return _parse_json_block(text)


def _trim(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n // 2] + "\n…[truncated]…\n" + s[-n // 2 :]


def _parse_json_block(text: str) -> dict | None:
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


# --- heuristic fallback --------------------------------------------------------

_CAPITALIZED_BIGRAM = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){1,2})\b")
_ARR_RE = re.compile(
    r"(?:"
    r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?\s*(?:ARR|MRR)"  # $5M ARR
    r"|"
    r"(?:ARR|MRR)\s*[:\-]?\s*\$\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?"  # ARR $500K
    r")",
    re.IGNORECASE,
)
_RAISE_RE = re.compile(r"raising\s+\$?\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?", re.IGNORECASE)


def _extract_heuristic(memo: str, deck: str, site: str) -> dict:
    blob = "\n".join([memo, deck, site])
    company = _guess_company(memo, deck, site) or "Unknown"

    arr = None
    m = _ARR_RE.search(blob)
    if m:
        num = m.group(1) or m.group(3)
        suffix = m.group(2) or m.group(4)
        if num is not None:
            arr = float(num) * _scale(suffix)

    ask = None
    m = _RAISE_RE.search(blob)
    if m:
        ask = float(m.group(1)) * _scale(m.group(2))

    return {
        "company_name": company,
        "metrics": {"arr_usd": arr},
        "ask_amount_usd": ask,
    }


def _guess_company(memo: str, deck: str, site: str) -> str | None:
    for src in (deck, memo, site):
        lines = [l.strip() for l in src.splitlines() if l.strip()][:8]
        for line in lines:
            m = _CAPITALIZED_BIGRAM.match(line)
            if m and len(line) < 80:
                return m.group(1)
    return None


def _scale(suffix: str | None) -> float:
    if not suffix:
        return 1.0
    s = suffix.lower()
    return {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(s, 1.0)
