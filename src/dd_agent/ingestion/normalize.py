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
Do not invent values. Output ONLY the JSON object — no preamble, no markdown fences, no \
``` blocks. Start your response with the literal character `{` and end with `}`.

CRITICAL: `company_name` is required if it appears anywhere in the inputs. Look for phrases \
like "Company: X", "# X — ...", "X is the ...", "We are X", "Investment Memo — X". If a \
single company name appears in the memo header or first paragraph, use it.

For monetary values, return USD as a number (no string suffixes). $5M → 5000000. $200K → 200000.
For growth rates, return YoY as a decimal where 2.5 = 250%.

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

    heuristic = _extract_heuristic(raw_memo, raw_deck, raw_site)
    llm = await _extract_with_llm(raw_memo, raw_deck, raw_site) or {}
    data = _merge(heuristic, llm)

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

_CAPITALIZED_BIGRAM = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){0,2})\b")
_ARR_RE = re.compile(
    r"(?:"
    r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?\s*(?:ARR|MRR)"  # $5M ARR
    r"|"
    r"(?:ARR|MRR)\s*[:\-]?\s*\$\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?"  # ARR $500K
    r")",
    re.IGNORECASE,
)
_RAISE_RE = re.compile(r"raising\s*[:\-]?\s*\$\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?", re.IGNORECASE)
# Common memo headers
_COMPANY_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?(?:Company|Startup|Target)\s*:?\s*(?:\*\*)?\s*([A-Z][\w\.\- ]{1,60})",
    re.IGNORECASE,
)
_MEMO_TITLE_RE = re.compile(
    r"#\s*(?:Investment\s+Memo|Deal\s+Memo|Memo)\s*[—\-:]\s*([A-Z][\w\.\- ]{1,60})",
    re.IGNORECASE,
)
_STAGE_RE = re.compile(
    # Allow markdown emphasis or stray punctuation between "Stage" and the value.
    r"\b(?:stage|round)\b[^A-Za-z]{0,6}(seed|pre-seed|series[\s\-_]?[a-h])\b",
    re.IGNORECASE,
)
_VALUATION_RE = re.compile(
    r"(?:"
    # "valuation: $400M", "pre-money $400M"
    r"(?:valuation|pre[\s\-]?money|post[\s\-]?money)[^A-Za-z0-9\$]{0,8}\$\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?"
    r"|"
    # "$400M valuation", "$400M pre-money"
    r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?\s+(?:valuation|pre[\s\-]?money|post[\s\-]?money)"
    r")",
    re.IGNORECASE,
)


def _extract_heuristic(memo: str, deck: str, site: str) -> dict:
    blob = "\n".join([memo, deck, site])
    company = _guess_company(memo, deck, site)

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

    stage = None
    m = _STAGE_RE.search(blob)
    if m:
        stage = m.group(1).lower().replace(" ", "_").replace("-", "_")

    valuation = None
    m = _VALUATION_RE.search(blob)
    if m:
        num = m.group(1) or m.group(3)
        suffix = m.group(2) or m.group(4)
        if num is not None:
            valuation = float(num) * _scale(suffix)

    result: dict = {}
    if company:
        result["company_name"] = company
    if stage:
        result["stage"] = stage
    if arr is not None:
        result["metrics"] = {"arr_usd": arr}
    if ask is not None:
        result["ask_amount_usd"] = ask
    if valuation is not None:
        result["ask_valuation_usd"] = valuation
    return result


def _guess_company(memo: str, deck: str, site: str) -> str | None:
    """Pick the most likely company name. Priority:
    1. Explicit "Company: X" line (covers both memo and deck)
    2. "# Investment Memo — X" header
    3. First capitalized phrase in the first 8 lines (last resort)
    """
    blob = "\n".join([memo, deck, site])
    m = _COMPANY_LINE_RE.search(blob)
    if m:
        name = m.group(1).strip().rstrip(".,;:—-")
        if name and name.lower() not in {"unknown", "tbd", "n/a"}:
            return name
    m = _MEMO_TITLE_RE.search(blob)
    if m:
        name = m.group(1).strip().rstrip(".,;:—-")
        if name:
            return name
    for src in (deck, memo, site):
        lines = [l.strip() for l in src.splitlines() if l.strip()][:8]
        for line in lines:
            if line.startswith("#") or line.startswith("**"):
                continue
            m = _CAPITALIZED_BIGRAM.match(line)
            if m and len(line) < 80:
                return m.group(1)
    return None


def _merge(base: dict, overlay: dict) -> dict:
    """Merge `overlay` on top of `base`. Non-null overlay values win; nested
    dicts merge recursively. Lists in overlay win wholesale if non-empty."""
    out: dict = {}
    keys = set(base) | set(overlay)
    for k in keys:
        b, o = base.get(k), overlay.get(k)
        if isinstance(b, dict) or isinstance(o, dict):
            out[k] = _merge(b or {}, o or {})
        elif isinstance(o, list):
            out[k] = o if o else (b or [])
        elif o is not None and o != "":
            out[k] = o
        elif b is not None:
            out[k] = b
    return out


def _scale(suffix: str | None) -> float:
    if not suffix:
        return 1.0
    s = suffix.lower()
    return {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(s, 1.0)
