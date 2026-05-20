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

REVENUE EXTRACTION — Be exhaustive. Look for ANY of these phrases and slot them correctly:
  - "ARR", "annual recurring revenue" → arr_usd
  - "MRR", "monthly recurring revenue" → mrr_usd
  - "GMV", "gross merchandise value", "GTV", "transaction volume" → gmv_usd
  - "gross revenue", "top-line revenue", "billings" → gross_revenue_usd
  - "net revenue", "net sales" → net_revenue_usd
  - "TPV", "payment volume" → transaction_volume_usd
  - "take rate", "rake" (percentage) → take_rate (as decimal: 12% → 0.12)
  - "NRR", "net retention", "net revenue retention" → net_retention (as decimal)
  - Annualized run-rates derived from monthly numbers should populate the corresponding
    annualized field, with a note in arr_quality_notes.

ARR QUALITY — if the memo states ARR or implies it, set `arr_quality` to ONE of:
  - "recurring_subscription"  → genuine SaaS subscription contracts
  - "annualized_contracts"    → multi-year customer commits, billed annually
  - "annualized_pilots"       → paid pilots, not yet renewed
  - "annualized_transactions" → one-time sales annualized (NOT real ARR)
  - "gmv_or_take_rate"        → marketplace volume, not company revenue (NOT real ARR)
  - "one_time_hardware"       → hardware sales, not recurring (NOT real ARR)
  - "unclear"                 → can't tell from the inputs
And write a 1-2 sentence `arr_quality_notes` justification. NEVER invent — only call it
recurring_subscription if the memo explicitly says "subscription" or "recurring".

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
    "arr_quality": str | null,           // see ARR QUALITY taxonomy above
    "arr_quality_notes": str | null,
    "gmv_usd": float | null, "gross_revenue_usd": float | null,
    "net_revenue_usd": float | null, "transaction_volume_usd": float | null,
    "take_rate": float | null,           // 12% → 0.12
    "growth_rate_yoy": float | null,     // 2.5 = 250%
    "burn_usd_monthly": float | null, "runway_months": float | null,
    "gross_margin": float | null, "customer_count": int | null,
    "nps": float | null, "churn_monthly": float | null,
    "net_retention": float | null        // 110% → 1.10
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

    # Post-merge sanity check: if the company_name turned out to be a section
    # header (e.g. "TERMS", "Our Story") because the LLM grabbed a PDF heading,
    # clear it so downstream code can say "Unknown" rather than misroute.
    cn = data.get("company_name")
    if cn and _is_section_header(cn):
        data["company_name"] = None

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


# Context windows around an ARR figure. Used by the heuristic to classify
# revenue quality even when the LLM doesn't fill arr_quality.
_QUALITY_KEYWORDS = {
    "annualized_pilots":      ("pilot", "trial", "POC", "proof of concept", "letter of intent"),
    "one_time_hardware":      ("hardware sale", "hardware units", "device sales", "unit sales"),
    "gmv_or_take_rate":       ("GMV", "marketplace", "transaction volume", "take rate", "take-rate", "TPV"),
    "annualized_contracts":   ("signed contract", "multi-year contract", "annual contract", "ACV"),
    "annualized_transactions": ("one-time", "one time", "non-recurring", "project work", "services revenue"),
    "recurring_subscription": ("recurring", "subscription", "SaaS", "monthly subscription", "annual subscription"),
}


def _classify_arr_quality(blob: str, arr_match_span: tuple[int, int]) -> tuple[str | None, str | None]:
    """Look at the ±200-char window around the ARR figure and classify.

    Returns (quality_label, notes). Both None if no signal."""
    start = max(0, arr_match_span[0] - 200)
    end = min(len(blob), arr_match_span[1] + 200)
    window = blob[start:end].lower()
    for label, keywords in _QUALITY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in window:
                # Find a short excerpt for the note
                idx = window.find(kw.lower())
                snippet_start = max(0, idx - 40)
                snippet_end = min(len(window), idx + 80)
                snippet = window[snippet_start:snippet_end].strip().replace("\n", " ")
                return label, f'heuristic classification — saw "...{snippet}..." near the ARR figure'
    return None, None




_CAPITALIZED_BIGRAM = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){0,2})\b")
# Approximation markers (~, ≈, ca., approx., about) allowed between the label
# and the value. Also tolerate markdown emphasis (* / **) around the label.
_APPROX = r"(?:\s|~|≈|approx\.?|about|ca\.?|roughly|\*)*"
_ARR_RE = re.compile(
    r"(?:"
    r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?\s*ARR\b"                            # $5M ARR / ~$5M ARR
    r"|"
    r"\bARR\b" + _APPROX + r"[:\-]?" + _APPROX + r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?"  # ARR: $500K / ARR: ~$8M
    r")",
    re.IGNORECASE,
)
_MRR_RE = re.compile(
    r"(?:"
    r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?\s*MRR\b"
    r"|"
    r"\bMRR\b" + _APPROX + r"[:\-]?" + _APPROX + r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?"
    r")",
    re.IGNORECASE,
)
# Look for any of: GMV, GTV, "gross merchandise value", "transaction volume"
_GMV_RE = re.compile(
    r"(?:"
    r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?\s*(?:GMV|GTV|gross merchandise value|transaction volume|payment volume)\b"
    r"|"
    r"\b(?:GMV|GTV|gross merchandise value|transaction volume|payment volume|TPV)\s*[:\-]?\s*\$\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?"
    r")",
    re.IGNORECASE,
)
_GROSS_REV_RE = re.compile(
    r"(?:"
    r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?\s*(?:gross revenue|top-line|billings)\b"
    r"|"
    r"\b(?:gross revenue|top[\s\-]?line revenue|billings)\s*[:\-]?\s*\$\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?"
    r")",
    re.IGNORECASE,
)
_NET_REV_RE = re.compile(
    r"\b(?:net revenue|net sales)\s*[:\-]?\s*\$\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?",
    re.IGNORECASE,
)
_TAKE_RATE_RE = re.compile(
    r"\btake[\s\-]?rate\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
    re.IGNORECASE,
)
_NRR_RE = re.compile(
    r"\b(?:NRR|net retention|net revenue retention)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
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

    def _money(pat: re.Pattern[str], text: str) -> float | None:
        m = pat.search(text)
        if not m:
            return None
        num = m.group(1) or (m.group(3) if m.lastindex and m.lastindex >= 3 else None)
        suffix = m.group(2) or (m.group(4) if m.lastindex and m.lastindex >= 4 else None)
        if num is None:
            return None
        return float(num) * _scale(suffix)

    # ARR + quality classification from surrounding context
    arr = _money(_ARR_RE, blob)
    arr_quality: str | None = None
    arr_quality_notes: str | None = None
    if arr is not None:
        m_arr = _ARR_RE.search(blob)
        if m_arr:
            arr_quality, arr_quality_notes = _classify_arr_quality(blob, m_arr.span())
    mrr = _money(_MRR_RE, blob)
    gmv = _money(_GMV_RE, blob)
    gross_rev = _money(_GROSS_REV_RE, blob)
    # _NET_REV_RE has only 2 groups so the generic _money fn miscounts; handle directly.
    net_rev = None
    m = _NET_REV_RE.search(blob)
    if m:
        net_rev = float(m.group(1)) * _scale(m.group(2))
    take_rate = None
    m = _TAKE_RATE_RE.search(blob)
    if m:
        take_rate = float(m.group(1)) / 100.0
    nrr = None
    m = _NRR_RE.search(blob)
    if m:
        nrr = float(m.group(1)) / 100.0

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
    metrics: dict = {}
    if arr is not None:
        metrics["arr_usd"] = arr
        if arr_quality:
            metrics["arr_quality"] = arr_quality
            metrics["arr_quality_notes"] = arr_quality_notes
    if mrr is not None:
        metrics["mrr_usd"] = mrr
        # If only MRR is known, also surface annualized — flagged as such in notes.
        if arr is None:
            metrics["arr_usd"] = mrr * 12
            metrics["arr_quality"] = "unclear"
            metrics["arr_quality_notes"] = "annualized from stated MRR; needs verification"
    if gmv is not None:
        metrics["gmv_usd"] = gmv
    if gross_rev is not None:
        metrics["gross_revenue_usd"] = gross_rev
    if net_rev is not None:
        metrics["net_revenue_usd"] = net_rev
    if take_rate is not None:
        metrics["take_rate"] = take_rate
    if nrr is not None:
        metrics["net_retention"] = nrr
    if metrics:
        result["metrics"] = metrics
    if ask is not None:
        result["ask_amount_usd"] = ask
    if valuation is not None:
        result["ask_valuation_usd"] = valuation
    return result


# Common deck / memo section headers that look like company names but aren't.
# Hit by Alfred AngelList memo which has "OUR STORY" as a section title that
# our regex grabbed as the company name.
_SECTION_HEADERS = frozenset(s.lower() for s in {
    "Our Story", "Our Team", "Our Mission", "Our Vision", "Our Values",
    "Our Product", "Our Customers", "Our Investors", "Our Approach",
    "Team", "Traction", "Market", "Overview", "Summary", "Problem",
    "Solution", "Product", "Customers", "Investors", "Founders",
    "Why Now", "The Ask", "Use of Funds", "Financials", "Metrics",
    "Background", "Vision", "Mission", "Story", "Investment Memo",
    "Deal Memo", "Pitch Deck", "Company Overview", "Executive Summary",
    "Confidential", "Terms of Service", "Privacy Policy", "Legal",
    # AngelList / Carta common templates:
    "Highlights", "Round Details", "Cap Table", "Recent Investors",
    "Lead Investor", "Notable Investors",
    # Single-word ALL CAPS section titles that PDFs often have at the top:
    "Terms", "TERMS", "Disclaimer", "Disclosure", "Index", "Contents",
    "Appendix", "About", "About Us", "Contact", "Notes",
})


def _is_section_header(name: str) -> bool:
    """A guessed company-name candidate is a section header if it matches one
    of the common deck/memo section titles (case-insensitive)."""
    return name.strip().lower() in _SECTION_HEADERS


def _guess_company(memo: str, deck: str, site: str) -> str | None:
    """Pick the most likely company name. Priority:
    1. Explicit "Company: X" line (covers both memo and deck)
    2. "# Investment Memo — X" header
    3. First capitalized phrase in the first 8 lines that ISN'T a section header
    """
    blob = "\n".join([memo, deck, site])
    m = _COMPANY_LINE_RE.search(blob)
    if m:
        name = m.group(1).strip().rstrip(".,;:—-")
        if name and name.lower() not in {"unknown", "tbd", "n/a"} \
                and not _is_section_header(name):
            return name
    m = _MEMO_TITLE_RE.search(blob)
    if m:
        name = m.group(1).strip().rstrip(".,;:—-")
        if name and not _is_section_header(name):
            return name
    for src in (deck, memo, site):
        lines = [l.strip() for l in src.splitlines() if l.strip()][:8]
        for line in lines:
            if line.startswith("#") or line.startswith("**"):
                continue
            m = _CAPITALIZED_BIGRAM.match(line)
            if m and len(line) < 80:
                candidate = m.group(1)
                if not _is_section_header(candidate):
                    return candidate
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
