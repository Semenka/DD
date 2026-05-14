"""notice.co secondary-market price adapter.

Best-effort: attempt to find the company's notice.co page, fetch it, and parse
any visible last-trade / bid / ask / implied-valuation numbers. If notice.co
blocks the request, requires auth, or doesn't list the company, return a
NoticeCoSnapshot with `available=False` and a clear `note` so the report
renders an honest empty state.

We never fabricate. If we can't see a number, we say so.
"""

from __future__ import annotations

import logging
import re

import httpx

from ..context import NoticeCoSnapshot
from .search import web_search

log = logging.getLogger("dd_agent.notice_co")

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"


async def fetch_snapshot(company_name: str) -> NoticeCoSnapshot:
    """Return a NoticeCoSnapshot. `available=True` iff we successfully parsed
    at least one numeric quote from notice.co."""
    notice_url = await _find_notice_url(company_name)
    if not notice_url:
        return NoticeCoSnapshot(
            available=False,
            note=f"No notice.co listing found for {company_name} via web search.",
        )

    html, status = await _fetch_html(notice_url)
    if not html or status >= 400:
        return NoticeCoSnapshot(
            available=False,
            source_url=notice_url,
            note=(
                f"notice.co returned HTTP {status} (likely anti-bot or auth-gated). "
                "Open the page manually or paste a session cookie into NOTICE_CO_COOKIE."
            ),
        )

    parsed = _parse_quotes(html)
    if not parsed:
        return NoticeCoSnapshot(
            available=False,
            source_url=notice_url,
            note=(
                "notice.co page fetched but no current quote was visible (likely the "
                "company has no recent trades, or the price is behind a login wall)."
            ),
        )

    return NoticeCoSnapshot(
        available=True,
        last_price_per_share=parsed.get("last_price"),
        implied_valuation_usd=parsed.get("implied_valuation"),
        bid=parsed.get("bid"),
        ask=parsed.get("ask"),
        bid_ask_mid=_mid(parsed.get("bid"), parsed.get("ask")),
        last_trade_date=parsed.get("last_trade_date"),
        source_url=notice_url,
    )


async def _find_notice_url(company_name: str) -> str | None:
    """Try direct slug guesses first, fall back to web search."""
    candidates = _slug_candidates(company_name)
    for slug in candidates:
        guessed = f"https://notice.co/{slug}"
        html, status = await _fetch_html(guessed)
        if html and status < 400 and len(html) > 1000:
            return guessed

    # Fall back to web search. With Perplexity/Gemini configured (per .env),
    # this is a real query, not the old DDG dead-end.
    for q in (f'site:notice.co "{company_name}"',
              f'notice.co "{company_name}" company secondary market'):
        results = await web_search(q, max_results=4)
        for r in results:
            if "notice.co/" in r.url:
                return r.url
    return None


def _slug_candidates(company_name: str) -> list[str]:
    """notice.co uses a variety of slug formats. Try several."""
    base = company_name.lower().strip()
    out: list[str] = []
    # 1. lowercase + dashes:  "Open AI" → "open-ai"
    out.append(re.sub(r"[^a-z0-9]+", "-", base).strip("-"))
    # 2. lowercase contiguous: "Open AI" → "openai"
    out.append(re.sub(r"[^a-z0-9]+", "", base))
    # 3. first-word only:  "Linear App" → "linear"
    first = re.split(r"\s+", base, maxsplit=1)[0]
    out.append(re.sub(r"[^a-z0-9]+", "", first))
    # de-dupe preserving order
    seen: set[str] = set()
    return [s for s in out if s and not (s in seen or seen.add(s))]


async def _fetch_html(url: str) -> tuple[str | None, int]:
    import os
    cookie = os.environ.get("NOTICE_CO_COOKIE")
    headers = {"User-Agent": UA}
    if cookie:
        headers["Cookie"] = cookie
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            r = await client.get(url, headers=headers)
            return r.text, r.status_code
    except Exception as exc:
        log.debug("notice.co fetch error %s: %s", url, exc)
        return None, 0


# --- parsing -----------------------------------------------------------------

_MONEY_RE = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?")
_DATE_RE = re.compile(r"(\b[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b)")


def _scale(suffix: str | None) -> float:
    return {"k": 1e3, "m": 1e6, "b": 1e9, "K": 1e3, "M": 1e6, "B": 1e9}.get(suffix or "", 1.0)


def _mid(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2


def _parse_quotes(html: str) -> dict:
    """Try multiple label patterns common on secondary-market sites. We strip
    HTML to plain text first to make labels match across markup variations."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)

    out: dict = {}

    last_price = _label_money(text, ["last price", "last trade", "last", "latest"])
    if last_price is not None:
        out["last_price"] = last_price

    bid = _label_money(text, ["bid", "best bid", "highest bid"])
    if bid is not None:
        out["bid"] = bid

    ask = _label_money(text, ["ask", "best ask", "lowest ask", "offer"])
    if ask is not None:
        out["ask"] = ask

    valuation = _label_money(text, ["implied valuation", "company valuation", "valuation"])
    if valuation is not None:
        out["implied_valuation"] = valuation

    m = _DATE_RE.search(text)
    if m:
        out["last_trade_date"] = m.group(1)

    return out


def _label_money(text: str, labels: list[str]) -> float | None:
    """Find the first money figure within ~50 chars after any of the given labels."""
    for label in labels:
        pat = re.compile(
            rf"\b{re.escape(label)}\b.{{0,60}}?\$\s*([0-9]+(?:[\.,][0-9]+)?)\s*([KkMmBb])?",
            re.IGNORECASE,
        )
        m = pat.search(text)
        if m:
            try:
                num = float(m.group(1).replace(",", ""))
            except ValueError:
                continue
            return num * _scale(m.group(2))
    return None
