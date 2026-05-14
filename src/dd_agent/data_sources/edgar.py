"""SEC EDGAR fetcher — pulls 10-K / 10-Q filings for public comps.

Used for: total revenue, growth rate, gross margin, operating expense ratios
to ground the public-SaaS comp distribution in `models/comps.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

_BASE = "https://data.sec.gov"
_DEFAULT_UA = "DD Agent research contact@example.com"


def _headers() -> dict[str, str]:
    return {"User-Agent": os.environ.get("SEC_EDGAR_USER_AGENT", _DEFAULT_UA)}


@dataclass(frozen=True)
class EdgarFiling:
    ticker: str
    cik: str
    form: str           # "10-K" | "10-Q"
    filed_date: str
    accession: str
    primary_doc_url: str


async def cik_for_ticker(ticker: str) -> str | None:
    """Look up CIK number for a ticker symbol."""
    async with httpx.AsyncClient(timeout=15.0, headers=_headers()) as client:
        try:
            r = await client.get("https://www.sec.gov/files/company_tickers.json")
            r.raise_for_status()
        except Exception:
            return None
    data = r.json()
    ticker_u = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker_u:
            return str(entry["cik_str"]).zfill(10)
    return None


async def recent_filings(ticker: str, form: str = "10-Q", limit: int = 4) -> list[EdgarFiling]:
    """Pull most recent N filings of a given form for a ticker."""
    cik = await cik_for_ticker(ticker)
    if not cik:
        return []
    async with httpx.AsyncClient(timeout=15.0, headers=_headers()) as client:
        try:
            r = await client.get(f"{_BASE}/submissions/CIK{cik}.json")
            r.raise_for_status()
        except Exception:
            return []
    data = r.json()
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accs = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    out: list[EdgarFiling] = []
    for f, d, a, doc in zip(forms, dates, accs, docs):
        if f != form:
            continue
        acc_clean = a.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}"
        out.append(EdgarFiling(ticker.upper(), cik, f, d, a, url))
        if len(out) >= limit:
            break
    return out


async def company_facts(ticker: str) -> dict | None:
    """Pull all reported XBRL facts for a company (raw, large)."""
    cik = await cik_for_ticker(ticker)
    if not cik:
        return None
    async with httpx.AsyncClient(timeout=20.0, headers=_headers()) as client:
        try:
            r = await client.get(f"{_BASE}/api/xbrl/companyfacts/CIK{cik}.json")
            r.raise_for_status()
        except Exception:
            return None
    return r.json()
