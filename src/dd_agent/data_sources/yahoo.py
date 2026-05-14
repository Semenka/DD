"""Yahoo Finance adapter via yfinance — public comp market data.

Returns ticker price, market cap, enterprise value, revenue (TTM), revenue
growth, gross margin, and EV/Revenue. Used to populate the public-SaaS comp
distribution that `models/comps.py` percentile-ranks against.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class PublicCompFinancials:
    ticker: str
    name: str
    market_cap_usd: float | None
    enterprise_value_usd: float | None
    revenue_ttm_usd: float | None
    revenue_growth_yoy: float | None
    gross_margin: float | None
    ev_to_revenue: float | None


def _fetch_sync(ticker: str) -> PublicCompFinancials | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception:
        return None

    mcap = info.get("marketCap")
    ev = info.get("enterpriseValue")
    rev = info.get("totalRevenue")
    growth = info.get("revenueGrowth")
    gm = info.get("grossMargins")
    ev_rev = info.get("enterpriseToRevenue")

    if all(v is None for v in (mcap, ev, rev)):
        return None

    return PublicCompFinancials(
        ticker=ticker.upper(),
        name=info.get("shortName") or info.get("longName") or ticker,
        market_cap_usd=float(mcap) if mcap else None,
        enterprise_value_usd=float(ev) if ev else None,
        revenue_ttm_usd=float(rev) if rev else None,
        revenue_growth_yoy=float(growth) if growth is not None else None,
        gross_margin=float(gm) if gm is not None else None,
        ev_to_revenue=float(ev_rev) if ev_rev is not None else None,
    )


async def fetch(ticker: str) -> PublicCompFinancials | None:
    return await asyncio.to_thread(_fetch_sync, ticker)


async def fetch_many(tickers: list[str]) -> list[PublicCompFinancials]:
    results = await asyncio.gather(*(fetch(t) for t in tickers))
    return [r for r in results if r is not None]
