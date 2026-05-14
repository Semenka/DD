"""Public-SaaS comp distribution.

Loads `data/public_comps.json`, fetches financials via `data_sources.yahoo`,
computes EV/Revenue and growth percentiles segmented by sector and growth band.

For a target deal:
  - implied_valuation_range(target_arr, target_growth, sector) → (p25, p50, p75) EV/ARR
  - percentile_for(growth_yoy, sector) → 0..100
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..data_sources.yahoo import PublicCompFinancials, fetch_many


def _comp_universe_path() -> Path:
    return Path(os.environ.get("DD_DATA_DIR", "./data")) / "public_comps.json"


def load_universe() -> list[dict]:
    path = _comp_universe_path()
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    tickers = data.get("tickers", [])
    seen: set[str] = set()
    out: list[dict] = []
    for t in tickers:
        sym = t.get("ticker")
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(t)
    return out


@dataclass
class CompDistribution:
    sector: str | None
    comps: list[PublicCompFinancials] = field(default_factory=list)
    ev_revenue_p25: float | None = None
    ev_revenue_p50: float | None = None
    ev_revenue_p75: float | None = None
    growth_p25: float | None = None
    growth_p50: float | None = None
    growth_p75: float | None = None

    def implied_valuation(self, target_revenue: float) -> dict[str, float | None]:
        return {
            "low_p25": (self.ev_revenue_p25 * target_revenue) if self.ev_revenue_p25 else None,
            "mid_p50": (self.ev_revenue_p50 * target_revenue) if self.ev_revenue_p50 else None,
            "high_p75": (self.ev_revenue_p75 * target_revenue) if self.ev_revenue_p75 else None,
        }

    def growth_percentile(self, growth_yoy: float | None) -> float | None:
        if growth_yoy is None or not self.comps:
            return None
        growths = [c.revenue_growth_yoy for c in self.comps if c.revenue_growth_yoy is not None]
        if not growths:
            return None
        rank = sum(1 for g in growths if g < growth_yoy)
        return 100.0 * rank / len(growths)

    def to_dict(self) -> dict:
        return {
            "sector": self.sector,
            "n_comps": len(self.comps),
            "tickers": [c.ticker for c in self.comps],
            "ev_revenue_p25": self.ev_revenue_p25,
            "ev_revenue_p50": self.ev_revenue_p50,
            "ev_revenue_p75": self.ev_revenue_p75,
            "growth_p25": self.growth_p25,
            "growth_p50": self.growth_p50,
            "growth_p75": self.growth_p75,
        }


async def build_distribution(sector: str | None = None) -> CompDistribution:
    """Build a comp distribution. If sector is given, filter to that sector first;
    if too few comps remain (<5), fall back to the full universe."""
    universe = load_universe()
    selected = [t for t in universe if not sector or t.get("sector") == sector]
    if len(selected) < 5:
        selected = universe

    tickers = [t["ticker"] for t in selected]
    comps = await fetch_many(tickers)
    return _stats(comps, sector)


def _stats(comps: list[PublicCompFinancials], sector: str | None) -> CompDistribution:
    ev_rev = np.array([c.ev_to_revenue for c in comps if c.ev_to_revenue is not None])
    growths = np.array([c.revenue_growth_yoy for c in comps if c.revenue_growth_yoy is not None])

    dist = CompDistribution(sector=sector, comps=comps)
    if ev_rev.size:
        dist.ev_revenue_p25 = float(np.percentile(ev_rev, 25))
        dist.ev_revenue_p50 = float(np.percentile(ev_rev, 50))
        dist.ev_revenue_p75 = float(np.percentile(ev_rev, 75))
    if growths.size:
        dist.growth_p25 = float(np.percentile(growths, 25))
        dist.growth_p50 = float(np.percentile(growths, 50))
        dist.growth_p75 = float(np.percentile(growths, 75))
    return dist
