"""Reverse DCF: given an asked valuation and current ARR, solve for the
(growth, terminal margin, years) triple that justifies it. Report the
percentile of each component vs the public-SaaS distribution.

Model assumptions (kept transparent so the agent can cite them):
  - Discount rate: 12% (mid-cap tech cost of equity)
  - Years to terminal: 7 by default; agent can sweep 5/7/10
  - Terminal growth: 3% (nominal GDP)
  - Revenue scales at compound `growth_yoy` for `years_to_terminal`
  - Terminal year FCF = revenue * fcf_margin
  - Terminal value = terminal_fcf * (1 + g) / (r - g), discounted back
  - Enterprise value = TV / (1+r)^years (we treat interim FCF as ~0 for
    high-growth SaaS where reinvestment ≈ revenue growth)

Inputs:
  ask_valuation_usd: float — pre-money or post-money
  current_arr_usd: float
  sector: str | None — used to pull comparable public distribution
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReverseDCFResult:
    ask_valuation_usd: float
    current_arr_usd: float
    years_to_terminal: int
    discount_rate: float
    terminal_growth: float
    required_growth_yoy: float | None
    required_fcf_margin: float | None
    implied_terminal_revenue_usd: float | None
    implied_terminal_fcf_usd: float | None
    growth_percentile_vs_public: float | None
    margin_percentile_vs_public: float | None
    interpretation: str

    def to_dict(self) -> dict:
        return {
            "ask_valuation_usd": self.ask_valuation_usd,
            "current_arr_usd": self.current_arr_usd,
            "years_to_terminal": self.years_to_terminal,
            "discount_rate": self.discount_rate,
            "terminal_growth": self.terminal_growth,
            "required_growth_yoy": self.required_growth_yoy,
            "required_fcf_margin": self.required_fcf_margin,
            "implied_terminal_revenue_usd": self.implied_terminal_revenue_usd,
            "implied_terminal_fcf_usd": self.implied_terminal_fcf_usd,
            "growth_percentile_vs_public": self.growth_percentile_vs_public,
            "margin_percentile_vs_public": self.margin_percentile_vs_public,
            "interpretation": self.interpretation,
        }


def _terminal_value(terminal_fcf: float, r: float, g: float) -> float:
    return terminal_fcf * (1 + g) / (r - g)


def _pv(future: float, r: float, years: int) -> float:
    return future / (1 + r) ** years


def solve_required_growth(
    *,
    ask_valuation_usd: float,
    current_arr_usd: float,
    fcf_margin: float = 0.25,
    years_to_terminal: int = 7,
    discount_rate: float = 0.12,
    terminal_growth: float = 0.03,
) -> float | None:
    """Given a target valuation and a fixed terminal margin, solve for the
    annual revenue growth rate that justifies the ask. Returns growth as a
    decimal (e.g. 0.45 = 45%/year)."""
    if current_arr_usd <= 0 or ask_valuation_usd <= 0:
        return None

    def pv_for_growth(g_rev: float) -> float:
        terminal_rev = current_arr_usd * (1 + g_rev) ** years_to_terminal
        terminal_fcf = terminal_rev * fcf_margin
        tv = _terminal_value(terminal_fcf, discount_rate, terminal_growth)
        return _pv(tv, discount_rate, years_to_terminal)

    lo, hi = 0.0, 5.0  # 0% – 500% YoY
    if pv_for_growth(hi) < ask_valuation_usd:
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        if pv_for_growth(mid) < ask_valuation_usd:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def solve_required_margin(
    *,
    ask_valuation_usd: float,
    current_arr_usd: float,
    growth_yoy: float,
    years_to_terminal: int = 7,
    discount_rate: float = 0.12,
    terminal_growth: float = 0.03,
) -> float | None:
    """Given a fixed growth assumption, solve for the terminal FCF margin that
    justifies the ask. Returns margin as a decimal."""
    if current_arr_usd <= 0 or ask_valuation_usd <= 0:
        return None
    terminal_rev = current_arr_usd * (1 + growth_yoy) ** years_to_terminal
    if terminal_rev <= 0:
        return None

    pv_per_unit_fcf = (1 + terminal_growth) / (
        (discount_rate - terminal_growth) * (1 + discount_rate) ** years_to_terminal
    )
    return ask_valuation_usd / (terminal_rev * pv_per_unit_fcf)


def run(
    *,
    ask_valuation_usd: float,
    current_arr_usd: float,
    fcf_margin: float = 0.25,
    years_to_terminal: int = 7,
    discount_rate: float = 0.12,
    terminal_growth: float = 0.03,
    growth_percentile_fn=None,  # callable(growth_yoy) -> percentile | None
) -> ReverseDCFResult:
    """Convenience: solve for required growth holding margin fixed, then
    compute the implied terminal revenue and FCF and (if provided) percentile
    rank that growth against the public comp distribution."""
    g_required = solve_required_growth(
        ask_valuation_usd=ask_valuation_usd,
        current_arr_usd=current_arr_usd,
        fcf_margin=fcf_margin,
        years_to_terminal=years_to_terminal,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
    )

    terminal_rev = (
        current_arr_usd * (1 + g_required) ** years_to_terminal
        if g_required is not None
        else None
    )
    terminal_fcf = terminal_rev * fcf_margin if terminal_rev is not None else None

    growth_percentile = None
    if g_required is not None and growth_percentile_fn is not None:
        try:
            growth_percentile = growth_percentile_fn(g_required)
        except Exception:
            growth_percentile = None

    interpretation = _interpret(g_required, growth_percentile, years_to_terminal, fcf_margin)

    return ReverseDCFResult(
        ask_valuation_usd=ask_valuation_usd,
        current_arr_usd=current_arr_usd,
        years_to_terminal=years_to_terminal,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
        required_growth_yoy=g_required,
        required_fcf_margin=fcf_margin,
        implied_terminal_revenue_usd=terminal_rev,
        implied_terminal_fcf_usd=terminal_fcf,
        growth_percentile_vs_public=growth_percentile,
        margin_percentile_vs_public=None,
        interpretation=interpretation,
    )


def _interpret(growth: float | None, pct: float | None, years: int, margin: float) -> str:
    if growth is None:
        return "Valuation unreachable with default margin assumption — model can't justify this ask."
    g_pct = growth * 100
    body = (
        f"To justify the ask at a {int(margin*100)}% terminal FCF margin over {years} years, "
        f"the company must compound revenue at {g_pct:.0f}% per year."
    )
    if pct is None:
        return body
    if pct >= 90:
        return body + f" That is the {pct:.0f}th percentile of the public-SaaS distribution — outlier-only territory."
    if pct >= 75:
        return body + f" That is the {pct:.0f}th percentile of public SaaS — top-quartile, achievable but rare."
    if pct >= 50:
        return body + f" That is the {pct:.0f}th percentile — within historical norms for strong public SaaS."
    return body + f" That is the {pct:.0f}th percentile — well within the realm of normal public-SaaS performance."


def sweep(
    *,
    ask_valuation_usd: float,
    current_arr_usd: float,
    margins: list[float] = (0.15, 0.20, 0.25, 0.30, 0.35),
    years_options: list[int] = (5, 7, 10),
    discount_rate: float = 0.12,
    terminal_growth: float = 0.03,
) -> list[dict]:
    """Sweep over (margin, years) and return required growth for each cell."""
    rows: list[dict] = []
    for m in margins:
        for y in years_options:
            g = solve_required_growth(
                ask_valuation_usd=ask_valuation_usd,
                current_arr_usd=current_arr_usd,
                fcf_margin=m,
                years_to_terminal=y,
                discount_rate=discount_rate,
                terminal_growth=terminal_growth,
            )
            rows.append({
                "fcf_margin": m,
                "years_to_terminal": y,
                "required_growth_yoy": g,
            })
    return rows
