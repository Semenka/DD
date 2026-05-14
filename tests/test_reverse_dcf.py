"""Reverse DCF math sanity checks.

These don't need any external API — they only test the deterministic solver.
"""

import math

import pytest

from dd_agent.models import reverse_dcf as r


def test_required_growth_monotonic_in_valuation():
    """Higher valuation must require higher growth."""
    g_low = r.solve_required_growth(
        ask_valuation_usd=100_000_000, current_arr_usd=10_000_000,
    )
    g_high = r.solve_required_growth(
        ask_valuation_usd=500_000_000, current_arr_usd=10_000_000,
    )
    assert g_low is not None and g_high is not None
    assert g_high > g_low


def test_required_growth_monotonic_in_arr():
    """Same valuation but higher ARR must require lower growth."""
    g_small = r.solve_required_growth(
        ask_valuation_usd=200_000_000, current_arr_usd=5_000_000,
    )
    g_big = r.solve_required_growth(
        ask_valuation_usd=200_000_000, current_arr_usd=50_000_000,
    )
    assert g_small is not None and g_big is not None
    assert g_small > g_big


def test_required_growth_reproduces_pv_identity():
    """At the solved growth, the discounted terminal value should equal the
    requested valuation (within numerical tolerance)."""
    arr = 10_000_000
    target = 250_000_000
    margin = 0.25
    years = 7
    r_rate = 0.12
    g_term = 0.03
    g_rev = r.solve_required_growth(
        ask_valuation_usd=target, current_arr_usd=arr,
        fcf_margin=margin, years_to_terminal=years,
        discount_rate=r_rate, terminal_growth=g_term,
    )
    assert g_rev is not None
    terminal_rev = arr * (1 + g_rev) ** years
    terminal_fcf = terminal_rev * margin
    tv = terminal_fcf * (1 + g_term) / (r_rate - g_term)
    pv = tv / (1 + r_rate) ** years
    assert math.isclose(pv, target, rel_tol=1e-3)


def test_sweep_shape():
    rows = r.sweep(ask_valuation_usd=200_000_000, current_arr_usd=10_000_000)
    margins = sorted({row["fcf_margin"] for row in rows})
    years = sorted({row["years_to_terminal"] for row in rows})
    assert margins == [0.15, 0.20, 0.25, 0.30, 0.35]
    assert years == [5, 7, 10]
    assert len(rows) == len(margins) * len(years)


def test_solve_required_margin_inverts():
    """If we solve for growth at margin X, then solve for margin at that growth,
    we should recover X."""
    arr = 8_000_000
    target = 150_000_000
    margin = 0.22
    years = 7
    g = r.solve_required_growth(
        ask_valuation_usd=target, current_arr_usd=arr,
        fcf_margin=margin, years_to_terminal=years,
    )
    assert g is not None
    m_back = r.solve_required_margin(
        ask_valuation_usd=target, current_arr_usd=arr,
        growth_yoy=g, years_to_terminal=years,
    )
    assert m_back is not None
    assert math.isclose(m_back, margin, rel_tol=1e-3)


def test_run_returns_interpretation_string():
    result = r.run(
        ask_valuation_usd=200_000_000, current_arr_usd=10_000_000,
        growth_percentile_fn=lambda g: 92.0,
    )
    assert result.required_growth_yoy is not None
    assert "92" in result.interpretation or "percentile" in result.interpretation.lower()


def test_zero_arr_returns_none():
    g = r.solve_required_growth(ask_valuation_usd=200_000_000, current_arr_usd=0)
    assert g is None
