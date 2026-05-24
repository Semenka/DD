"""v8 chart helpers — each helper must produce valid SVG or a non-empty
base64-encoded PNG inside an `<img>` element, OR return '' when input
data is insufficient. Nothing in the chart pipeline is allowed to raise."""

from __future__ import annotations

import re

import pytest

from dd_agent.report import charts


# ---------- svg_percentile_ruler -------------------------------------------


def test_percentile_ruler_renders_valid_svg():
    svg = charts.svg_percentile_ruler(percentile=78, label="Growth")
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert 'xmlns="http://www.w3.org/2000/svg"' in svg
    assert "78th percentile" in svg
    assert "Growth" in svg


def test_percentile_ruler_returns_empty_for_none():
    assert charts.svg_percentile_ruler(percentile=None) == ""


def test_percentile_ruler_rejects_out_of_range():
    assert charts.svg_percentile_ruler(percentile=-5) == ""
    assert charts.svg_percentile_ruler(percentile=150) == ""


def test_percentile_ruler_handles_string_input():
    # Real world: extras values often arrive as floats stringified
    svg = charts.svg_percentile_ruler(percentile="78", label="x")
    assert svg.startswith("<svg")


# ---------- svg_trait_bars -------------------------------------------------


def test_trait_bars_renders_all_five_traits():
    pct = {
        "resilience": 60, "intensity": 92, "warmth": 30,
        "presentation_polish": 78, "energy": 70,
    }
    svg = charts.svg_trait_bars(trait_percentiles=pct)
    assert svg.startswith("<svg")
    for trait in pct:
        assert trait in svg
    # Each percentile should show as "<n>th" label
    assert "92th" in svg
    assert "30th" in svg


def test_trait_bars_renders_subset():
    """Test should pass even if only some traits are populated."""
    svg = charts.svg_trait_bars(trait_percentiles={"intensity": 88})
    assert svg.startswith("<svg")
    assert "intensity" in svg
    assert "88th" in svg


def test_trait_bars_empty_returns_empty():
    assert charts.svg_trait_bars(trait_percentiles=None) == ""
    assert charts.svg_trait_bars(trait_percentiles={}) == ""


def test_trait_bars_includes_score_when_provided():
    svg = charts.svg_trait_bars(
        trait_percentiles={"intensity": 92},
        trait_scores={"intensity": 4.6},
    )
    # Score shown as "(4.6/5)"
    assert "(4.6/5)" in svg


# ---------- svg_cohort_donut -----------------------------------------------


def test_cohort_donut_renders_segments():
    svg = charts.svg_cohort_donut(cohort_breakdown={
        "yc_top_100": 4, "unicorn_private": 3, "public_sp500_nasdaq": 3,
    })
    assert svg.startswith("<svg")
    # n total in middle
    assert "n=10" in svg
    # All cohorts named in legend
    assert "yc_top_100" in svg
    assert "unicorn_private" in svg
    assert "public_sp500_nasdaq" in svg
    # 3 path arcs (one per non-empty cohort)
    assert svg.count("<path") == 3


def test_cohort_donut_empty_returns_empty():
    assert charts.svg_cohort_donut(cohort_breakdown=None) == ""
    assert charts.svg_cohort_donut(cohort_breakdown={}) == ""
    assert charts.svg_cohort_donut(cohort_breakdown={"x": 0}) == ""


# ---------- png_dcf_heatmap ------------------------------------------------


def _valid_data_url(html: str) -> bool:
    """Helper: validate the returned HTML contains a data:image/png URL with
    actual base64-encoded content."""
    m = re.search(r'src="data:image/png;base64,([^"]+)"', html)
    return bool(m and len(m.group(1)) > 200)


def test_dcf_heatmap_renders_with_real_sweep():
    sweep = [
        {"fcf_margin": 0.25, "years_to_terminal": 5, "required_growth_yoy": 0.65},
        {"fcf_margin": 0.25, "years_to_terminal": 7, "required_growth_yoy": 0.45},
        {"fcf_margin": 0.25, "years_to_terminal": 10, "required_growth_yoy": 0.30},
        {"fcf_margin": 0.35, "years_to_terminal": 5, "required_growth_yoy": 0.55},
        {"fcf_margin": 0.35, "years_to_terminal": 7, "required_growth_yoy": 0.40},
        {"fcf_margin": 0.35, "years_to_terminal": 10, "required_growth_yoy": 0.28},
    ]
    html = charts.png_dcf_heatmap(sweep=sweep)
    assert html.startswith("<figure")
    assert _valid_data_url(html)
    assert "Required annual revenue growth" in html


def test_dcf_heatmap_empty_returns_empty():
    assert charts.png_dcf_heatmap(sweep=None) == ""
    assert charts.png_dcf_heatmap(sweep=[]) == ""


def test_dcf_heatmap_all_unreachable_returns_empty():
    """If every cell has required_growth_yoy=None, render nothing."""
    sweep = [
        {"fcf_margin": 0.25, "years_to_terminal": 5, "required_growth_yoy": None},
        {"fcf_margin": 0.35, "years_to_terminal": 7, "required_growth_yoy": None},
    ]
    assert charts.png_dcf_heatmap(sweep=sweep) == ""


# ---------- png_funding_timeline -------------------------------------------


def test_funding_timeline_renders_real_rounds():
    rounds = [
        {"round_type": "Series A", "date": "2019-03", "amount_usd": 10_000_000,
         "lead_investors": ["Sequoia"]},
        {"round_type": "Series B", "date": "2021-06", "amount_usd": 50_000_000,
         "lead_investors": ["Founders Fund"]},
        {"round_type": "Series C", "date": "2023-09", "amount_usd": 200_000_000,
         "lead_investors": ["a16z"]},
    ]
    html = charts.png_funding_timeline(rounds=rounds)
    assert html.startswith("<figure")
    assert _valid_data_url(html)


def test_funding_timeline_skips_rounds_without_date():
    rounds = [
        {"round_type": "Seed", "date": None, "amount_usd": 1_000_000},
        {"round_type": "Series A", "date": "2020-03", "amount_usd": 10_000_000},
        # second row alone won't render — need ≥1 plottable point. With 1 point
        # we still produce a figure (single bubble).
    ]
    html = charts.png_funding_timeline(rounds=rounds)
    # With one plottable point matplotlib still produces a figure
    assert html.startswith("<figure")


def test_funding_timeline_all_missing_returns_empty():
    rounds = [
        {"round_type": "Seed", "date": None, "amount_usd": None},
    ]
    assert charts.png_funding_timeline(rounds=rounds) == ""


def test_funding_timeline_empty_returns_empty():
    assert charts.png_funding_timeline(rounds=None) == ""
    assert charts.png_funding_timeline(rounds=[]) == ""


# ---------- png_arr_trajectory ---------------------------------------------


def test_arr_trajectory_needs_two_points():
    html = charts.png_arr_trajectory(points=[("Q1'23", 1_000_000)])
    assert html == ""


def test_arr_trajectory_renders_series():
    pts = [("Q1'23", 1_000_000), ("Q3'23", 2_500_000), ("Q1'24", 5_000_000)]
    html = charts.png_arr_trajectory(points=pts)
    assert html.startswith("<figure")
    assert _valid_data_url(html)


# ---------- build_chart_bundle --------------------------------------------


def test_build_chart_bundle_empty_extras():
    """No data → no charts. Bundle is a dict but every entry is '' or missing."""
    bundle = charts.build_chart_bundle(extras={})
    # Should not crash, should return a dict
    assert isinstance(bundle, dict)
    # No keys should have non-empty values
    for v in bundle.values():
        assert v == "" or v == {} or v is None


def test_build_chart_bundle_full_extras():
    extras = {
        "reverse_dcf": {"growth_percentile_vs_public": 78},
        "sweep": [
            {"fcf_margin": 0.25, "years_to_terminal": 5, "required_growth_yoy": 0.65},
            {"fcf_margin": 0.35, "years_to_terminal": 10, "required_growth_yoy": 0.28},
        ],
        "funding_rounds": [
            {"round_type": "Series A", "date": "2020-03",
             "amount_usd": 10_000_000, "lead_investors": ["Sequoia"]},
            {"round_type": "Series B", "date": "2022-09",
             "amount_usd": 80_000_000, "lead_investors": ["a16z"]},
        ],
        "photo_analyses": [{
            "founder_name": "RJ Scaringe",
            "available": True,
            "trait_percentiles": {"intensity": 92, "resilience": 78},
            "trait_scores": {"intensity": 4.6, "resilience": 4.0},
        }],
    }
    bundle = charts.build_chart_bundle(extras=extras)
    assert bundle["market_comp_ruler"].startswith("<svg")
    assert bundle["dcf_heatmap"].startswith("<figure")
    assert bundle["funding_timeline"].startswith("<figure")
    assert "trait_bars_by_founder" in bundle
    assert "RJ Scaringe" in bundle["trait_bars_by_founder"]
    assert bundle["trait_bars_by_founder"]["RJ Scaringe"].startswith("<svg")
