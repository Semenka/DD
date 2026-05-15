"""Revenue extraction heuristics — GMV, gross, net, MRR, NRR, take rate."""

from dd_agent.ingestion import normalize as norm


def test_extracts_gmv():
    data = norm._extract_heuristic("Marketplace doing $50M GMV growing 80% YoY.", "", "")
    assert data["metrics"]["gmv_usd"] == 50_000_000


def test_extracts_mrr_and_annualizes_to_arr():
    """If only MRR is stated, fill ARR (annualized) and flag the quality."""
    data = norm._extract_heuristic("Doing $200K MRR, no ARR figure disclosed yet.", "", "")
    assert data["metrics"]["mrr_usd"] == 200_000
    assert data["metrics"]["arr_usd"] == 2_400_000
    assert data["metrics"]["arr_quality"] == "unclear"
    assert "annualized" in data["metrics"]["arr_quality_notes"].lower()


def test_mrr_does_not_overwrite_arr():
    """If both ARR and MRR are stated, prefer ARR — don't annualize MRR over it."""
    data = norm._extract_heuristic(
        "Beta Corp doing $3.5M ARR and $300K MRR.", "", ""
    )
    assert data["metrics"]["arr_usd"] == 3_500_000
    assert data["metrics"]["mrr_usd"] == 300_000
    assert data["metrics"].get("arr_quality") is None


def test_extracts_gross_and_net_revenue():
    data = norm._extract_heuristic(
        "Last quarter: gross revenue $12M, net revenue $9M after refunds.", "", ""
    )
    assert data["metrics"]["gross_revenue_usd"] == 12_000_000
    assert data["metrics"]["net_revenue_usd"] == 9_000_000


def test_extracts_take_rate_as_decimal():
    data = norm._extract_heuristic("Take-rate: 8% on every transaction.", "", "")
    assert data["metrics"]["take_rate"] == 0.08


def test_extracts_nrr_as_decimal():
    data = norm._extract_heuristic("NRR: 125%, churn is low.", "", "")
    assert data["metrics"]["net_retention"] == 1.25


def test_arr_takes_priority_over_mrr_when_both_present():
    data = norm._extract_heuristic("ARR $5M, MRR $400K", "", "")
    assert data["metrics"]["arr_usd"] == 5_000_000
    assert data["metrics"]["mrr_usd"] == 400_000


def test_no_revenue_metric_means_no_metrics_key():
    data = norm._extract_heuristic("Some company doing things.", "", "")
    assert "metrics" not in data
