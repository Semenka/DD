"""Stage canonicalization + ARR-band fallback."""

from dd_agent.ingestion import normalize as norm


# --- canonical name mapping -------------------------------------------------


def test_canonical_stage_recognizes_all_aliases():
    cases = [
        ("Pre-Seed", "pre_seed"),
        ("preseed", "pre_seed"),
        ("Seed", "seed"),
        ("Series A", "series_a"),
        ("series_a", "series_a"),
        ("A", "series_a"),
        ("Series B", "series_b"),
        ("seriesb", "series_b"),
        ("Series C", "series_c_plus"),
        ("Series D", "series_c_plus"),
        ("Series E", "series_c_plus"),
        ("Growth", "growth"),
        ("late_stage", "growth"),
        ("Pre-IPO", "growth"),
    ]
    for raw, expected in cases:
        result = norm._canonical_stage(raw)
        assert result == expected, f"{raw!r} → {result!r}, expected {expected!r}"


def test_canonical_stage_handles_underscored_extensions():
    """`series_a_extension` should still map to series_a via prefix match."""
    assert norm._canonical_stage("series_a_extension") == "series_a"


# --- ARR-band fallback -------------------------------------------------------


def test_arr_band_fallback_when_stage_unknown():
    """When stage is unknown, infer from ARR magnitude."""
    assert norm._canonical_stage(None, arr_usd=500_000) == "seed"
    assert norm._canonical_stage(None, arr_usd=3_000_000) == "series_a"
    assert norm._canonical_stage(None, arr_usd=25_000_000) == "series_b"
    assert norm._canonical_stage(None, arr_usd=150_000_000) == "growth"


def test_arr_band_fallback_edge_cases():
    """Boundary values pick the higher tier."""
    assert norm._canonical_stage(None, arr_usd=1_000_000) == "series_a"
    assert norm._canonical_stage(None, arr_usd=10_000_000) == "series_b"
    assert norm._canonical_stage(None, arr_usd=50_000_000) == "growth"
    # Zero ARR → seed
    assert norm._canonical_stage(None, arr_usd=0) == "seed"


def test_no_stage_no_arr_returns_none():
    """If both are missing, return None — the prompt will default to series_a."""
    assert norm._canonical_stage(None, arr_usd=None) is None
    assert norm._canonical_stage("", arr_usd=None) is None


def test_stage_string_wins_over_arr():
    """When stage is provided, use it even if ARR would suggest a different tier."""
    # Stated Series A but ARR is $100M (would map to growth)
    assert norm._canonical_stage("Series A", arr_usd=100_000_000) == "series_a"


def test_canonical_stages_constant_contents():
    """The CANONICAL_STAGES tuple should contain exactly 6 entries in order."""
    assert norm.CANONICAL_STAGES == (
        "pre_seed", "seed", "series_a", "series_b", "series_c_plus", "growth",
    )
