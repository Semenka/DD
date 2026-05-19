"""Photo classifier degrades gracefully without InsightFace or corpus."""

import os

import pytest

from dd_agent.modules import photo_classifier as pc


async def test_disabled_via_env(monkeypatch):
    monkeypatch.setenv("DD_ENABLE_PHOTO_CLASSIFIER", "0")
    result = await pc.analyze_founder_photo(founder_name="X", photo_url=None)
    assert result.available is False
    assert "disabled" in (result.note or "").lower()


async def test_no_photo_bytes_returns_unavailable(monkeypatch):
    monkeypatch.setenv("DD_ENABLE_PHOTO_CLASSIFIER", "1")
    result = await pc.analyze_founder_photo(founder_name="X", photo_url=None)
    assert result.available is False


def test_normalize_unit_vector():
    import numpy as np
    v = pc._normalize(np.array([3.0, 4.0]))
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_traits_constant_matches_module():
    assert set(pc.TRAITS) == {"resilience", "intensity", "warmth", "presentation_polish", "energy"}


def test_founder_match_carries_cohort():
    """The FounderMatch dataclass surfaces cohort so downstream code can group
    nearest matches by public_sp500_nasdaq / yc_top_100 / unicorn_private."""
    m = pc.FounderMatch(
        founder_id="x", company="Stripe", photo_url=None, similarity=0.81,
        cohort="yc_top_100",
    )
    assert m.cohort == "yc_top_100"


def test_photo_analysis_cohort_breakdown_defaults_empty():
    pa = pc.PhotoAnalysis(
        founder_name="X", photo_source="", nearest=[],
        trait_scores={t: 0.0 for t in pc.TRAITS}, available=False,
    )
    assert pa.cohort_breakdown == {}
    assert "cohort_breakdown" in pa.to_dict()


def test_default_k_is_10():
    """k bumped from 5 → 10 with the larger 500-founder corpus."""
    import inspect
    sig = inspect.signature(pc.analyze_founder_photo)
    assert sig.parameters["k"].default == 10
