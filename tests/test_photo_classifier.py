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
