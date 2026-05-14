"""Ingestion heuristic fallback (no LLM, no network)."""

import pytest

from dd_agent.ingestion import normalize as norm


def test_heuristic_extracts_basic_company():
    data = norm._extract_heuristic("Acme Robotics raising $5M for our seed round.", "", "")
    assert data["company_name"] == "Acme Robotics"
    assert data["ask_amount_usd"] == 5_000_000


def test_heuristic_extracts_arr():
    data = norm._extract_heuristic("Beta Corp doing $3.5M ARR growing 200% YoY.", "", "")
    assert data["metrics"]["arr_usd"] == 3_500_000


async def test_normalize_without_codex_falls_back_to_heuristic(monkeypatch):
    """If the codex CLI is unavailable, normalize must still succeed via heuristics."""
    monkeypatch.setenv("DD_CODEX_BIN", "/nonexistent/codex-binary-for-test")
    ctx = await norm.normalize(
        memo_text="Gamma Labs raising $10M seed. ARR $500K.",
        deck_text=None,
        website_text=None,
    )
    assert ctx.company_name == "Gamma Labs"
    assert ctx.ask_amount_usd == 10_000_000
    assert ctx.metrics.arr_usd == 500_000


def test_parse_json_block_handles_fences():
    out = norm._parse_json_block('```json\n{"a": 1, "b": "x"}\n```')
    assert out == {"a": 1, "b": "x"}


def test_parse_json_block_handles_bare_object():
    out = norm._parse_json_block('preamble {"a": 1} trailing')
    assert out == {"a": 1}
