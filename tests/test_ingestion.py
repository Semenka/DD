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


def test_heuristic_finds_company_from_explicit_line():
    """The 'Company: X' line in a structured memo should always win over later text."""
    memo = (
        "# Investment Memo — Linear\n"
        "**Company:** Linear\n"
        "**Sector:** ai_devtools\n"
    )
    data = norm._extract_heuristic(memo, "", "")
    assert data["company_name"] == "Linear"


def test_heuristic_finds_company_from_memo_title():
    """The '# Investment Memo — X' title should fall through when no Company line exists."""
    memo = "# Investment Memo — Stripe\n\nSomething about Stripe.\n"
    data = norm._extract_heuristic(memo, "", "")
    assert data["company_name"] == "Stripe"


def test_heuristic_extracts_stage_and_valuation():
    memo = "**Stage:** Series B\nRaising $50M at $400M valuation.\n"
    data = norm._extract_heuristic(memo, "", "")
    assert data["stage"] == "series_b"
    assert data["ask_amount_usd"] == 50_000_000
    assert data["ask_valuation_usd"] == 400_000_000


def test_merge_overlay_wins_for_non_null():
    base = {"a": 1, "b": None, "metrics": {"x": 10, "y": None}}
    overlay = {"a": 2, "b": 3, "metrics": {"y": 20, "z": None}}
    out = norm._merge(base, overlay)
    assert out["a"] == 2
    assert out["b"] == 3
    assert out["metrics"]["x"] == 10
    assert out["metrics"]["y"] == 20


def test_merge_keeps_base_when_overlay_is_null():
    base = {"company_name": "Linear"}
    overlay = {"company_name": None}
    out = norm._merge(base, overlay)
    assert out["company_name"] == "Linear"


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
