"""Tests for ingestion/identity.py — company-identity verifier.

The Rivian regression: a .docx file was extracted as "PK" (ZIP magic header)
and a 14-minute pipeline ran on the wrong company. The identity verifier
catches this whole class of failures before the pipeline starts."""

from __future__ import annotations

import pytest

from dd_agent.ingestion.identity import (
    VerifyResult,
    _from_filename,
    _most_frequent_capitalized,
    _passes_body_check,
    verify_company_identity,
)


# Sample body that mentions Rivian many times.
RIVIAN_BODY = """
Rivian Automotive designs and manufactures electric vehicles. Rivian's R1T pickup
launched in 2021. Rivian's R1S SUV launched in 2022. Rivian operates a plant in
Normal, Illinois. Rivian's Series F round was led by Amazon. Rivian went public
on NASDAQ in November 2021 as RIVN. Rivian competes with Ford F-150 Lightning,
Tesla Cybertruck, and the Chevrolet Silverado EV. Rivian's founder is RJ
Scaringe. Rivian reported $4.4B in revenue in 2023.
"""


# Body where the LLM mis-extracted "PK" but the real company is Rivian.
PK_REGRESSION_BODY = RIVIAN_BODY


# ---------- _passes_body_check ----------


def test_passes_body_check_finds_company():
    assert _passes_body_check("Rivian", RIVIAN_BODY, min_count=3) is True


def test_passes_body_check_case_insensitive():
    assert _passes_body_check("rivian", RIVIAN_BODY, min_count=3) is True
    assert _passes_body_check("RIVIAN", RIVIAN_BODY, min_count=3) is True


def test_passes_body_check_rejects_stop_words():
    """Even if 'PK' appeared 10x, the stop-word filter should reject it."""
    body = "PK PK PK PK PK PK PK PK PK PK PK Rivian"
    assert _passes_body_check("PK", body, min_count=3) is False


def test_passes_body_check_requires_minimum_count():
    body = "Acme is a company. Acme builds widgets."  # 2 occurrences
    assert _passes_body_check("Acme", body, min_count=3) is False
    assert _passes_body_check("Acme", body, min_count=2) is True


def test_passes_body_check_empty_inputs():
    assert _passes_body_check("", RIVIAN_BODY, 3) is False
    assert _passes_body_check("Rivian", "", 3) is False


# ---------- _from_filename ----------


def test_from_filename_strips_meta_tokens():
    assert _from_filename("Rivan_Investment_Memo.docx") == "Rivan"
    assert _from_filename("Acme-DD-Series-A.pdf") == "Acme"
    assert _from_filename("/absolute/path/Linear_Deck.pdf") == "Linear"


def test_from_filename_handles_none():
    assert _from_filename(None) is None
    assert _from_filename("") is None


def test_from_filename_all_stop_words_returns_none():
    # All tokens are deal-meta words → no candidate
    assert _from_filename("Investment_Memo_Deck.pdf") is None


# ---------- _most_frequent_capitalized ----------


def test_most_frequent_capitalized_finds_rivian():
    result = _most_frequent_capitalized(RIVIAN_BODY, min_count=3)
    assert result == "Rivian"


def test_most_frequent_capitalized_skips_stop_words():
    # "Company" appears many times but is a stop word.
    body = "Company Company Company Company Company Acme Acme Acme"
    result = _most_frequent_capitalized(body, min_count=3)
    assert result == "Acme"


def test_most_frequent_capitalized_below_threshold_returns_none():
    body = "Acme is a thing. Bcme is another."
    assert _most_frequent_capitalized(body, min_count=5) is None


# ---------- verify_company_identity (top-level) ----------


@pytest.mark.asyncio
async def test_verify_accepts_correct_extraction():
    result = await verify_company_identity(
        extracted_name="Rivian",
        raw_memo=RIVIAN_BODY,
    )
    assert result.verified is True
    assert result.company_name == "Rivian"
    assert result.source == "memo-frequency"


@pytest.mark.asyncio
async def test_verify_rescues_via_filename():
    """LLM returned 'PK' (ZIP magic). Filename says 'Acme' and body
    mentions Acme once — the filename rescue should kick in."""
    body = "Acme has been building widgets since 2019. The team is small."
    result = await verify_company_identity(
        extracted_name="PK",
        raw_memo=body,
        source_filename="Acme_Investment_Memo.docx",
    )
    assert result.verified is True
    assert result.company_name == "Acme"
    assert result.source == "filename"
    assert result.original_name == "PK"


@pytest.mark.asyncio
async def test_verify_rescues_via_frequent_noun():
    """LLM returned 'PK'. Filename gives no signal but the body
    overwhelmingly mentions Rivian — frequency rescue wins."""
    result = await verify_company_identity(
        extracted_name="PK",
        raw_memo=RIVIAN_BODY,
        source_filename="memo.docx",   # filename is generic
    )
    assert result.verified is True
    assert result.company_name == "Rivian"
    assert result.source == "memo-frequency"
    assert result.original_name == "PK"


@pytest.mark.asyncio
async def test_verify_refuses_when_no_signal_anywhere(monkeypatch):
    """Body is too short and has no recognizable company name. Filename
    is generic. Grounded rescue is short-circuited (body < 200 chars).
    Verifier should refuse the pipeline."""
    result = await verify_company_identity(
        extracted_name="PK",
        raw_memo="Short body. Nothing here.",
        source_filename="memo.docx",
    )
    assert result.verified is False
    assert result.company_name is None
    assert result.source == "refused"
    assert result.notes and "could not confirm" in result.notes.lower()


@pytest.mark.asyncio
async def test_verify_uses_deck_text_as_fallback():
    """Memo is empty but the deck text contains many Rivian mentions."""
    result = await verify_company_identity(
        extracted_name="PK",
        raw_memo="",
        raw_deck=RIVIAN_BODY,
        source_filename="memo.docx",
    )
    assert result.verified is True
    assert result.company_name == "Rivian"
