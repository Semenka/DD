"""v10 Quality Gate — deterministic layer unit tests.

The deterministic layer is the wrong-company / empty-report backstop. These
golden tests pin its behavior: a good report scores high, a garbage report
(wrong company, no competitors, no founders) scores low, and each individual
check fires on the right signal.
"""

from __future__ import annotations

import pytest

from dd_agent.citations import Citation, CitationBook
from dd_agent.context import DealContext, Founder
from dd_agent.report import quality


# ---------- fixtures --------------------------------------------------------


def _good_ctx() -> DealContext:
    ctx = DealContext(
        deal_id="good",
        company_name="Rivian",
        raw_memo="Rivian builds EVs. Rivian's R1T launched. Rivian is based in Irvine. "
                 "Rivian competes with Tesla. Rivian went public as RIVN.",
        founders=[Founder(name="RJ Scaringe", bio="Founder & CEO, MIT PhD",
                          photo_url="/tmp/rj.jpg", prior_companies=["MIT"])],
    )
    return ctx


def _good_markdown() -> str:
    return """# Rivian

## Synthesis

### Exec summary

**Founders.** RJ Scaringe, MIT PhD, scaled Rivian from 0 to $4.4B revenue [1].

**Co-investors.** Amazon led the Series F; T. Rowe Price participated [2].

**Growth metrics.** $4.4B revenue, 2x YoY [3]. arr_quality: one_time_hardware.

**Competitive position.** Category winner. Competes with Tesla, Ford, GM, Lucid.

### Recommendation
Lean in — proceed to references.

## Market

### Competitor matrix

| Company | Funding | Lead |
|---|---|---|
| Tesla | public | — |
| Ford | public | — |
| Lucid | public | — |
| GM | public | — |

Rivian, Tesla, Ford, Lucid, General Motors, Canoo, Fisker all compete here [4][5].

## Founders

RJ Scaringe founded Rivian [6].
"""


def _garbage_ctx() -> DealContext:
    # The "PK" regression: company name is the ZIP magic header token.
    return DealContext(
        deal_id="pk",
        company_name="PK",
        raw_memo="PK\x03\x04 binary garbage with no real company content here.",
        founders=[],
    )


def _garbage_markdown() -> str:
    return """# PK

## Synthesis

### Exec summary
PK is unknown. Data is undisclosed.

## Market

Nothing found.
"""


def _book(n: int) -> CitationBook:
    b = CitationBook()
    for i in range(n):
        b.add(Citation(key=f"https://x.com/{i}", title=f"S{i}", url=f"https://x.com/{i}"))
    return b


# ---------- whole-report scoring -------------------------------------------


@pytest.mark.asyncio
async def test_good_report_scores_high():
    ctx = _good_ctx()
    md = _good_markdown()
    merged = {"market": _extract_market(md), "citations": _book(12)}
    report = await quality.score_report(
        ctx=ctx, markdown=md, merged=merged,
        extras={"photo_analyses": [{"available": True, "founder_name": "RJ Scaringe"}]},
        use_llm=False,
    )
    assert report.deterministic_score >= 8.0, report.to_dict()
    assert report.passed


@pytest.mark.asyncio
async def test_garbage_report_scores_low():
    ctx = _garbage_ctx()
    md = _garbage_markdown()
    merged = {"market": "Nothing found.", "citations": _book(0)}
    report = await quality.score_report(
        ctx=ctx, markdown=md, merged=merged, extras={}, use_llm=False,
    )
    assert report.deterministic_score < 4.0, report.to_dict()
    assert not report.passed
    assert "company_identity" in report.failed_checks


# ---------- individual checks ----------------------------------------------


def test_company_identity_rejects_section_header():
    ctx = DealContext(deal_id="x", company_name="TERMS", raw_memo="TERMS " * 10)
    r = quality._check_company_identity(ctx)
    assert not r.passed and "section-header" in r.detail


def test_company_identity_requires_three_mentions():
    ctx = DealContext(deal_id="x", company_name="Acme",
                      raw_memo="Acme is a company. Acme builds things.")  # 2x
    r = quality._check_company_identity(ctx)
    assert not r.passed and "appears only 2" in r.detail


def test_company_identity_passes_real_company():
    ctx = _good_ctx()
    r = quality._check_company_identity(ctx)
    assert r.passed


def test_four_pillars_detects_missing():
    md = "**Founders.** x\n**Co-investors.** y"  # missing 2
    r = quality._check_four_pillars(md)
    assert not r.passed
    assert "Growth metrics" in r.detail and "Competitive position" in r.detail


def test_four_pillars_passes_complete():
    md = ("**Founders.** a **Co-investors.** b "
          "**Growth metrics.** c **Competitive position.** d")
    r = quality._check_four_pillars(md)
    assert r.passed


def test_recommendation_detects_decisive():
    md = "### Recommendation\nLead this round."
    assert quality._check_recommendation(md).passed


def test_recommendation_fails_when_absent():
    md = "### Recommendation\nWe are not sure about anything here."
    # "not sure" contains no controlled-vocab token
    assert not quality._check_recommendation(md).passed


def test_citation_density():
    assert quality._check_citations({"citations": _book(10)}, "").passed
    assert not quality._check_citations({"citations": _book(3)}, "").passed


def test_no_leaks_detects_unknown():
    assert not quality._check_no_leaks("Ask: unknown valuation").passed
    assert not quality._check_no_leaks("Revenue: —\n").passed
    assert quality._check_no_leaks("Clean prose with an em—dash inside a word").passed


def test_founders_substantive():
    ctx = DealContext(deal_id="x", company_name="Y",
                      founders=[Founder(name="A B", photo_url="/tmp/a.jpg")])
    assert quality._check_founders(ctx, {}).passed
    ctx2 = DealContext(deal_id="x", company_name="Y",
                       founders=[Founder(name="A B")])
    assert not quality._check_founders(ctx2, {}).passed


# ---------- verdict + gate thresholds --------------------------------------


def test_auto_verdict_tiers():
    assert "Ship-ready" in quality._auto_verdict(8.0, [])
    assert "Borderline" in quality._auto_verdict(5.0, ["competitors_named"])
    assert "LOW CONFIDENCE" in quality._auto_verdict(2.0, ["company_identity"])


@pytest.mark.asyncio
async def test_final_score_is_min_of_layers(monkeypatch):
    """When the LLM rubric scores lower than deterministic, the final score
    takes the min (conservative)."""
    ctx = _good_ctx()
    md = _good_markdown()
    merged = {"market": _extract_market(md), "citations": _book(12)}

    async def fake_llm(ctx, markdown):
        return 5.0, [{"section": "Traction", "why": "no unit economics"}], "thin traction"

    monkeypatch.setattr(quality, "_llm_score", fake_llm)
    report = await quality.score_report(
        ctx=ctx, markdown=md, merged=merged,
        extras={"photo_analyses": [{"available": True, "founder_name": "RJ Scaringe"}]},
        use_llm=True,
    )
    assert report.llm_score == 5.0
    assert report.score == 5.0  # min(det>=8, llm=5)
    assert report.weakest_sections


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_deterministic(monkeypatch):
    ctx = _good_ctx()
    md = _good_markdown()
    merged = {"market": _extract_market(md), "citations": _book(12)}

    async def boom(ctx, markdown):
        raise RuntimeError("gemini down")

    monkeypatch.setattr(quality, "_llm_score", boom)
    report = await quality.score_report(
        ctx=ctx, markdown=md, merged=merged,
        extras={"photo_analyses": [{"available": True, "founder_name": "RJ Scaringe"}]},
        use_llm=True,
    )
    # Falls back to deterministic — still ships a score
    assert report.llm_score is None
    assert report.score == report.deterministic_score


# ---------- helpers ---------------------------------------------------------


def _extract_market(md: str) -> str:
    return quality._extract_section(md, "Market")
