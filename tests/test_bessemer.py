"""Bessemer-style memo synthesis — prompt loader sanity + template flow.

The 6th synthesis call itself requires codex/Perplexity, so we don't
end-to-end it here. Instead we test:
  1. The prompt file exists, is non-empty, and contains the 9 required sections.
  2. The renderer threads `bessemer_memo` through the report template.
"""

from pathlib import Path

from dd_agent.context import DealContext
from dd_agent.citations import CitationBook, Citation
from dd_agent.report.renderer import render_markdown


PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "src/dd_agent/modules/bessemer_prompt.md"
)


def test_prompt_file_exists_and_is_substantial():
    text = PROMPT_PATH.read_text(encoding="utf-8")
    assert len(text) > 2000, "Bessemer prompt looks too thin"
    assert "Bessemer" in text


def test_prompt_lists_all_required_sections():
    text = PROMPT_PATH.read_text(encoding="utf-8")
    required = [
        "### Investment Thesis",
        "### Company",
        "### Market",
        "### Why Now",
        "### Team",
        "### Traction",
        "### Outcomes Analysis",
        "### What we'd need to see in the data room",
        "### Recommendation",
    ]
    for section in required:
        assert section in text, f"Bessemer prompt is missing section: {section}"


def test_prompt_forbids_buzzwords():
    text = PROMPT_PATH.read_text(encoding="utf-8")
    # The prompt should explicitly call out banned buzzwords so the LLM avoids them.
    assert "robust" in text.lower()
    assert "synergistic" in text.lower() or "best-in-class" in text.lower()


def test_render_markdown_includes_bessemer_section_when_provided():
    ctx = DealContext(deal_id="abc", company_name="Test Co", sector="ai_devtools")
    book = CitationBook()
    book.add(Citation(key="https://x.com/1", title="Source 1", url="https://x.com/1"))
    bessemer = "### Investment Thesis\n\nTest Co is building Y at the moment Z changes."
    md = render_markdown(
        ctx=ctx,
        synthesis="Exec summary placeholder.",
        market="Market content.",
        founders="Founder content.",
        traction="Traction content.",
        coinvestors="Co-investor content.",
        citations=book,
        bessemer_memo=bessemer,
    )
    assert "Investment Memo (long-form, Bessemer-style)" in md
    assert "Test Co is building Y" in md


def test_render_markdown_omits_bessemer_section_when_none():
    """When the 6th synthesis fails, the report still renders cleanly without
    the Bessemer section."""
    ctx = DealContext(deal_id="abc", company_name="Test Co", sector="ai_devtools")
    book = CitationBook()
    md = render_markdown(
        ctx=ctx,
        synthesis="Synth body.",
        market="m", founders="f", traction="t", coinvestors="c",
        citations=book,
        bessemer_memo=None,
    )
    assert "Investment Memo (long-form, Bessemer-style)" not in md
