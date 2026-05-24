"""v8 sparse-context leak audit.

The user requirement is verbatim: *"No 'unknown' parts in the memo (if smth
unknown — remove section)"*. To enforce this, render the template with a
maximally-sparse `DealContext` (almost every field None) and assert that
NO `unknown` / `— |` / `(speculation)` / `None` strings leak through the
template's static parts.

LLM-generated section content is opaque to this test — we only audit the
template's emitted text. The synthesis/bessemer prompts enforce the
parallel rule on the LLM side (tested in test_synthesis_prompt.py)."""

from __future__ import annotations

from dd_agent.citations import Citation, CitationBook
from dd_agent.context import DealContext
from dd_agent.report.renderer import render_markdown


def _empty_ctx() -> DealContext:
    """A DealContext where every optional field is None."""
    return DealContext(
        deal_id="deadbeef",
        company_name="Stealth Co",
        # Everything else stays at default (None / empty / 0).
    )


def _render_with_sparse_inputs(synthesis="Synthesis body.", market="Market.",
                               founders="Founders.", traction="Traction.",
                               coinvestors="Co-investors.") -> str:
    """Render the template with a sparse context + minimal section bodies."""
    return render_markdown(
        ctx=_empty_ctx(),
        synthesis=synthesis,
        market=market,
        founders=founders,
        traction=traction,
        coinvestors=coinvestors,
        citations=CitationBook(),
    )


# ---------- the leak audit -------------------------------------------------


def test_no_unknown_string_in_top_matter():
    md = _render_with_sparse_inputs()
    # The hard contract: nowhere in the rendered markdown should the literal
    # 'unknown' appear (case-insensitive). 'unknown' came from the old
    # template's `{{ ctx.ask_amount_usd or 'unknown' }}` pattern.
    # Allow it inside LLM-rendered section bodies, but the template itself
    # must not emit it — and our sparse-input test bodies are clean prose
    # that contain no 'unknown'.
    assert "unknown" not in md.lower()


def test_no_emdash_placeholder_in_main_flow():
    """The em-dash '—' was emitted ~60 times as a missing-field placeholder.
    With everything None, the template should emit ZERO placeholder em-dashes
    (em-dashes used as typography in headings are fine — we just check for
    the specific 'missing-data' patterns: `: —`, `— |`, `| —`)."""
    md = _render_with_sparse_inputs()
    main_body = md.split("<details", 1)[0]
    leak_patterns = [": —", "— |", "| —", "—\n"]
    for p in leak_patterns:
        assert p not in main_body, (
            f"em-dash placeholder pattern {p!r} leaked into main body:\n{main_body}"
        )


def test_omits_top_matter_lines_with_no_data():
    md = _render_with_sparse_inputs()
    # When ctx fields are all None, the template should not render the
    # corresponding bullet lines at all.
    assert "- **Ask:**" not in md
    assert "- **Round:**" not in md
    assert "- **Founded:**" not in md
    assert "- **HQ:**" not in md
    assert "- **Website:**" not in md


def test_renders_provided_top_matter_lines():
    ctx = DealContext(
        deal_id="abc", company_name="Rivian",
        ask_amount_usd=200_000_000.0,
        ask_valuation_usd=15_000_000_000.0,
        round_type="Series F",
        website="https://rivian.com",
        founded_year=2009,
        hq_location="Irvine, CA",
    )
    md = render_markdown(
        ctx=ctx, synthesis="s", market="m", founders="f",
        traction="t", coinvestors="c", citations=CitationBook(),
    )
    assert "- **Ask:** $200,000,000 at $15,000,000,000 valuation" in md
    assert "- **Round:** Series F" in md
    assert "- **Founded:** 2009" in md
    assert "- **HQ:** Irvine, CA" in md
    assert "[https://rivian.com](https://rivian.com)" in md
    assert "unknown" not in md.lower()


def test_appendix_block_present():
    """The collapsible appendix is always rendered (even when empty) so the
    HTML reader sees the consistent structural element. It's a tiny static
    block — no leaks possible."""
    md = _render_with_sparse_inputs()
    assert "<details" in md
    assert "Diligence appendix" in md
    assert "</details>" in md


def test_appendix_skips_revenue_metrics_when_all_none():
    md = _render_with_sparse_inputs()
    # With every metric None, the "Revenue metrics" subsection must not appear
    assert "### Revenue metrics" not in md
    # Likewise no Reverse DCF, no Round-by-round history, no notice.co
    assert "### Reverse DCF" not in md
    assert "### Round-by-round history" not in md
    assert "notice.co" not in md.lower()
    assert "Comparable exits" not in md


def test_appendix_renders_revenue_metrics_when_data_present():
    ctx = DealContext(deal_id="x", company_name="Test")
    ctx.metrics.arr_usd = 5_000_000
    ctx.metrics.growth_rate_yoy = 1.5
    ctx.metrics.arr_quality = "recurring_subscription"
    md = render_markdown(
        ctx=ctx, synthesis="s", market="m", founders="f",
        traction="t", coinvestors="c", citations=CitationBook(),
    )
    assert "### Revenue metrics" in md
    assert "ARR: $5,000,000" in md
    assert "Growth (YoY): 150%" in md
    assert "recurring_subscription" in md
    # Negative — fields that ARE None must not appear with `—` placeholder
    assert "MRR" not in md  # mrr_usd is None
    assert "GMV" not in md  # gmv_usd is None


def test_charts_embedded_when_provided():
    """When the orchestrator passes charts={'dcf_heatmap': '<figure>...'},
    the template embeds them inline in the Traction section."""
    md = render_markdown(
        ctx=_empty_ctx(),
        synthesis="s", market="m", founders="f", traction="t", coinvestors="c",
        citations=CitationBook(),
        charts={
            "dcf_heatmap": "<figure class=\"chart\">FAKE_HEATMAP</figure>",
            "funding_timeline": "<figure class=\"chart\">FAKE_TIMELINE</figure>",
            "market_comp_ruler": "<svg>FAKE_RULER</svg>",
        },
    )
    assert "FAKE_HEATMAP" in md
    assert "FAKE_TIMELINE" in md
    assert "FAKE_RULER" in md


def test_charts_absent_when_not_provided():
    md = _render_with_sparse_inputs()
    # No chart strings in input → no chart HTML in output
    assert "<figure class=\"chart\">" not in md
    # The static SVG-marker comment that the template uses is also absent
    assert "Reverse-DCF — required growth" not in md
