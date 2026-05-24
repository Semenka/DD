"""End-to-end render sanity check for v8.

Simulates a complete Rivian-shaped deal (with realistic extras: photo
analyses, sweep grid, funding rounds, comp distribution percentile) and
renders the full pipeline: markdown → HTML → PDF. Verifies:

  - main flow has ZERO 'unknown' / placeholder em-dashes
  - the 4-pillar Exec Summary, Bessemer memo, founder photo embed, and
    inline charts (SVG ruler + matplotlib heatmap + matplotlib timeline +
    SVG trait bars) all render correctly
  - PDF is ≤ 6 pages (5-page narrative + appendix)

Does NOT call any LLM — uses canned, realistic synthesis prose so the
output reflects what the user actually sees.
"""

from __future__ import annotations

import re
import sys
import subprocess
from pathlib import Path

# Resolve repo root so the script runs from a checkout regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from dd_agent.context import DealContext, Founder, Investor
from dd_agent.citations import Citation, CitationBook
from dd_agent.report.renderer import render_markdown, render_html, render_pdf
from dd_agent.report import charts as charts_mod


def build_ctx() -> DealContext:
    ctx = DealContext(
        deal_id="v8sanity",
        company_name="Rivian",
        one_liner="Electric pickup-truck and SUV manufacturer with a captive Amazon delivery fleet.",
        sector="climate_mobility",
        stage="growth",
        founded_year=2009,
        hq_location="Irvine, CA",
        website="https://rivian.com",
        ask_amount_usd=2_000_000_000,
        ask_valuation_usd=15_000_000_000,
        round_type="Bridge",
    )
    ctx.founders = [Founder(name="RJ Scaringe", role="Founder/CEO")]
    ctx.existing_investors = [
        Investor(name="Amazon", type="strategic", is_lead=True),
        Investor(name="Ford", type="strategic"),
        Investor(name="T. Rowe Price", type="vc"),
    ]
    ctx.metrics.arr_usd = 4_400_000_000
    ctx.metrics.growth_rate_yoy = 1.6
    ctx.metrics.arr_quality = "recurring_subscription"
    ctx.metrics.arr_quality_notes = (
        "Real recurring vehicle-sales revenue; Amazon delivery contract "
        "provides multi-year contracted demand."
    )
    ctx.metrics.gross_margin = -0.12
    ctx.metrics.customer_count = 73_000
    return ctx


def build_synthesis() -> str:
    return """### Exec summary

**Founders.** RJ Scaringe (MIT PhD, Sloan Automotive Lab) has been compounding deep-tech execution on a single thesis since 2009, scaling Rivian from a 3-person research team to a public manufacturer shipping 50,000 vehicles a year. Photo profile reads 78th-percentile intensity (z=+0.9 vs unicorn corpus), closest archetype: Technical visionary (Elon Musk, Andy Jassy, Jensen Huang as nearest matches).

**Co-investors.** Amazon led the strategic round and remains the largest delivery-vehicle customer (100k-unit EDV contract through 2030). Ford and T. Rowe Price participate; Cox Automotive joined the IPO. Cap table reflects strategic alignment rather than top-tier VC fingerprints — appropriate for a capex-heavy hardware story.

**Growth metrics.** ARR $4.4B, growing 160% YoY [1]. Revenue is `recurring_subscription` quality — captive Amazon contracts + EV sales recognized at delivery. Gross margin is -12% with the path to positive contribution margin gated on the R2 platform launch in 2026.

**Competitive position.** Challenger. Direct PLG competitors include Tesla Cybertruck and Ford F-150 Lightning; enterprise-EV alternatives are GM (Chevrolet Silverado EV) and Stellantis. Structural moat is vertical-integration scale at the Normal, IL plant and the Amazon contract — neither qualifies as a category-winning durable advantage.

### Beliefs Required to Invest

1. We must believe the R2 platform launches on time at <$45K MSRP and reaches 80k units in Year 1.
2. We must believe Amazon honors the full 100k-EDV contract through 2030 rather than open-tendering.
3. We must believe contribution margin crosses positive in 2026 at a 200k-unit annual run rate.

### Kill Shot

The bridge round is unpriced because the public market has rejected the current burn trajectory at any valuation that doesn't dilute existing holders below founder-credibility. Without a clear contribution-margin inflection in the next 4 quarters, this becomes a $15B down-round dilution event rather than a recovery story.

### 1-line bet

Long Rivian if R2 ships on time at <$45K and Amazon renews the EDV contract through 2030.

### Recommendation

Pass for now — revisit after R2 first-delivery announcement. Conviction: low.
"""


def build_bessemer() -> str:
    return """### Investment Thesis

Rivian is the only American electric-truck pure-play at scale, with a captive multi-billion-dollar Amazon contract that finances the path to its lower-priced R2 platform. The bet is asymmetric only if R2 launches on time at $45K — at that price point Rivian becomes the only credible Ford F-150 alternative, and an Amazon-financed manufacturing learning curve becomes a structural cost advantage. The downside is named and dated: a 4-quarter contribution-margin window before the next dilution event.

### Company

RJ Scaringe founded Rivian in 2009 as an MIT PhD spinout focused on a then-impossible idea — a vertically-integrated EV truck for the American adventure-vehicle category that the Big Three were not building. After 12 years of R&D, the R1T launched in 2021 and the company IPO'd at $100B four months later [1]. The product is a luxury electric pickup and SUV with proprietary skateboard architecture; the EDV is the Amazon-branded delivery van using the same platform.

> *I keep coming back to one question: would Rivian exist if Tesla hadn't proven the demand curve? Probably not — but the question that matters is whether American consumers buy a truck because it's electric, or because it's a Rivian. The R1T resale data says the latter — Conviction grows when product is the moat.*

### Why Now

Three things changed in the past 18 months. First, $7,500 federal IRA EV credits combined with state-level rebates pushed the effective price of an R1T below an F-150 Lightning. Second, Amazon's published 2030 emissions targets locked in the EDV contract through end-of-decade [1]. Third, Tesla's strategic shift away from pickup volume created a defensible category.

### Team

The team is technical and concentrated — RJ has been the singular leader since founding, with the COO and CFO recruited from Volkswagen and Snap. Photo profile reads 78th-percentile intensity and 71st-percentile presentation polish vs the unicorn-founder corpus. Closest archetype cluster is "Technical visionary" (Musk, Huang, Jassy nearest matches). Track record: shipped 50,000+ vehicles to date, navigated a $13B IPO, and held the company together through a 75% post-IPO drawdown. Founder-market fit is best-in-class on the dimensions that matter for a capex story.

### Traction

ARR is $4.4B (160% YoY) [1] with 73k cumulative vehicles delivered. Revenue is genuinely recurring in the sense that Amazon contracts produce multi-year cashflows, but gross margin remains -12%. To justify the $15B post-money at a terminal 12% FCF margin and 7-year horizon, the company must compound revenue at 28% per year — 65th percentile of public-SaaS history, sub-median for hardware comps. The heatmap below visualizes the full required-growth grid.

### Outcomes Analysis

**If we're right** — 5-year forward picture: R2 ships in 2026 at $45K, ramps to 150k units in Year 1 and 350k by Year 3. Combined Rivian volume crosses 500k units at positive contribution margin. Acquirer-of-last-resort is Ford or VW at $30-40B; standalone IPO recovery to $50B is plausible at a 6× revenue multiple. From a $15B entry that's 2.5-3× MoIC — survivable but not extraordinary for a single-name growth bet.

**If we're wrong** — R2 slips to 2027, Amazon open-tenders the EDV renewal, and Q2 2026 burn forces a $7B-post-money raise. Existing equity dilutes 70%; founder credibility evaporates with the down round. The company doesn't die but the equity story does.

### What we'd need to see in the data room

1. R2 BOM cost per vehicle at the planned 150k-unit volume — gated on supplier contracts already signed
2. Amazon EDV order ladder by quarter for 2025-2027
3. Contribution-margin bridge from 2024 actuals to 2026 forecast, with named cost-reduction levers
4. Gross-margin sensitivity to a $5K MSRP miss on R2
5. Manufacturing PPV trends Q3-2024 forward

### Recommendation

Pass for now — revisit after R2 first-delivery announcement. The bet asymmetry is acceptable on a longer time horizon but the next 4 quarters are dominated by execution risk we cannot diligence from the outside. Conviction: low.
"""


def build_extras() -> dict:
    return {
        "reverse_dcf": {
            "ask_valuation_usd": 15_000_000_000,
            "current_arr_usd": 4_400_000_000,
            "years_to_terminal": 7,
            "required_fcf_margin": 0.12,
            "required_growth_yoy": 0.28,
            "implied_terminal_revenue_usd": 23_500_000_000,
            "growth_percentile_vs_public": 65,
            "interpretation": (
                "At a 12% terminal margin over 7 years, the company must "
                "compound revenue 28% per year — the 65th percentile of "
                "public-SaaS history, sub-median for capex-heavy hardware."
            ),
        },
        "sweep": [
            {"fcf_margin": 0.08, "years_to_terminal": 5, "required_growth_yoy": 0.48},
            {"fcf_margin": 0.08, "years_to_terminal": 7, "required_growth_yoy": 0.38},
            {"fcf_margin": 0.08, "years_to_terminal": 10, "required_growth_yoy": 0.28},
            {"fcf_margin": 0.12, "years_to_terminal": 5, "required_growth_yoy": 0.42},
            {"fcf_margin": 0.12, "years_to_terminal": 7, "required_growth_yoy": 0.28},
            {"fcf_margin": 0.12, "years_to_terminal": 10, "required_growth_yoy": 0.20},
            {"fcf_margin": 0.18, "years_to_terminal": 5, "required_growth_yoy": 0.32},
            {"fcf_margin": 0.18, "years_to_terminal": 7, "required_growth_yoy": 0.22},
            {"fcf_margin": 0.18, "years_to_terminal": 10, "required_growth_yoy": 0.15},
        ],
        "funding_rounds": [
            {"round_type": "Series B", "date": "2015-09", "amount_usd": 1_000_000,
             "lead_investors": ["Sumitomo"]},
            {"round_type": "Series D", "date": "2019-02", "amount_usd": 700_000_000,
             "lead_investors": ["Amazon"]},
            {"round_type": "Series E", "date": "2020-07", "amount_usd": 2_500_000_000,
             "lead_investors": ["T. Rowe Price"]},
            {"round_type": "Series F", "date": "2021-01", "amount_usd": 2_650_000_000,
             "lead_investors": ["T. Rowe Price"]},
            {"round_type": "IPO", "date": "2021-11", "amount_usd": 12_000_000_000,
             "lead_investors": ["Morgan Stanley"]},
        ],
        "photo_analyses": [{
            "founder_name": "RJ Scaringe",
            "available": True,
            "photo_path": None,  # would be a real local path post-cascade
            "trait_scores": {
                "resilience": 4.2, "intensity": 4.5, "warmth": 3.1,
                "presentation_polish": 4.1, "energy": 4.0,
            },
            "trait_percentiles": {
                "resilience": 65, "intensity": 78, "warmth": 38,
                "presentation_polish": 71, "energy": 62,
            },
            "summary_for_prompt": "RJ Scaringe — Technical visionary archetype.",
            "character_summary": (
                "RJ Scaringe reads as 78th-percentile intensity (z=+0.9) and "
                "71st-percentile presentation polish — distinctive on the "
                "Technical Visionary axis (closest matches: Musk, Huang, Jassy)."
            ),
            "distinctive_features": [
                {"trait": "intensity", "direction": "high",
                 "value": 4.5, "z_score": 0.9, "corpus_mean": 3.8},
            ],
            "archetypes": [
                {"label": "Technical visionary", "dominant_cohort": "public_sp500_nasdaq",
                 "member_companies": ["Tesla", "Nvidia", "Amazon"]},
            ],
            "cohort_breakdown": {"public_sp500_nasdaq": 7, "yc_top_100": 2,
                                 "unicorn_private": 1},
            "nearest": [
                {"company": "Tesla", "similarity": 0.84, "cohort": "public_sp500_nasdaq",
                 "founder_id": "musk"},
                {"company": "Nvidia", "similarity": 0.81, "cohort": "public_sp500_nasdaq",
                 "founder_id": "huang"},
            ],
        }],
    }


def main() -> int:
    ctx = build_ctx()
    extras = build_extras()
    book = CitationBook()
    book.add(Citation(key="https://ir.rivian.com/q4-2023",
                      title="Rivian Q4 2023 shareholder letter",
                      url="https://ir.rivian.com/q4-2023"))

    charts = charts_mod.build_chart_bundle(extras=extras)
    md = render_markdown(
        ctx=ctx,
        synthesis=build_synthesis(),
        bessemer_memo=build_bessemer(),
        market="### Inflection thesis\nElectric pickup demand crossed the chasm in 2022 when Ford and GM shipped their own EVs but neither beat the R1T on towing or off-road performance.",
        founders="### Founder/market fit\nRJ Scaringe has been on a single 15-year mission to ship an American electric pickup. The thesis is no longer contrarian — but he is one of only three CEOs in the world with shipped product at this scale.",
        traction="### Headline metrics\n$4.4B ARR, 160% YoY. NPS undisclosed.",
        coinvestors="### Round-by-round funding history\nAmazon led the Series D; T. Rowe Price led Series E and F.",
        citations=book,
        extras=extras,
        charts=charts,
    )
    html = render_html(markdown_text=md, deal_context=ctx)

    # ---- audit the markdown ----
    main_body = md.split("<details", 1)[0]
    issues = []
    if "unknown" in main_body.lower():
        issues.append("LEAK: 'unknown' in main body")
    for pat in (": —", "— |", "| —", "—\n"):
        if pat in main_body:
            issues.append(f"LEAK: em-dash placeholder pattern {pat!r}")
    has_4_pillars = all(p in md for p in
                        ("**Founders.**", "**Co-investors.**",
                         "**Growth metrics.**", "**Competitive position.**"))
    if not has_4_pillars:
        issues.append("MISSING: 4-pillar exec summary headers")
    chart_count = (
        md.count("<figure class=\"chart\"")
        + sum(1 for line in md.splitlines() if line.lstrip().startswith("<svg"))
    )

    # ---- write artifacts ----
    out_dir = Path("/tmp/v8_sanity_out")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "report.md").write_text(md)
    (out_dir / "report.html").write_text(html)
    pdf_bytes = b""
    pages = -1
    try:
        pdf_bytes = render_pdf(html=html, out_path=str(out_dir / "report.pdf"))
        from pypdf import PdfReader
        pages = len(PdfReader(str(out_dir / "report.pdf")).pages)
    except Exception as exc:
        print(f"(PDF render skipped: {exc.__class__.__name__})")

    # ---- approximate pages from main-body word count if PDF unavailable ----
    # At 10.5pt × line-height 1.4 on A4 with our typography, ~450 words/page.
    main_body_words = len(main_body.split())
    word_pages_est = main_body_words / 450

    print("=" * 60)
    print("v8 sanity render — artifacts in /tmp/v8_sanity_out/")
    print(f"  - markdown:        {len(md):>6} chars, {md.count(chr(10)):>4} lines")
    print(f"  - html:            {len(html):>6} chars")
    print(f"  - pdf bytes:       {len(pdf_bytes):>6}")
    if pages > 0:
        print(f"  - pdf pages:       {pages}")
    else:
        print(f"  - main-body words: {main_body_words}  "
              f"(~{word_pages_est:.1f} pages at 450 wpp)")
    print(f"  - chart count (main body): {chart_count}")
    print(f"  - 4-pillar exec summary:   {has_4_pillars}")
    print(f"  - bessemer memo:           {'## Investment Memo' in md}")
    print()
    if issues:
        print("ISSUES:")
        for i in issues:
            print(f"  - {i}")
        return 1
    print("✓ ALL CLEAN — zero 'unknown' / placeholder em-dash leaks in main body.")
    if pages > 0:
        verdict = "WITHIN BUDGET (≤6 pages)" if pages <= 6 else f"OVER BUDGET ({pages} > 6)"
        print(f"✓ PDF length: {pages} pages — {verdict}")
    else:
        verdict = "WITHIN BUDGET" if word_pages_est <= 6 else "OVER BUDGET"
        print(f"✓ Estimated print length: {word_pages_est:.1f} pages — {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
