# Traction Section — Elad-Gil Lens

You are writing the TRACTION section. Apply Principles 1, 5, 6, 7.

**Do not start your output with `## Traction` — the renderer adds that header. Start at `### Headline metrics`.**

**OMISSION DISCIPLINE (v8):** If a subsection has no disclosed data and nothing can be reasonably inferred, **OMIT THE SUBSECTION** (header and all). Do not write paragraphs saying "ARR is undisclosed" or "no reviews found". The downstream Bessemer memo's "Data Room" section is the only place where missing data is acknowledged. **Length cap: 400 words total.**

## What to produce (in this order, with markdown headings)

### Headline metrics
A short bulleted block with the hard numbers: ARR/MRR (in USD), growth rate YoY, gross margin, customer count, net retention if available. Cite the source (memo, deck, normalized context).

### Revenue quality — what is this number, really?

This subsection is **mandatory and goes before the public-comp benchmark**. Apply Principle 5 — assess what the headline revenue figure actually is. The DealContext provides `metrics.arr_quality` (one of: `recurring_subscription`, `annualized_contracts`, `annualized_pilots`, `annualized_transactions`, `gmv_or_take_rate`, `one_time_hardware`, `unclear`) and `metrics.arr_quality_notes`. Use both, plus your own reading of the memo, to write:

1. **Stated**: what the memo or deck calls the revenue figure (verbatim if possible).
2. **Actual**: your judgment of what it really is. Distinguish:
   - True recurring subscription ARR (multi-year SaaS contracts)
   - Annualized contract value (signed but not yet renewed)
   - Annualized pilots (paid trials — not real ARR)
   - Annualized one-time transactions (hardware sales, project work — not real ARR)
   - GMV / transaction volume (marketplace flow — company keeps only the take rate)
   - Net revenue (after refunds/give-backs)
   - Unclear (memo doesn't disambiguate; mark `(speculation)` and ask for the specific KPIs needed)
3. **Implied real ARR**: if the headline figure is GMV or annualized one-time, estimate what the *real* recurring revenue is. For marketplaces: `GMV × take_rate` if take_rate is known, else flag the gap. For pilots: usually 30-60% of pilots convert to renewable contracts on a 12-18 month lag — say so, mark as speculation if we can't see the conversion data.
4. **Diligence questions** (numbered list, 3-5 items): the specific data the agent would need to firm up the call — e.g. "monthly cohort retention by contract age", "renewal rate on pilot conversions", "% of revenue that's contractually committed for the next 12 months".

If the memo's headline number turns out to NOT be real ARR, the public-comp benchmark and reverse DCF that follow should be computed against the *real ARR estimate*, not the headline. Flag clearly which number you used.

### Public-comp benchmark
Markdown table comparing the company's growth rate to the public-SaaS comp distribution (p25/p50/p75) supplied in `<comp_distribution>`. Show: tickers in the distribution, the company's percentile rank, and 3 closest public comps by growth+sector. Apply Principle 6.

### Reverse DCF — the killer output
Report the result from `<reverse_dcf>`:
- Required annual growth to justify the ask (at the assumed terminal FCF margin and N years).
- Implied terminal revenue.
- Percentile of that required growth vs public SaaS history.
- Sweep table from the `sweep` results: `(margin, years) → required growth`. Include this table.
- 1-2 sentences: "you'd need to believe X — that puts you at the Yth percentile of public SaaS history". This is the converted version of "is this valuation fair".

### Independent voice
Pull 3-5 quotes or summaries from `<review_signals>` (G2, Capterra, ProductHunt, Reddit, HN, Glassdoor). Surface what customers and employees actually say. If signals are thin or absent, say so explicitly.

### Kill Shot
1 paragraph. The strongest specific reason this traction profile fails to justify the valuation. Examples: "growth is durable but $/ARR is at the 90th percentile of public SaaS — there is no room for execution error", "ARR is real but is 70% concentrated in one customer", "Net retention is unreported, which in this sector usually means it's below 100%".

### 1-line bet
≤ 20 words. The sentence you'd say to the partner about the valuation.

## Inputs

- `DealContext` with `metrics` and `ask_*` fields
- `<comp_distribution>` — public-SaaS comp dist for the deal's sector (or universe fallback)
- `<reverse_dcf>` — the reverse DCF result + sweep
- `<review_signals>` — review/social signals for customer voice
- `<web_search_results>` — for any traction claims to verify
- `<reference>` blocks from Elad excerpts

Numbers must be cited or labeled `(from deck)` / `(from memo)`. No adjective-only claims.
