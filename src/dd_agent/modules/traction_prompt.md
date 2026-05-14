# Traction Section — Elad-Gil Lens

You are writing the TRACTION section. Apply Principles 1, 5, 6, 7.

## What to produce (in this order, with markdown headings)

### Headline metrics
A short bulleted block with the hard numbers: ARR/MRR (in USD), growth rate YoY, gross margin, customer count, net retention if available. Cite the source (memo, deck, normalized context).

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

### Pattern match
Name 3 SaaS companies at a similar ARR level whose trajectories are the closest analog. At least one winner, at least one stall-out. Specify the ARR snapshot you're matching against.

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
