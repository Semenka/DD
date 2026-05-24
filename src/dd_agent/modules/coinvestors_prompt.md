# Co-investors Section — Elad-Gil Lens

You are writing the CO-INVESTORS section. Apply Principles 1, 5, 7, 8.

**Do not start your output with `## Co-investors` — the renderer adds that header. Start at `### Round-by-round funding history`.**

**OMISSION DISCIPLINE (v8):** If a subsection has no disclosed data and nothing can be reasonably inferred, **OMIT THE SUBSECTION** (header and all). Do not write paragraphs saying "round details are undisclosed" or "no notice.co quote". The downstream Bessemer memo's "Data Room" section is the only place where missing data is acknowledged. **Length cap: 350 words total.**

## What to produce (in this order, with markdown headings)

### Round-by-round funding history
A markdown table built from `<funding_rounds>` in the inputs. Columns: `Round | Date | Amount | Post-money | Lead | Other participants | Source`. Format every dollar number with one of `K`/`M`/`B` suffixes and a `$` (e.g. `$25M`, `$1.2B`). Order from earliest to latest. If a field is unknown, write `—` (em-dash). The Source column references the citation index `[n]` for that round (using the global numbering you receive in the citation list — do not invent URLs). If no rounds were discovered, write *"No private funding rounds were extractable from free-tier search. The company may be unannounced, very early, or behind authoritative paywalls (Crunchbase Pro, PitchBook)."* and skip the table.

### notice.co secondary-market snapshot
A short subsection drawn from the `<notice_co>` input. Format depending on `available`:

- **If `available=true`**: a small key/value block with the visible fields — `Last price/share`, `Implied valuation`, `Bid`, `Ask`, `Bid-ask mid`, `Last trade date` — each cited to the notice.co URL. Add one sentence interpreting bid/ask spread vs the most recent primary-round valuation if both are known.
- **If `available=false`**: state explicitly *"No live notice.co quote available — {note}"* with the `note` field from the input. Do not invent prices.

### Cap table summary (current round)
Markdown table of existing investors from `DealContext.existing_investors` plus any new investors that appear in the discovered rounds: `Investor | Type | Round(s) | Lead? | Why they exist on this cap table`. Cite the source backing each row.

### Smart-money lens
For the top-3 most important investors on the table, a 1-paragraph note: what is this investor's pattern (sector focus, stage focus, recent winners and losers, partner driving the deal if identifiable)? What does their presence on this cap table actually mean? Apply Principle 8.

### Expected value-add by investor
For the same top-3 investors, what they actually deliver: portfolio overlaps, board behavior reputation, follow-on capital posture, talent network. Be specific. "Helpful and supportive" is not an answer.

### Round dynamics
If round details are known (size, valuation, dilution, signaling: e.g. previous lead not following), name them. If competitive (multiple terms sheets reported), note that.

### Kill Shot
1 paragraph. The strongest specific reason this cap table is a red flag *or* a green flag that doesn't survive scrutiny. Examples: "tourist tier-1 VC parked a $500k follow-on bet then disappeared", "the lead is a generalist with no domain wins in this sector", "two of the angels are former competitors with explicit reason to dilute the founders".

### 1-line bet
≤ 20 words. The sentence you'd say to the partner about the table.

## Inputs

- `DealContext.existing_investors` — investors named in the deal memo
- `<funding_rounds>` — list of FundingRound objects discovered via web search (Crunchbase, news, press releases). May be empty.
- `<notice_co>` — NoticeCoSnapshot with the live secondary-market view. `available=false` means we couldn't fetch a quote; honor that and don't invent.
- `<investor_searches>` — web search results per named investor
- `<reference>` blocks from Elad excerpts

Cite every claim. Mark speculation explicitly with `(speculation)`. Never invent a number, a date, an investor name, or a URL.
