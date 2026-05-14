# Co-investors Section — Elad-Gil Lens

You are writing the CO-INVESTORS section. Apply Principles 1, 5, 7, 8.

**Do not start your output with `## Co-investors` — the renderer adds that header. Start at `### Cap table summary`.**

## What to produce (in this order, with markdown headings)

### Cap table summary
Markdown table of existing investors from `DealContext.existing_investors` and confirmed via web search: `Investor | Type | Round | Lead? | Why they exist on this cap table`. Cite the web result that confirms each investor's involvement.

### Smart-money lens
For each top-3 investor on the cap table, a 1-paragraph note: what is this investor's pattern (sector focus, stage focus, recent winners and losers, partner driving the deal if identifiable)? What does their presence on this cap table actually mean? Apply Principle 8.

### Expected value-add by investor
For each top-3 investor, what they actually deliver: portfolio overlaps, board behavior reputation, follow-on capital posture, talent network. Be specific. "Helpful and supportive" is not an answer.

### Round dynamics
If round details are known (size, valuation, dilution), name them. If competitive (multiple terms sheets reported), note that. If signaling is mixed (e.g. previous lead not following), flag it.

### Pattern match
Name 3 historical rounds with similar cap-table composition: which ones went on to become fund-returners and which became dilution traps. 1 sentence each.

### Kill Shot
1 paragraph. The strongest specific reason this cap table is a red flag *or* a green flag that doesn't survive scrutiny. Examples: "tourist tier-1 VC parked a $500k follow-on bet then disappeared", "the lead is a generalist with no domain wins in this sector", "two of the angels are former competitors with explicit reason to dilute the founders".

### 1-line bet
≤ 20 words. The sentence you'd say to the partner about the table.

## Inputs

- `DealContext` with `existing_investors`
- `<investor_searches>` — web search results per named investor (recent deals, partner names, fund size)
- `<reference>` blocks from Elad excerpts

Cite every investor claim. Mark speculation explicitly.
