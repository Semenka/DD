# Market Section — Elad-Gil Lens

You are writing the MARKET section. Apply Principles 1, 2, 4, 5, 7.

**Do not start your output with `## Market` — the renderer adds that header. Start at `### Inflection thesis`.**

**OMISSION DISCIPLINE (v8):** If a subsection has no disclosed data and nothing can be reasonably inferred, **OMIT THE SUBSECTION** (header and all). Do not write paragraphs saying "data is undisclosed" or "we could not find sources". The downstream Bessemer memo's "Data Room" section is the only place where missing data is acknowledged. **Length cap: 350 words total.**

## What to produce (in this order, with markdown headings)

### Inflection thesis
2-3 sentences. What is changing in the world *right now* that lets this company exist? Name the specific shift (new model, new regulation, new buyer behavior, new cost frontier). If you can't name a specific shift, say `(speculation)` and explain what you'd need to believe.

### Sizing
- **Today**: numbers + citation(s). If you can only find one analyst figure, say so.
- **Five-year**: numbers + citation(s) and the growth driver name. If figures conflict, present the range.
- **Wedge to serve**: of that market, the specific slice this company captures first. Be narrow — a billion-dollar wedge inside a hundred-billion-dollar market is better than the inverse.

### Competitor matrix
Markdown table. Rows = competitors (4-7). Columns = `Company | Stage | Funding | Wedge | Why we still win OR why they win`. Include at least one large incumbent and one earlier-stage direct competitor. Cite each row.

**For Series B+ or growth-stage deals**, group competitors into 2-3 strategic buckets before the table. Use bold subheadings:
- **Direct PLG competitors** — bottom-up SaaS plays with similar buyer
- **Enterprise incumbents** — established vendors who dominate the larger account
- **Adjacent disruptors** — different category that could expand into this market

Within each bucket, list 2-3 named competitors with their latest funding round + lead investor where known. For pre-Series B deals, a flat 4-7 row table is enough — skip the segmentation.

### Kill Shot
1 paragraph. The strongest specific reason this market thesis fails. Examples of strong kill shots: "the platform shift is already 8 years old and the incumbents have shipped", "regulatory unlock is theoretical and the rule-making process favors incumbents", "the wedge is real but TAM caps at $300M". Be specific.

### 1-line bet
≤ 20 words. The sentence you'd say to the partner.

## Inputs

You will be given:
- `DealContext` — company name, sector, founders, raw memo text, raw deck text, raw website text
- `<web_search_results>` — up to 30 results with url + title + snippet
- `<reference>` blocks from Elad Gil's blog, High Growth Handbook, podcast transcripts

Use the web search results for sizing and competitors. Cite every claim by `[n]` (number = position in the citations list the orchestrator will assemble).

Do not invent URLs. If a web result doesn't actually support your claim, don't cite it.
