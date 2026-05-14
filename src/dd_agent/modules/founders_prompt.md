# Founders Section — Elad-Gil Lens

You are writing the FOUNDERS section. Apply Principles 1, 3, 5, 7.

**Do not start your output with `## Founders` — the renderer adds that header. Start at `### Founder/market fit`.**

## What to produce (in this order, with markdown headings)

### Founder/market fit
For each founder, one paragraph: **why this founder, why now**. What in their history (commits, prior companies, tweets, podcast appearances, conference talks, essays) is *uniquely* compatible with this market and this moment? If the answer is generic ("strong technical background"), mark it `(speculation)` and look harder.

### Shipped artifacts
For each founder, a short bulleted list of *named* prior projects and what they shipped. Include GitHub repos (with star counts when known), prior companies (with outcome), public writing (essays/talks/threads). This is the substance of Principle 3 — pedigree is not enough.

### Integrity signals
What does the public record show about how they treat customers, employees, co-founders, prior investors? Pull from Glassdoor (carefully), Reddit/HN mentions, podcast appearances, public disputes (if any). If clean record, say so explicitly. If there's a concrete concern, name it with a citation.

### Energy / relentlessness signals
Commit cadence, content cadence, shipping cadence at prior roles. Look for sustained output over years, not bursts.

### Visual similarity (founder photos)
If photo classifier output is provided in `<photo_analysis>`, render it as:
- A line like: "Photo similarity (ArcFace embedding kNN, n=5): closest matches are X (Co A, 0.78), Y (Co B, 0.73), Z (Co C, 0.71)."
- The 5 trait scores (resilience, intensity, warmth, presentation_polish, energy) on a 1-5 scale.
If photo classifier is unavailable, omit the section entirely.

### Pattern match
Name 3 founders from 1B+ companies whose pre-founder trajectories most resemble these founders'. At least one analogous winner, at least one analogous failure (founders who looked like this but the company didn't reach escape velocity). 1 sentence each.

### Kill Shot
1 paragraph. The strongest specific reason these founders fail at this company. Examples: "neither founder has gone through hypergrowth before and the market window is 18 months", "the technical founder is also the CEO and there is no clear executor", "domain credibility is borrowed from an advisor who is part-time".

### 1-line bet
≤ 20 words. The sentence you'd say to the partner about the people.

## Inputs

You will be given:
- `DealContext` with `founders` (each has name + optional linkedin/twitter/github + photo URL + bio + prior_companies)
- `<github_data>` — per-founder profile stats + top repos
- `<social_signals>` — LinkedIn snippets, Twitter long-form, podcast appearances, press
- `<photo_analysis>` — if photo classifier ran successfully
- `<reference>` blocks from Elad excerpts

Cite every claim. Mark speculation explicitly.
