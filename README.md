# DD — Elad-Gil-Style Due Diligence Agent

Opinionated, evidence-first due diligence for private investment deals. Modeled after Elad Gil's investing style: pattern-match against history, ask the unfair question, prize inflection over TAM, evidence over opinion.

Powered by **OpenAI GPT-5.5** for all subagent reasoning and synthesis.

Exposed as an **MCP server** so OpenClaw, Claude Code, Cursor, Codex, or any MCP-capable agent can submit deals and pull reports.

## What it produces

A single report with four sections plus a synthesis page:

1. **Market** — current/projected size, competitor matrix, inflection thesis
2. **Founders** — track record, integrity signals, energy, founder/market fit, last shipped projects, photo similarity vs 1B+ company founders
3. **Traction** — reverse DCF + public-comp percentiles, ARR growth vs SaaS comps, independent-voice (G2/HN/Reddit) signal
4. **Co-investors** — cap-table breakdown vs top VCs and super-angels, per-investor value-add hypothesis

Synthesis page: **Kill Shot**, **1-line bet** (≤20 words), and **Beliefs Required to Invest** — the 3-5 propositions that must each be true for this to be a fund-returner.

## Install

Requires Python 3.11+. Uses [uv](https://github.com/astral-sh/uv) (recommended) or pip.

```bash
git clone https://github.com/semenka/DD.git
cd DD
uv venv && source .venv/bin/activate
uv pip install -e ".[photo]"
cp .env.example .env       # fill in OPENAI_API_KEY, TAVILY_API_KEY
```

Photo classifier and podcast-transcription deps are optional extras:

```bash
uv pip install -e ".[photo,podcasts,dev]"
```

## One-time corpus builds

```bash
python scripts/build_elad_corpus.py      # scrape blog.eladgil.com + transcripts → BM25 index
python scripts/build_unicorn_corpus.py   # founder photos → InsightFace embeddings + trait scores
```

Both ship with seeded fallback data so the agent works before you run them.

## Run as MCP server

```bash
dd-agent serve              # stdio MCP — connect from OpenClaw / Claude Code / Cursor
```

### OpenClaw / Claude Code config

Add to your MCP config (`~/.claude.json` for Claude Code, OpenClaw equivalent):

```json
{
  "mcpServers": {
    "dd-agent": {
      "command": "dd-agent",
      "args": ["serve"],
      "env": { "OPENAI_API_KEY": "sk-...", "DD_MODEL": "gpt-5.5" }
    }
  }
}
```

Then in OpenClaw / Claude Code:

> "Use the dd-agent: submit this deal — memo at ./memo.md, deck at ./deck.pdf, company URL https://example.com"

## MCP tools

| Tool | Args | Returns |
|------|------|---------|
| `submit_deal` | `memo_text?`, `deck_path?`, `company_url?`, `founder_names?[]` | `{deal_id, status}` |
| `get_report_status` | `deal_id` | `{phase, progress_pct, eta_seconds}` |
| `get_report` | `deal_id` | `{markdown, html, citations}` |
| `list_deals` | — | `[{deal_id, company, status, created_at}]` |

`submit_deal` returns immediately; the actual DD runs in the background. Poll `get_report_status` until `phase == "done"`, then call `get_report`.

## CLI for testing without MCP

```bash
python scripts/smoke_test.py --memo examples/sample_deal/memo.md --deck examples/sample_deal/deck.pdf
```

Writes `report.md` and `report.html` to the current directory.

## Architecture

```
ingestion → DealContext → orchestrator
                              │
                              ├── market subagent      ┐
                              ├── founders subagent    │  4× asyncio.gather
                              ├── traction subagent    │
                              └── coinvestors subagent ┘
                              │
                              └→ synthesis call → report renderer
```

Each subagent runs in its own scoped OpenAI GPT-5.5 chat completion. Each pulls retrieval snippets from Elad's blog / High Growth Handbook / podcast transcripts via BM25 and grounds claims in citations. Configure model via `DD_MODEL` (default `gpt-5.5`) and a faster model for ingestion/photo trait scoring via `DD_MODEL_FAST` (default `gpt-5.5-mini`).

## License

MIT.
