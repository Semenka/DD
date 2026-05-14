# DD — Elad-Gil-Style Due Diligence Agent

Opinionated, evidence-first due diligence for private investment deals. Modeled after Elad Gil's investing style: pattern-match against history, ask the unfair question, prize inflection over TAM, evidence over opinion.

Powered by **OpenAI GPT-5.5** via the **`codex` CLI** for all subagent reasoning and synthesis. No API key needed — codex uses your existing ChatGPT login.

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
# 1. codex CLI (one-time): authenticates with your ChatGPT account
npm install -g @openai/codex
codex login

# 2. Python deps
git clone https://github.com/semenka/DD.git
cd DD
uv venv && source .venv/bin/activate
uv pip install -e ".[photo]"
cp .env.example .env       # optional: TAVILY_API_KEY for higher-quality web search
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

### Register with OpenClaw (one command)

```bash
openclaw mcp set dd-agent '{
  "command": "/absolute/path/to/DD/.venv/bin/dd-agent",
  "args": ["serve"],
  "env": {
    "DD_MODEL": "gpt-5.5",
    "DD_MODEL_FAST": "gpt-5.5-mini",
    "DD_DATA_DIR": "/absolute/path/to/DD/data",
    "DD_DB_PATH": "/absolute/path/to/DD/data/deals.db",
    "PATH": "/path/to/node/bin:/opt/homebrew/bin:/usr/bin:/bin"
  }
}'
openclaw mcp list      # confirms "dd-agent" is registered
openclaw mcp show dd-agent   # prints the stored config
```

The `PATH` env is necessary because OpenClaw spawns dd-agent as a fresh subprocess; without `codex` on the inherited PATH, all LLM calls inside dd-agent will fail.

### Register with Claude Code / Cursor / Codex (MCP-compatible clients)

Add to your client's MCP config (`~/.claude.json`, `~/.codex/config.toml`, etc.):

```json
{
  "mcpServers": {
    "dd-agent": {
      "command": "/absolute/path/to/DD/.venv/bin/dd-agent",
      "args": ["serve"],
      "env": { "DD_MODEL": "gpt-5.5", "PATH": "/path/to/node/bin:/usr/bin:/bin" }
    }
  }
}
```

### Submitting a deal through OpenClaw

From the OpenClaw CLI (or chat):

```bash
openclaw agent --agent main --message '
Use the dd-agent MCP server. Call submit_deal with:
  - memo_text: read the file /Users/me/deal_memo.md
  - deck_path: /Users/me/deck.pdf       (absolute path; null if no deck)
  - company_url: https://example.com    (null if no website)
  - founder_names: ["Founder One", "Founder Two"]  (optional)
Return the deal_id.
'
```

Then poll status and fetch the report:

```bash
openclaw agent --agent main --message '
Call dd-agent get_report_status with deal_id="...". When status is "done",
call get_report and print the markdown.
'
```

A verified end-to-end run against `examples/sample_deal/memo.md` is committed at [`examples/sample_deal/linear_report.md`](examples/sample_deal/linear_report.md) — submitted from OpenClaw's `main` agent (`openai-codex/gpt-5.5`), executed in ~5 min, produces all required artifacts (Synthesis, Beliefs Required, Kill Shot, 1-line bet, Recommendation, Reverse DCF with sweep table, public-comp benchmark).

### End-to-end verification without OpenClaw

`scripts/openclaw_pipeline_test.py` reads OpenClaw's MCP config and drives dd-agent through exactly the same MCP lifecycle OpenClaw uses (spawn → initialize → tools/list → submit_deal → poll → get_report). Use it to verify the pipeline before sending real deals through OpenClaw:

```bash
.venv/bin/python scripts/openclaw_pipeline_test.py \
  --memo examples/sample_deal/memo.md \
  --url https://linear.app
# → report.md, report.html
```

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

Each subagent runs in its own scoped `codex exec` subprocess against GPT-5.5. No API key — codex uses your ChatGPT login. Each pulls retrieval snippets from Elad's blog / High Growth Handbook / podcast transcripts via BM25 and grounds claims in citations. Configure model via `DD_MODEL` (default `gpt-5.5`) and a faster model for ingestion/photo trait scoring via `DD_MODEL_FAST` (default `gpt-5.5-mini`). Override the codex binary path via `DD_CODEX_BIN`.

## License

MIT.
