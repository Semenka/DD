# DD — Elad-Gil-Style Due Diligence Agent

Opinionated, evidence-first due diligence for private investment deals. Modeled after Elad Gil's investing style: pattern-match against history, ask the unfair question, prize inflection over TAM, evidence over opinion.

Powered by **OpenAI GPT-5.5** via the **`codex` CLI** for all subagent reasoning and synthesis. No API key needed — codex uses your existing ChatGPT login.

Exposed as an **MCP server** so OpenClaw, Claude Code, Cursor, Codex, or any MCP-capable agent can submit deals and pull reports.

## What it produces

A single report (markdown + HTML + **PDF**, default HTML with active citation links) with three layers of analysis:

### Layer 1: Bessemer-style Investment Memo (long-form)

A 1-2 page narrative memo modeled on the [Bessemer Venture Partners memos](https://www.bvp.com/memos): third-person present-tense prose interleaved with first-person partner asides. Required sections in order:
**Investment Thesis** • **Company** • **Market** • **Why Now** • **Team** • **Traction** • **Outcomes Analysis** (if-we're-right + if-we're-wrong scenarios) • **Data Room** (5-7 diligence questions) • **Recommendation** with conviction level.

### Layer 2: Elad-Gil-style four-section analyst report

1. **Market** — current/projected size, competitor matrix, inflection thesis
2. **Founders** — track record, integrity, energy, founder/market fit, photo similarity vs founder-led S&P 500 / Nasdaq 100 / YC top-100 / 1B+ private companies (with cohort breakdown of nearest matches)
3. **Traction** — **revenue-quality classifier** (real ARR vs annualized pilots vs GMV vs one-time hardware), reverse DCF + public-comp percentiles, ARR growth vs SaaS comps, independent-voice (G2/HN/Reddit) signal
4. **Co-investors** — round-by-round private funding history (round, date, amount, post-money, lead, participants) + live notice.co secondary-market snapshot + cap-table breakdown vs top VCs/super-angels + per-investor value-add

### Layer 3: Synthesis page

**Kill Shot** • **1-line bet** (≤20 words) • **Beliefs Required to Invest** — the 3-5 propositions that must each be true for this to be a fund-returner.

## Inputs

dd-agent accepts any of:
- **PDF memo + PDF deck** — the original `submit_deal` signature
- **Plain text memo** via `memo_text`
- **Obsidian Web Clipper `.md`** — YAML frontmatter is parsed for source URL + title + author; any embedded pitch-deck link (Pitch.com, Google Slides, DocSend, Figma, Notion, Gamma, Slideshare) is screenshotted via headless Chromium and OCR'd via Gemini Vision into the deck text
- **Company URL** alone — fetches and parses the company website

## Search backends

dd-agent uses a 5-tier cascade — configure any backend and the agent picks the first that's available. Override the order with `DD_SEARCH_PREFERRED=gemini,duckduckgo` (any backend you list comes first; the rest stay as fallbacks).

| Tier | Backend | Env / setup | Notes |
|------|---------|------------|-------|
| 1 | **`openclaw infer web search`** *(default)* | OpenClaw installed + `DD_OPENCLAW_SEARCH_PROVIDER` (default: `gemini`) | Reuses OpenClaw's configured providers — no extra key beyond what OpenClaw already has. Returns rich synthesized content + citations in one call. |
| 2 | **Direct Perplexity Sonar** | `PERPLEXITY_API_KEY` | Direct Perplexity API. Search + grounded synthesis. |
| 3 | **Direct Gemini grounding** | `GEMINI_API_KEY` + `DD_GEMINI_MODEL` (default `gemini-2.5-flash`) | Direct Gemini API with the Google Search tool. |
| 4 | Tavily | `TAVILY_API_KEY` | URL list only. |
| 5 | DuckDuckGo HTML | (none) | Last-resort, rate-limited. |

**Bypass Perplexity entirely:** set `DD_OPENCLAW_SEARCH_PROVIDER=gemini` (default) and don't set `PERPLEXITY_API_KEY`. The agent will route every search through OpenClaw → Gemini.

**Gemini model selector** (`DD_GEMINI_MODEL`):

| Value | Why |
|-------|-----|
| `gemini-2.5-flash` *(default)* | Fast, cheap, grounding works. |
| `gemini-3.5-flash` | Newer, slightly better synthesis quality. Same grounding interface. |
| `gemini-3-pro-preview` / `gemini-3.1-pro-preview` | Highest quality, slower, higher cost. |

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

For Obsidian-clipping + pitch-deck screenshot OCR, also install Playwright's Chromium:

```bash
.venv/bin/playwright install chromium
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

**Photo corpus realistic ceiling (as of v4):** the agent's Perplexity-generated 276-founder seed list resolves to ~126 successful InsightFace embeddings via the four-tier photo cascade (Wikipedia → company team page → web search → grounded LLM URL lookup). The remaining 150 founders are mostly Indian / Southeast Asian unicorn founders without English Wikipedia presence and whose company sites' `/team` and `/leadership` paths either 404 or don't expose alt-text matching their Romanized names. The architecture supports the full 500 — extending coverage requires either Crunchbase/PitchBook API access or hand-curated photo URLs for those tail founders.

The build supports incremental runs: just re-run `python scripts/build_unicorn_corpus.py` and only founders not yet in the parquet are attempted. Checkpoints write every 20 newly-embedded founders so crashes don't lose progress.

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

### Telegram → DD agent → Telegram PDF (the production flow)

The intended workflow is: drop the deal memo and pitch deck PDFs into a Telegram chat with OpenClaw, ask Andrey-style ("analyze this deal" or similar), and receive a PDF back on the same channel a few minutes later.

Verified end-to-end on 2026-05-15:
- **PDF intake**: `openclaw agent --agent main` called `dd-agent__submit_deal` with `memo_path=memo.pdf`, deal created, company "Linear" extracted correctly.
- **Report rendering**: 8-page, 74 KB PDF written to `data/reports/<deal_id>.pdf` via weasyprint.
- **Telegram delivery**: `openclaw message` tool with `filePath=…linear_report.pdf` returned `ok=true messageId=10 chatId=148594943` — the PDF was delivered to the user's Telegram chat as a document attachment.

This is wired through OpenClaw's `main` agent. The agent's instructions live in `~/.openclaw/workspace/TOOLS.md` under the "DD agent" heading and tell `main` to:

1. Save Telegram PDF attachments to disk
2. Call `dd-agent submit_deal` with `memo_path` and `deck_path`
3. Acknowledge the user with the `deal_id`
4. Poll `get_report_status` every 30s
5. Call `get_report` with `include_pdf_base64=true`
6. Reply on the same channel with the PDF attached, filename `{Company}-DD-{deal_id}.pdf`, and the report's 1-line bet as message text

The dd-agent MCP server has been extended for this:

- `submit_deal` now accepts `memo_path` (PDF/text/markdown file path) in addition to `memo_text` — so OpenClaw can hand off whatever the user attached.
- `get_report` accepts `include_pdf_base64=true` to return the report PDF inline (base64-encoded), and `pdf_path` (absolute path on the host).

### Submitting a deal manually from the OpenClaw CLI

```bash
openclaw agent --agent main --message '
Use dd-agent. Call submit_deal with:
  - memo_path: "/Users/me/deal_memo.pdf"
  - deck_path: "/Users/me/deck.pdf"
  - company_url: "https://example.com"
Return the deal_id.
'
```

Then poll + retrieve:

```bash
openclaw agent --agent main --message '
Call dd-agent get_report_status with deal_id="...". When done, call
get_report with include_pdf_base64=true. Save the PDF to /tmp/report.pdf.
'
```

A verified end-to-end run against `examples/sample_deal/memo.pdf` is committed at [`examples/sample_deal/linear_report.md`](examples/sample_deal/linear_report.md) (and the corresponding `.pdf`) — submitted from OpenClaw's `main` agent (`openai-codex/gpt-5.5`), executed in ~6 min, produces all required artifacts (Synthesis, Beliefs Required, Kill Shot, 1-line bet, Recommendation, Reverse DCF with sweep table, public-comp benchmark, **round-by-round funding history**, **notice.co snapshot**).

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
