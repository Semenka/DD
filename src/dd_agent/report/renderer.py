"""Render the final DD report.

Markdown is the source of truth. HTML is a simple styled wrapper around the
markdown (converted via a tiny Markdown→HTML pass — no external dep — and
embedded in a self-contained HTML document with a print stylesheet).
"""

from __future__ import annotations

import html as html_escape
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..citations import CitationBook
from ..context import DealContext

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_markdown(
    *,
    ctx: DealContext,
    synthesis: str,
    market: str,
    founders: str,
    traction: str,
    coinvestors: str,
    citations: CitationBook,
    extras: dict | None = None,
    bessemer_memo: str | None = None,
    charts: dict | None = None,
) -> str:
    """Render the deal report to markdown.

    `charts` (v8): optional dict containing pre-rendered chart strings keyed
    by where they should appear in the template:
      - `market_comp_ruler`: inline SVG percentile ruler
      - `dcf_heatmap`: inline `<img>` (matplotlib PNG base64) for the
                      reverse-DCF heatmap
      - `funding_timeline`: inline `<img>` for the funding-rounds timeline
      - `trait_bars_by_founder`: dict[founder_name -> SVG bars]
    Missing keys cause the corresponding section to be skipped via Jinja
    `{% if charts.x %}` guards in the template."""
    env = _env()
    template = env.get_template("report.md.j2")
    return template.render(
        ctx=ctx,
        synthesis=synthesis,
        bessemer_memo=bessemer_memo,
        market=market,
        founders=founders,
        traction=traction,
        coinvestors=coinvestors,
        citations_md=citations.render_markdown(),
        extras=extras or {},
        charts=charts or {},
    )


def render_html(
    *, markdown_text: str, deal_context: DealContext,
    photo_b64_by_path: dict[str, str] | None = None,
) -> str:
    """Render markdown to a styled HTML document.

    `photo_b64_by_path` maps absolute photo paths to base64-encoded image
    bytes. The markdown→HTML conversion rewrites `![alt](path)` references
    to `<img src="data:image/jpeg;base64,...">` so the HTML is self-
    contained when emailed or sent over Telegram."""
    if photo_b64_by_path:
        # Inline-replace each known path in the markdown before HTMLification
        for path, b64 in photo_b64_by_path.items():
            data_url = f"data:image/jpeg;base64,{b64}"
            # Escape parens in the path so they don't break the regex
            import re as _re
            markdown_text = _re.sub(
                r"!\[([^\]]*)\]\(" + _re.escape(path) + r"\)",
                lambda m, du=data_url: f"![{m.group(1)}]({du})",
                markdown_text,
            )
    env = _env()
    template = env.get_template("report.html.j2")
    return template.render(
        deal_context=deal_context,
        body_html=_markdown_to_html(markdown_text),
    )


def render_pdf(*, html: str, out_path: str | None = None) -> bytes:
    """Render the HTML report to a PDF. Returns the PDF bytes; if `out_path` is
    provided, also writes to disk. Uses weasyprint (no external binary needed)."""
    try:
        from weasyprint import HTML
    except ImportError as exc:  # pragma: no cover - covered by install path
        raise RuntimeError(
            "weasyprint not installed. `pip install weasyprint` "
            "(macOS may also need `brew install pango cairo`)."
        ) from exc
    pdf_bytes = HTML(string=html).write_pdf()
    if out_path:
        with open(out_path, "wb") as f:
            f.write(pdf_bytes)
    return pdf_bytes


# --- minimal markdown → HTML (no external dep) -------------------------------


_RAW_HTML_BLOCK_TAGS = (
    "<details", "</details>",
    "<summary", "</summary>",
    "<svg", "</svg>",
    "<figure", "</figure>",
    "<figcaption", "</figcaption>",
)


def _is_raw_html_block_line(line: str) -> bool:
    """A line is treated as raw HTML if it begins with one of the whitelisted
    block-level tags. Used to gate the v8 appendix / chart embeds — we want
    `<details>` and `<svg>` to survive the markdown pass intact."""
    stripped = line.lstrip()
    return any(stripped.startswith(tag) for tag in _RAW_HTML_BLOCK_TAGS)


def _markdown_to_html(text: str) -> str:
    """Tiny markdown subset: headings, bold/italic, links, code, tables, lists,
    blockquotes, refs `[n]`. Sufficient for a clean DD report; not a general
    markdown engine.

    v8: also passes through raw HTML for a small whitelist of block tags
    (<details>, <summary>, <svg>, <figure>) so the collapsible appendix +
    inline SVG charts render correctly. A line starting with one of these
    tags is emitted verbatim; multi-line SVG blocks stay in raw mode until
    their matching close tag is seen."""
    lines = text.splitlines()
    out: list[str] = []
    in_code = False
    in_table = False
    table_rows: list[list[str]] = []
    in_list = False
    list_type = None
    in_raw_svg = False  # multi-line SVG content pass-through

    def flush_table():
        nonlocal in_table, table_rows
        if not table_rows:
            return
        head = table_rows[0]
        body = [r for r in table_rows[2:] if r] if len(table_rows) > 1 else []
        out.append("<table>")
        out.append("<thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead>")
        if body:
            out.append("<tbody>")
            for r in body:
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
            out.append("</tbody>")
        out.append("</table>")
        table_rows = []
        in_table = False

    def flush_list():
        nonlocal in_list, list_type
        if in_list:
            out.append(f"</{list_type}>")
            in_list = False
            list_type = None

    for raw in lines:
        line = raw.rstrip()

        # --- raw HTML pass-through (v8 appendix + inline SVG charts) ---
        # Once inside a multi-line <svg>, every line passes through verbatim
        # until we see </svg>.
        if in_raw_svg:
            out.append(raw)
            if "</svg>" in line:
                in_raw_svg = False
            continue
        if _is_raw_html_block_line(line):
            flush_list()
            flush_table()
            out.append(raw)
            # If the SVG block didn't close on the same line, enter raw mode.
            if line.lstrip().startswith("<svg") and "</svg>" not in line:
                in_raw_svg = True
            continue

        if line.startswith("```"):
            flush_list()
            flush_table()
            if in_code:
                out.append("</code></pre>")
            else:
                out.append("<pre><code>")
            in_code = not in_code
            continue
        if in_code:
            out.append(html_escape.escape(line))
            continue

        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            in_table = True
            table_rows.append(cells)
            continue
        if in_table:
            flush_table()

        if line.startswith("###### "):
            flush_list(); out.append(f"<h6>{_inline(line[7:])}</h6>"); continue
        if line.startswith("##### "):
            flush_list(); out.append(f"<h5>{_inline(line[6:])}</h5>"); continue
        if line.startswith("#### "):
            flush_list(); out.append(f"<h4>{_inline(line[5:])}</h4>"); continue
        if line.startswith("### "):
            flush_list(); out.append(f"<h3>{_inline(line[4:])}</h3>"); continue
        if line.startswith("## "):
            flush_list(); out.append(f"<h2>{_inline(line[3:])}</h2>"); continue
        if line.startswith("# "):
            flush_list(); out.append(f"<h1>{_inline(line[2:])}</h1>"); continue

        if line.startswith("> "):
            flush_list()
            out.append(f"<blockquote>{_inline(line[2:])}</blockquote>")
            continue

        if re.match(r"^-\s+", line) or re.match(r"^\*\s+", line):
            if not in_list or list_type != "ul":
                flush_list(); out.append("<ul>"); in_list = True; list_type = "ul"
            out.append(f"<li>{_inline(line[2:])}</li>")
            continue
        if re.match(r"^\d+\.\s+", line):
            if not in_list or list_type != "ol":
                flush_list(); out.append("<ol>"); in_list = True; list_type = "ol"
            stripped = re.sub(r"^\d+\.\s+", "", line)
            out.append(f"<li>{_inline(stripped)}</li>")
            continue

        flush_list()
        if line.strip() == "":
            out.append("")
        elif line.startswith("---"):
            out.append("<hr>")
        else:
            # Reference rows in the bibliography start with `[N] ...` — give
            # the paragraph an id so inline citations can scroll-to it.
            m = re.match(r"^\[(\d+)\]\s", line)
            if m:
                out.append(f'<p id="ref-{m.group(1)}">{_inline(line)}</p>')
            else:
                out.append(f"<p>{_inline(line)}</p>")

    flush_table()
    flush_list()
    if in_code:
        out.append("</code></pre>")
    return "\n".join(out)


def _inline(text: str) -> str:
    text = html_escape.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    # Images: `![alt](src)` — render as <img>. Match BEFORE the link rule below
    # since markdown image syntax is identical to a link prefixed with '!'.
    text = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        r'<img class="founder-photo" src="\2" alt="\1" loading="lazy">',
        text,
    )
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<a href="\2" target="_blank" rel="noopener">\1</a>',
        text,
    )
    # Inline citation markers like [3] become anchor links to #ref-3 in the
    # References section, so the reader can click them and land on the source.
    text = re.sub(
        r"\[(\d+)\]",
        r'<sup class="ref"><a href="#ref-\1">[\1]</a></sup>',
        text,
    )
    return text
