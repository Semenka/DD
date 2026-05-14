"""Markdown → HTML conversion."""

from dd_agent.context import DealContext
from dd_agent.report.renderer import _markdown_to_html, render_markdown
from dd_agent.citations import CitationBook, Citation


def test_markdown_to_html_basics():
    md = "# Title\n\nA paragraph with **bold** and *italic* and `code`.\n"
    html = _markdown_to_html(md)
    assert "<h1>Title</h1>" in html
    assert "<strong>bold</strong>" in html
    assert "<em>italic</em>" in html
    assert "<code>code</code>" in html


def test_markdown_table():
    md = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n"
    html = _markdown_to_html(md)
    assert "<table>" in html
    assert "<thead>" in html and "<tbody>" in html
    assert "<th>A</th>" in html
    assert "<td>1</td>" in html


def test_markdown_refs_become_sup():
    md = "Claim foo [1] and bar [2]."
    html = _markdown_to_html(md)
    assert '<sup class="ref">[1]</sup>' in html
    assert '<sup class="ref">[2]</sup>' in html


def test_render_markdown_smoke():
    ctx = DealContext(deal_id="abc", company_name="Test Co", sector="ai_devtools")
    book = CitationBook()
    book.add(Citation(key="https://x.com/1", title="A source", url="https://x.com/1"))
    md = render_markdown(
        ctx=ctx,
        synthesis="### Exec summary\n\nTest synthesis.",
        market="Market content.",
        founders="Founder content.",
        traction="Traction content.",
        coinvestors="Co-investor content.",
        citations=book,
    )
    assert "Test Co" in md
    assert "Market content." in md
    assert "Traction content." in md
    assert "Co-investor content." in md
    assert "## References" in md
