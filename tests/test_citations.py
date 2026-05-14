"""Citation dedup + global numbering."""

from dd_agent.citations import Citation, CitationBook


def test_dedup_by_key():
    b = CitationBook()
    n1 = b.add(Citation(key="https://a.com", title="A"))
    n2 = b.add(Citation(key="https://a.com", title="A again"))
    n3 = b.add(Citation(key="https://b.com", title="B"))
    assert n1 == 1
    assert n2 == 1  # same key → same number
    assert n3 == 2
    assert len(b.citations) == 2


def test_render_markdown_includes_all():
    b = CitationBook()
    b.add(Citation(key="k1", title="Title 1", url="https://x.com/1", snippet="hello"))
    b.add(Citation(key="k2", title="Title 2", url="https://x.com/2"))
    md = b.render_markdown()
    assert "[1]" in md and "[2]" in md
    assert "Title 1" in md and "Title 2" in md
    assert "## References" in md


def test_empty_book_renders_empty_string():
    assert CitationBook().render_markdown() == ""


def test_to_list_indices():
    b = CitationBook()
    b.add(Citation(key="k1", title="A"))
    b.add(Citation(key="k2", title="B"))
    data = b.to_list()
    assert data[0]["n"] == 1 and data[1]["n"] == 2
