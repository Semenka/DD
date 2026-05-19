"""Obsidian Web Clipper parsing — frontmatter, deck-link detection, fall-through."""

from dd_agent.ingestion import clipper


def test_parse_returns_none_for_plain_markdown():
    """A plain `.md` with no frontmatter and no clipping signals returns None."""
    text = "# Some heading\n\nJust a body paragraph.\n\nAnother paragraph."
    assert clipper.parse(text) is None


def test_parse_extracts_yaml_frontmatter():
    text = (
        "---\n"
        "source: https://techcrunch.com/2024/foo\n"
        "title: A Founder Story\n"
        "author: Jane Doe\n"
        "created: 2024-09-15\n"
        "---\n\n"
        "# Body heading\n\nSome content."
    )
    ctx = clipper.parse(text)
    assert ctx is not None
    assert ctx.source_url == "https://techcrunch.com/2024/foo"
    assert ctx.title == "A Founder Story"
    assert ctx.author == "Jane Doe"
    assert ctx.clipped_at == "2024-09-15"
    assert ctx.body_text.startswith("# Body heading")


def test_parse_accepts_alternate_url_field_names():
    """Frontmatter source can be under any of source/url/link/original."""
    for field_name in ("url", "link", "original", "permalink", "canonical"):
        text = f"---\n{field_name}: https://example.com/x\ntitle: t\n---\n\nbody"
        ctx = clipper.parse(text)
        assert ctx is not None
        assert ctx.source_url == "https://example.com/x", f"failed for field {field_name}"


def test_parse_detects_deck_url_in_body():
    text = (
        "---\nsource: https://news.example/post\n---\n\n"
        "The team shared a deck: https://pitch.com/v/acme-fundraise-abc123\n\n"
        "Worth a read."
    )
    ctx = clipper.parse(text)
    assert ctx is not None
    assert ctx.deck_url == "https://pitch.com/v/acme-fundraise-abc123"


def test_extract_deck_url_recognizes_each_host():
    """Every supported deck host should match."""
    cases = [
        "https://docsend.com/view/abc123def",
        "https://www.pitch.com/v/team-deck",
        "https://figma.com/proto/abc/Title",
        "https://docs.google.com/presentation/d/1Abc",
        "https://slideshare.net/Andrey/some-deck",
        "https://gamma.app/docs/Some-Pitch-abc123",
        "https://notion.so/MyDeck-abc123def",
    ]
    for url in cases:
        body = f"Check this out: {url} — interesting."
        result = clipper.extract_deck_url(body)
        assert result is not None, f"failed to extract from {url}"
        assert url.replace("www.", "") in result.replace("www.", ""), \
            f"got {result!r} for {url}"


def test_extract_deck_url_returns_none_when_no_deck():
    body = "Just regular links: https://wikipedia.org and https://github.com/x/y"
    assert clipper.extract_deck_url(body) is None


def test_parse_uses_clipping_signal_without_frontmatter():
    """Body with a deck link but no frontmatter still parses as a clipping
    because the deck-link signal triggers detection."""
    text = "Some intro paragraph.\n\nDeck: https://pitch.com/v/acme-x\n\nMore text."
    ctx = clipper.parse(text)
    assert ctx is not None
    assert ctx.deck_url == "https://pitch.com/v/acme-x"


def test_parse_extracts_embedded_image_urls():
    text = (
        "---\nsource: https://x.com/y\n---\n\n"
        "![hero](https://cdn.example.com/hero.jpg)\n\n"
        "![chart](https://cdn.example.com/chart.png)\n\n"
        "And a wiki ref: ![[local-image.png]]"
    )
    ctx = clipper.parse(text)
    assert ctx is not None
    assert "https://cdn.example.com/hero.jpg" in ctx.embedded_image_urls
    assert "https://cdn.example.com/chart.png" in ctx.embedded_image_urls
    assert "local-image.png" in ctx.embedded_image_wiki_refs


def test_parse_falls_through_body_link_for_source_url():
    """When frontmatter has no source field, first body link becomes source."""
    text = "---\ntitle: Untitled\n---\n\nSee [the announcement](https://example.com/post) for details."
    ctx = clipper.parse(text)
    assert ctx is not None
    assert ctx.source_url == "https://example.com/post"


def test_parse_handles_quoted_frontmatter_values():
    """Values with quotes — both ' and " — should be stripped."""
    text = '---\nsource: "https://example.com/quoted"\ntitle: \'My Title\'\n---\n\nbody'
    ctx = clipper.parse(text)
    assert ctx is not None
    assert ctx.source_url == "https://example.com/quoted"
    assert ctx.title == "My Title"
