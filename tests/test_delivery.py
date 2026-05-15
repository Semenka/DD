"""Delivery layer — config parsing, attachment selection, one-line-bet extraction."""

import pytest

from dd_agent.delivery import DeliverTo, _pick_attachment, extract_one_line_bet


def test_deliver_to_from_dict_minimal():
    d = DeliverTo.from_dict({"channel": "telegram", "target": "148594943"})
    assert d is not None
    assert d.channel == "telegram"
    assert d.target == "148594943"
    assert d.account == "default"
    assert d.format == "html"
    assert d.summary_line is None


def test_deliver_to_from_dict_full():
    d = DeliverTo.from_dict({
        "channel": "telegram",
        "account": "cosmo",
        "target": "148594943",
        "format": "pdf",
        "summary_line": "test bet",
    })
    assert d.account == "cosmo"
    assert d.format == "pdf"
    assert d.summary_line == "test bet"


def test_deliver_to_from_dict_none_when_missing_required():
    assert DeliverTo.from_dict(None) is None
    assert DeliverTo.from_dict({}) is None
    assert DeliverTo.from_dict({"channel": "telegram"}) is None
    assert DeliverTo.from_dict({"target": "123"}) is None


def test_pick_attachment_honors_requested_format():
    assert _pick_attachment("html", "a.html", "b.pdf", "c.md") == "a.html"
    assert _pick_attachment("pdf", "a.html", "b.pdf", "c.md") == "b.pdf"
    assert _pick_attachment("markdown", "a.html", "b.pdf", "c.md") == "c.md"


def test_pick_attachment_falls_through_when_requested_unavailable():
    # Asked for HTML but only PDF exists → return PDF
    assert _pick_attachment("html", None, "b.pdf", None) == "b.pdf"
    # Asked for PDF but only markdown exists → return markdown
    assert _pick_attachment("pdf", None, None, "c.md") == "c.md"


def test_pick_attachment_returns_none_when_nothing_available():
    assert _pick_attachment("html", None, None, None) is None


def test_extract_one_line_bet_finds_section():
    md = (
        "## Synthesis\n\n"
        "### Exec summary\nLots of text.\n\n"
        "### Kill Shot\nSomething bad.\n\n"
        "### 1-line bet\n\nBet only if X beats Y before Z.\n\n"
        "### Recommendation\nLean in.\n"
    )
    assert extract_one_line_bet(md) == "Bet only if X beats Y before Z."


def test_extract_one_line_bet_handles_markdown_emphasis():
    md = "### 1-line bet\n\n**Bold bet** with emphasis."
    assert extract_one_line_bet(md) == "Bold bet** with emphasis."  # strips leading **


def test_extract_one_line_bet_returns_none_when_missing():
    assert extract_one_line_bet("## Just some content") is None
    assert extract_one_line_bet("") is None
