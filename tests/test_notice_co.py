"""notice.co adapter — HTML parsing and graceful no-data fallback."""

from dd_agent.data_sources import notice_co as nco


def test_parse_quotes_extracts_typical_labels():
    html = """
    <html><body>
      <div>Last price <span>$45.20</span></div>
      <div>Implied valuation: $9.2B</div>
      <div>Bid <strong>$44.80</strong></div>
      <div>Ask <strong>$45.60</strong></div>
      <div>Last trade Sep 12, 2025</div>
    </body></html>
    """
    out = nco._parse_quotes(html)
    assert out["last_price"] == 45.20
    assert out["bid"] == 44.80
    assert out["ask"] == 45.60
    assert out["implied_valuation"] == 9_200_000_000
    assert out["last_trade_date"] == "Sep 12, 2025"


def test_parse_quotes_handles_missing_fields():
    """If only some fields are present, the rest should be absent (not invented)."""
    html = "<p>Implied valuation: $1.5B</p>"
    out = nco._parse_quotes(html)
    assert out.get("implied_valuation") == 1_500_000_000
    assert "last_price" not in out
    assert "bid" not in out


def test_parse_quotes_empty_html_returns_empty():
    assert nco._parse_quotes("") == {}
    assert nco._parse_quotes("<p>nothing relevant here</p>") == {}


def test_mid_helper():
    assert nco._mid(10.0, 12.0) == 11.0
    assert nco._mid(None, 12.0) is None
    assert nco._mid(10.0, None) is None


def test_scale_suffixes():
    assert nco._scale("K") == 1e3
    assert nco._scale("m") == 1e6
    assert nco._scale("B") == 1e9
    assert nco._scale(None) == 1.0
    assert nco._scale("") == 1.0
