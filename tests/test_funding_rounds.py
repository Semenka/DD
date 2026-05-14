"""Funding-rounds adapter — JSON parsing + ranking."""

from dd_agent.context import FundingRound
from dd_agent.data_sources import funding_rounds as fr
from dd_agent.data_sources.search import SearchResult


def test_parse_json_handles_markdown_fences():
    out = fr._parse_json('```json\n{"rounds":[{"round_type":"seed"}]}\n```')
    assert out == {"rounds": [{"round_type": "seed"}]}


def test_parse_json_handles_preamble():
    out = fr._parse_json('Here is the JSON: {"rounds":[]} done')
    assert out == {"rounds": []}


def test_to_float():
    assert fr._to_float(None) is None
    assert fr._to_float("") is None
    assert fr._to_float("not a number") is None
    assert fr._to_float(5_000_000) == 5_000_000.0
    assert fr._to_float("12.5") == 12.5


def test_rank_for_fetch_prioritizes_authoritative_hosts():
    results = [
        SearchResult(url="https://random-blog.com/x", title="Random", snippet="", source="ddg"),
        SearchResult(url="https://crunchbase.com/organization/acme", title="CB", snippet="", source="ddg"),
        SearchResult(url="https://techcrunch.com/2024/01/acme-series-b", title="TC", snippet="", source="ddg"),
    ]
    ranked = fr._rank_for_fetch(results)
    assert "crunchbase.com" in ranked[0].url
    assert "techcrunch.com" in ranked[1].url


def test_to_jsonable_roundtrips():
    rounds = [
        FundingRound(round_type="seed", date="2020-03", amount_usd=2_000_000,
                     lead_investors=["Foo"], participants=["Bar"]),
        FundingRound(round_type="series_a", date="2021-09", amount_usd=12_000_000,
                     post_money_valuation_usd=80_000_000),
    ]
    data = fr.to_jsonable(rounds)
    assert len(data) == 2
    assert data[0]["round_type"] == "seed"
    assert data[1]["post_money_valuation_usd"] == 80_000_000
