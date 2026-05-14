"""Search cascade — verifies routing, parsing, and fallback behavior.

These tests don't require API keys; they mock the network calls.
"""

import pytest

from dd_agent.data_sources import search as s


def test_unwrap_ddg_passthrough_http():
    assert s._unwrap_ddg("https://example.com/page") == "https://example.com/page"


def test_unwrap_ddg_protocol_relative():
    assert s._unwrap_ddg("//example.com/x") == "https://example.com/x"


def test_unwrap_ddg_uddg_param():
    url = "/l/?kh=-1&uddg=https%3A%2F%2Fexample.com%2Fdest"
    assert s._unwrap_ddg(url) == "https://example.com/dest"


def test_unwrap_ddg_empty():
    assert s._unwrap_ddg("") is None
    assert s._unwrap_ddg(None) is None


def test_gemini_extract_handles_full_response():
    payload = {
        "candidates": [{
            "content": {"parts": [{"text": "Final answer prose."}]},
            "groundingMetadata": {
                "groundingChunks": [
                    {"web": {"uri": "https://a.com", "title": "A"}},
                    {"web": {"uri": "https://b.com", "title": "B"}},
                    {"other": "no url"},  # malformed; should be skipped
                ],
            },
        }],
    }
    text, sources = s._gemini_extract(payload)
    assert text == "Final answer prose."
    assert [r.url for r in sources] == ["https://a.com", "https://b.com"]
    assert all(r.source == "gemini" for r in sources)


def test_gemini_extract_empty_response():
    text, sources = s._gemini_extract({})
    assert text == ""
    assert sources == []


def test_search_result_is_immutable():
    r = s.SearchResult(url="https://x.com", title="X", snippet="y", source="perplexity")
    with pytest.raises(Exception):
        r.url = "different"  # frozen dataclass


async def test_web_search_falls_through_when_no_keys(monkeypatch):
    """No API keys → only DDG is attempted; we monkeypatch DDG to return a stub."""
    for k in ("PERPLEXITY_API_KEY", "GEMINI_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    called = {"ddg": 0}

    async def fake_ddg(query, max_results):
        called["ddg"] += 1
        return [s.SearchResult(url="https://stub", title="t", snippet="", source="duckduckgo")]

    monkeypatch.setattr(s, "_duckduckgo", fake_ddg)
    results = await s.web_search("anything", max_results=3)
    assert called["ddg"] == 1
    assert results[0].source == "duckduckgo"


async def test_web_search_prefers_perplexity(monkeypatch):
    """When PERPLEXITY_API_KEY is set, the cascade should call Perplexity first."""
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test")
    order: list[str] = []

    async def fake_pplx(query, max_results):
        order.append("pplx")
        return [s.SearchResult(url="https://p", title="p", snippet="", source="perplexity")]

    async def fake_ddg(query, max_results):
        order.append("ddg")
        return [s.SearchResult(url="https://d", title="d", snippet="", source="duckduckgo")]

    monkeypatch.setattr(s, "_perplexity_search", fake_pplx)
    monkeypatch.setattr(s, "_duckduckgo", fake_ddg)
    results = await s.web_search("test", max_results=3)
    assert order == ["pplx"]
    assert results[0].source == "perplexity"
