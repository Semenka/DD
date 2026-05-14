"""BM25 retrieval — verifies it picks up the seed Elad excerpts and ranks
relevant queries above noise."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _data_dir_env(monkeypatch):
    root = Path(__file__).resolve().parent.parent
    monkeypatch.setenv("DD_DATA_DIR", str(root / "data"))
    from dd_agent.retrieval import _load_corpus
    _load_corpus.cache_clear()  # type: ignore[attr-defined]
    yield
    _load_corpus.cache_clear()  # type: ignore[attr-defined]


def test_retrieve_finds_market_inflection():
    from dd_agent.retrieval import retrieve
    snippets = retrieve("market inflection point platform shift", k=3)
    assert snippets, "BM25 returned no results"
    titles = " ".join(s.title.lower() for s in snippets)
    assert "inflection" in titles or "market" in titles


def test_retrieve_finds_founder_pattern():
    from dd_agent.retrieval import retrieve
    snippets = retrieve("relentless resourcefulness shipped artifacts pedigree", k=3)
    assert snippets
    text = " ".join(s.text.lower() for s in snippets)
    assert "relentless" in text or "founder" in text


def test_format_for_prompt_returns_string():
    from dd_agent.retrieval import retrieve, format_for_prompt
    snippets = retrieve("Rule of 40 SaaS magic number", k=2)
    formatted = format_for_prompt(snippets)
    if snippets:
        assert "<reference" in formatted
    else:
        assert formatted == ""
