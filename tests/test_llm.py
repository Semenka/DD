"""Codex CLI helper — JSONL parsing, missing-binary handling."""

import pytest

from dd_agent.modules._llm import (
    CodexUnavailableError,
    _last_agent_message,
    codex_path,
    rewrite_citations,
)


def test_last_agent_message_picks_final_text():
    jsonl = "\n".join([
        '{"type":"thread.started","thread_id":"t1"}',
        '{"type":"turn.started"}',
        '{"type":"item.completed","item":{"id":"i0","type":"agent_message","text":"first"}}',
        '{"type":"item.completed","item":{"id":"i1","type":"agent_message","text":"final"}}',
        '{"type":"turn.completed","usage":{"input_tokens":1}}',
    ])
    assert _last_agent_message(jsonl) == "final"


def test_last_agent_message_ignores_non_message_items():
    jsonl = "\n".join([
        '{"type":"item.completed","item":{"type":"reasoning","text":"thinking"}}',
        '{"type":"item.completed","item":{"type":"agent_message","text":"actual reply"}}',
    ])
    assert _last_agent_message(jsonl) == "actual reply"


def test_last_agent_message_empty_when_no_messages():
    assert _last_agent_message("") == ""
    assert _last_agent_message('{"type":"turn.started"}') == ""


def test_last_agent_message_skips_malformed_lines():
    jsonl = "\n".join([
        "not json at all",
        '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}',
    ])
    assert _last_agent_message(jsonl) == "ok"


def test_codex_path_raises_when_missing(monkeypatch):
    monkeypatch.setenv("DD_CODEX_BIN", "/definitely/not/a/real/binary/codex")
    with pytest.raises(CodexUnavailableError):
        codex_path()


def test_rewrite_citations_remaps():
    text = "Foo [1] and [2] but not [99]."
    out = rewrite_citations(text, {1: 7, 2: 8})
    assert out == "Foo [7] and [8] but not [99]."


def test_rewrite_citations_handles_missing_map():
    text = "Foo [1]"
    out = rewrite_citations(text, {})
    assert out == "Foo [1]"
