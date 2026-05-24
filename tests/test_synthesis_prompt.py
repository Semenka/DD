"""Tests for the v7 4-pillar Exec Summary in orchestrator._SYNTH_PROMPT.

The Rivian regression exposed that the old Exec Summary prompt was free-form
("4-6 sentences. The whole deal in plain English."), which produces a hedging
paragraph instead of a partner-grade scorecard. v7 replaces it with Bessemer's
4 pillars: Founders / Co-investors / Growth metrics / Competitive position.
"""

from __future__ import annotations

from dd_agent.orchestrator import _SYNTH_PROMPT


# Required Bessemer 4-pillar headings — verbatim presence is a contract with
# the synthesis output, so if these strings ever drift the Exec Summary
# downstream will silently degrade.
_REQUIRED_PILLAR_HEADERS = [
    "**Founders.**",
    "**Co-investors.**",
    "**Growth metrics.**",
    "**Competitive position.**",
]


def test_all_four_pillar_headers_present():
    for header in _REQUIRED_PILLAR_HEADERS:
        assert header in _SYNTH_PROMPT, f"missing pillar: {header}"


def test_founders_pillar_references_photo_embed():
    """The Founders pillar must instruct the model to embed the founder
    photo via standard markdown image syntax."""
    assert "photo_profile" in _SYNTH_PROMPT
    assert "photo_path" in _SYNTH_PROMPT
    # The actual markdown image template the model is told to emit.
    assert "![" in _SYNTH_PROMPT and "](" in _SYNTH_PROMPT


def test_growth_metrics_pillar_references_arr_quality():
    """The Growth metrics pillar must use the arr_quality taxonomy so the
    model distinguishes real ARR from GMV / annualized pilots."""
    assert "arr_quality" in _SYNTH_PROMPT
    # All seven canonical labels should be present so the prompt is the
    # single source of truth for the controlled vocabulary.
    for label in (
        "recurring_subscription",
        "annualized_contracts",
        "annualized_pilots",
        "annualized_transactions",
        "gmv_or_take_rate",
        "one_time_hardware",
        "unclear",
    ):
        assert label in _SYNTH_PROMPT, f"missing arr_quality label: {label}"


def test_competitive_position_pillar_has_controlled_vocab():
    """Monopoly-likelihood verdict must use the 5-label controlled vocabulary."""
    for label in (
        "category winner",
        "co-leader",
        "challenger",
        "commodity",
        "uncertain",
    ):
        assert label in _SYNTH_PROMPT, f"missing monopoly label: {label}"


def test_coinvestors_pillar_references_named_partners():
    """The Co-investors pillar must distinguish top-tier VCs and super-angels."""
    # A representative top-tier VC name
    assert "Sequoia" in _SYNTH_PROMPT
    # A representative super-angel
    assert "Naval" in _SYNTH_PROMPT or "Elad Gil" in _SYNTH_PROMPT


def test_speculation_discipline_rule_present():
    """The v5 speculation discipline rule must survive in the synthesis prompt."""
    assert "speculation" in _SYNTH_PROMPT.lower()


def test_downstream_sections_unchanged():
    """The Beliefs Required / Kill Shot / 1-line bet / Recommendation sections
    are unchanged from v3 — verify they're still present so the rest of the
    report renderer stays compatible."""
    assert "### Beliefs Required to Invest" in _SYNTH_PROMPT
    assert "### Kill Shot" in _SYNTH_PROMPT
    assert "### 1-line bet" in _SYNTH_PROMPT
    assert "### Recommendation" in _SYNTH_PROMPT


# ---------- v8 additions: omission discipline + length cap -----------------


def test_v8_omission_discipline_rule_present():
    """The synthesis prompt must instruct the model to OMIT entire pillars
    rather than narrate absence — the core v8 'no unknown' fix."""
    assert "OMISSION DISCIPLINE" in _SYNTH_PROMPT
    # Specific language: prohibit absence-narrative phrases
    text = _SYNTH_PROMPT.lower()
    assert "omit" in text
    # Hedge phrases the model must NOT use
    assert "data is undisclosed" in text or '"unknown"' in text or "not disclosed" in text


def test_v8_word_cap_present():
    """The synthesis prompt must include a hard word-count budget so the
    final memo lands at the requested 5 pages."""
    assert "LENGTH BUDGET" in _SYNTH_PROMPT or "≤ 600 words" in _SYNTH_PROMPT


def test_v8_bessemer_omission_rule_present():
    """The Bessemer-memo prompt must enforce the same omission discipline."""
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "src" / "dd_agent" / "modules" / "bessemer_prompt.md"
    body = p.read_text()
    assert "OMISSION DISCIPLINE" in body
    assert "LENGTH BUDGET" in body or "≤ 1200 words" in body
