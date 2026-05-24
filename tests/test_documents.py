"""Tests for the document-extraction dispatcher.

v7 background: a .docx file is a ZIP archive whose magic bytes are PK\\x03\\x04.
Before v7 the orchestrator fell back to `Path.read_text()` for any extension
other than .pdf — that emitted the literal characters "PK" as the first
capitalized token, and the LLM picked it up as the company name. The
dispatcher in `ingestion/pdf.py::extract_document` fixes the source problem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dd_agent.ingestion.pdf import extract_docx, extract_document


def _make_docx(path: Path, text_blocks: list[str]) -> None:
    """Build a real .docx using python-docx so the test exercises the same
    library used in production."""
    import docx
    doc = docx.Document()
    for line in text_blocks:
        doc.add_paragraph(line)
    doc.save(str(path))


def test_extract_docx_returns_paragraph_text(tmp_path: Path) -> None:
    target = tmp_path / "memo.docx"
    _make_docx(target, [
        "Rivian Investment Memo",
        "Rivian designs and manufactures electric pickup trucks.",
        "Founder: RJ Scaringe (CEO).",
    ])
    text = extract_docx(target)
    assert "Rivian" in text
    assert "RJ Scaringe" in text
    # PK magic header MUST NOT leak through the extractor.
    assert "PK\x03\x04" not in text
    assert not text.startswith("PK")


def test_extract_document_dispatches_by_extension(tmp_path: Path) -> None:
    # .docx → python-docx
    docx_path = tmp_path / "deal.docx"
    _make_docx(docx_path, ["Acme Corp builds widgets."])
    text = extract_document(docx_path)
    assert "Acme" in text

    # .md → read_text
    md_path = tmp_path / "deal.md"
    md_path.write_text("# Acme\n\nRevenue: $5M ARR.")
    text = extract_document(md_path)
    assert "Acme" in text and "$5M ARR" in text

    # .txt → read_text
    txt_path = tmp_path / "deal.txt"
    txt_path.write_text("Acme is a SaaS company.")
    text = extract_document(txt_path)
    assert "SaaS" in text


def test_extract_document_legacy_doc_raises(tmp_path: Path) -> None:
    """Legacy .doc format isn't supported — must raise NotImplementedError
    rather than silently returning garbage from read_text."""
    fake = tmp_path / "old.doc"
    fake.write_bytes(b"\xd0\xcf\x11\xe0fake-CFBF-header")
    with pytest.raises(NotImplementedError):
        extract_document(fake)


def test_extract_document_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_document(tmp_path / "does_not_exist.pdf")


def test_extract_document_unknown_extension_falls_back_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown extensions should still return the text (best-effort) and log
    a warning — we never silently truncate the input."""
    weird = tmp_path / "deal.weird"
    weird.write_text("Some plain content.", encoding="utf-8")
    with caplog.at_level("WARNING", logger="dd_agent.documents"):
        text = extract_document(weird)
    assert text == "Some plain content."
    assert any("unknown file extension" in rec.message for rec in caplog.records)
