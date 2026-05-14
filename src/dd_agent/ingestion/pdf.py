"""PDF text extraction for memos and pitch decks."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_text(pdf_path: str | Path, max_pages: int | None = None) -> str:
    """Extract text from a PDF. Returns one string with form-feed page separators."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        if max_pages is not None and i >= max_pages:
            break
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001
            pages.append(f"[page {i+1} extraction failed: {exc}]")
    return "\f".join(pages)


def page_count(pdf_path: str | Path) -> int:
    return len(PdfReader(str(pdf_path)).pages)
