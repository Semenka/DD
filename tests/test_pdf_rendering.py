"""PDF rendering — verify weasyprint integration produces a valid PDF."""

from pathlib import Path

import pytest

from dd_agent.report.renderer import render_pdf


def test_render_pdf_returns_bytes(tmp_path: Path):
    html = (
        "<!DOCTYPE html><html><head><title>t</title></head>"
        "<body><h1>Hi</h1><p>Body</p></body></html>"
    )
    out = tmp_path / "test.pdf"
    pdf = render_pdf(html=html, out_path=str(out))
    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF")
    assert out.exists()
    assert out.stat().st_size == len(pdf)


def test_render_pdf_without_out_path(tmp_path: Path):
    html = "<html><body><h1>OK</h1></body></html>"
    pdf = render_pdf(html=html)
    assert pdf.startswith(b"%PDF")
