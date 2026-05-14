"""BM25 retrieval over Elad excerpts (HGH chapters, blog posts, podcast transcripts).

Loads `data/elad_excerpts/*.md` at first call, chunks by paragraph, indexes with
rank-bm25. Each chunk carries source metadata so the citations module can render
a bibliography. If the corpus dir is empty, returns no snippets — the agent still
runs, just without retrieval-augmented few-shot.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi


@dataclass(frozen=True)
class Snippet:
    source: str            # filename (e.g. "high_growth_handbook_ch3.md")
    title: str             # human-readable
    url: str | None        # original URL if known
    text: str
    score: float = 0.0


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-']{1,}", text.lower())


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta_raw, body = parts[1], parts[2]
    meta: dict[str, str] = {}
    for line in meta_raw.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, body.strip()


def _chunk(text: str, target_chars: int = 800) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= target_chars:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


@lru_cache(maxsize=1)
def _load_corpus() -> tuple[list[Snippet], BM25Okapi | None]:
    base = Path(os.environ.get("DD_DATA_DIR", "./data")) / "elad_excerpts"
    snippets: list[Snippet] = []
    if not base.exists():
        return snippets, None
    for path in sorted(base.glob("*.md")):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        meta, body = _parse_frontmatter(raw)
        title = meta.get("title", path.stem.replace("_", " ").title())
        url = meta.get("url")
        for chunk in _chunk(body):
            snippets.append(
                Snippet(source=path.name, title=title, url=url, text=chunk)
            )
    if not snippets:
        return snippets, None
    tokenized = [_tokenize(s.text) for s in snippets]
    return snippets, BM25Okapi(tokenized)


def retrieve(query: str, k: int = 5) -> list[Snippet]:
    """Return top-k Elad-excerpt snippets for the query."""
    snippets, bm25 = _load_corpus()
    if not snippets or bm25 is None:
        return []
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    order = sorted(range(len(snippets)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        Snippet(
            source=snippets[i].source,
            title=snippets[i].title,
            url=snippets[i].url,
            text=snippets[i].text,
            score=float(scores[i]),
        )
        for i in order
        if scores[i] > 0.0
    ]


def format_for_prompt(snippets: list[Snippet]) -> str:
    """Render retrieved snippets as <reference> blocks for prompt injection."""
    if not snippets:
        return ""
    blocks = []
    for s in snippets:
        head = s.title
        if s.url:
            head = f"{s.title} — {s.url}"
        blocks.append(f"<reference source=\"{s.source}\" title=\"{head}\">\n{s.text}\n</reference>")
    return "\n\n".join(blocks)
