"""Citation tracking and bibliography rendering.

Each subagent emits Citation objects pointing at web URLs, retrieved Elad
excerpts, SEC filings, or Yahoo Finance tickers. CitationBook dedupes by URL
(or source key) and assigns stable [n] reference numbers, then renders a
bibliography at the end of the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class Citation:
    """A single source citation."""
    key: str                    # stable dedup key (URL, source filename, etc.)
    title: str
    url: str | None = None
    snippet: str | None = None  # short quoted text or summary
    source_type: str = "web"    # "web" | "elad" | "edgar" | "yahoo" | "github" | "podcast"


@dataclass
class CitationBook:
    """Ordered, deduped citation collection. [n] refs are assigned by insertion order."""
    citations: list[Citation] = field(default_factory=list)
    _by_key: dict[str, int] = field(default_factory=dict)

    def add(self, c: Citation) -> int:
        """Add a citation. Returns its 1-indexed reference number."""
        if c.key in self._by_key:
            return self._by_key[c.key] + 1
        self.citations.append(c)
        self._by_key[c.key] = len(self.citations) - 1
        return len(self.citations)

    def add_many(self, items: Iterable[Citation]) -> None:
        for c in items:
            self.add(c)

    def ref_for(self, key: str) -> int | None:
        i = self._by_key.get(key)
        return None if i is None else i + 1

    def render_markdown(self) -> str:
        if not self.citations:
            return ""
        lines = ["## References", ""]
        for i, c in enumerate(self.citations, 1):
            if c.url:
                lines.append(f"[{i}] [{c.title}]({c.url}) — *{c.source_type}*")
            else:
                lines.append(f"[{i}] {c.title} — *{c.source_type}*")
            if c.snippet:
                snippet = c.snippet.strip().replace("\n", " ")
                if len(snippet) > 220:
                    snippet = snippet[:220].rstrip() + "…"
                lines.append(f"    > {snippet}")
        return "\n".join(lines)

    def to_list(self) -> list[dict]:
        return [
            {
                "n": i + 1,
                "key": c.key,
                "title": c.title,
                "url": c.url,
                "source_type": c.source_type,
                "snippet": c.snippet,
            }
            for i, c in enumerate(self.citations)
        ]
