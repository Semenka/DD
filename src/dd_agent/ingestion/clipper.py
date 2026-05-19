"""Obsidian Web Clipper ingestion.

Real deal memos sometimes arrive as `.md` files clipped from the web via the
Obsidian Web Clipper browser extension. They have:
  - YAML frontmatter with the source URL (field name varies: `source`, `url`,
    `link`, `original`)
  - Markdown body, often with embedded image links and external links
  - Sometimes a separate pitch-deck link inside the body
    (docsend.com, pitch.com, figma.com/proto, google docs presentation, etc.)

This module parses the frontmatter tolerantly, surfaces the source URL, and
detects the FIRST hosted-deck link so the orchestrator can screenshot it.

Falls through gracefully — returns None for non-clipping `.md` files so the
orchestrator can treat them as plain text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Hosted-deck patterns. Order matters — more specific patterns first. Each
# pattern is matched case-insensitively against URLs in the body text.
_DECK_HOSTS = (
    r"docsend\.com/view/[A-Za-z0-9]+",
    r"docsend\.com/v/[A-Za-z0-9/_-]+",
    r"pitch\.com/v/[A-Za-z0-9/_-]+",
    r"pitch\.com/public/[A-Za-z0-9/_-]+",
    r"pitch\.io/[A-Za-z0-9/_-]+",
    r"figma\.com/proto/[A-Za-z0-9/_-]+",
    r"figma\.com/deck/[A-Za-z0-9/_-]+",
    r"docs\.google\.com/presentation/d/[A-Za-z0-9/_-]+",
    r"slides\.com/[A-Za-z0-9/_-]+",
    r"slideshare\.net/[A-Za-z0-9/_-]+",
    r"gamma\.app/docs/[A-Za-z0-9/_-]+",
    r"notion\.so/[A-Za-z0-9/_-]+",  # least specific — could be any notion page
)
_DECK_RE = re.compile(
    r"https?://(?:www\.)?(?:" + "|".join(_DECK_HOSTS) + r")",
    re.IGNORECASE,
)

# Image links inside the markdown body — both standard `![alt](url)` and
# wiki-style `![[file]]` (Obsidian's local-vault syntax, which we can't
# follow externally — we just record them).
_IMG_MD_RE = re.compile(r"!\[[^\]]*\]\((https?://[^\)]+)\)")
_IMG_WIKI_RE = re.compile(r"!\[\[([^\]]+)\]\]")

# All standard markdown links — used to find the source URL when frontmatter
# is absent.
_LINK_MD_RE = re.compile(r"\[(?:[^\]]+)\]\((https?://[^\)]+)\)")


@dataclass
class ClippingContext:
    """Parsed Obsidian Web Clipper output."""
    source_url: str | None = None
    deck_url: str | None = None
    body_text: str = ""
    title: str | None = None
    author: str | None = None
    clipped_at: str | None = None
    embedded_image_urls: list[str] = field(default_factory=list)
    embedded_image_wiki_refs: list[str] = field(default_factory=list)
    raw_frontmatter: dict[str, str] = field(default_factory=dict)


def parse_file(path: str | Path) -> ClippingContext | None:
    """Read a `.md` file and return a ClippingContext if it's a clipping, else None."""
    p = Path(path)
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8", errors="ignore")
    return parse(text)


def parse(text: str) -> ClippingContext | None:
    """Parse a markdown string. Returns ClippingContext if the file looks like
    an Obsidian Web Clipper output (has frontmatter OR has a recognizable
    structure), else returns None for plain-markdown files."""
    fm, body = _split_frontmatter(text)
    has_clipping_signal = bool(fm) or _has_clipper_signal(body)
    if not has_clipping_signal:
        return None

    source_url = _pick_source_url(fm, body)
    deck_url = extract_deck_url(body)
    images_md = _IMG_MD_RE.findall(body)
    images_wiki = _IMG_WIKI_RE.findall(body)

    return ClippingContext(
        source_url=source_url,
        deck_url=deck_url,
        body_text=body.strip(),
        title=fm.get("title"),
        author=fm.get("author") or fm.get("creator") or fm.get("byline"),
        clipped_at=fm.get("created") or fm.get("clipped") or fm.get("date"),
        embedded_image_urls=images_md,
        embedded_image_wiki_refs=images_wiki,
        raw_frontmatter=fm,
    )


def extract_deck_url(body: str) -> str | None:
    """Return the first hosted-deck URL in the body, or None."""
    if not body:
        return None
    m = _DECK_RE.search(body)
    return m.group(0) if m else None


# --- internals ---------------------------------------------------------------


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Tolerant YAML frontmatter splitter. Accepts both `---\\n...\\n---` and
    `+++\\n...\\n+++` blocks; returns ({}, text) on any parse difficulty.

    We don't pull in PyYAML — frontmatter from Obsidian clippings is almost
    always flat `key: value` lines, sometimes with list values rendered as
    `[a, b, c]`. We parse defensively and return strings; lists become the
    raw bracketed string."""
    if not text:
        return {}, text
    stripped = text.lstrip()
    delim = None
    if stripped.startswith("---"):
        delim = "---"
    elif stripped.startswith("+++"):
        delim = "+++"
    if not delim:
        return {}, text
    # Find closing delimiter on its own line.
    rest = stripped[len(delim):]
    if rest.startswith("\n"):
        rest = rest[1:]
    closer_re = re.compile(rf"^{re.escape(delim)}\s*$", re.MULTILINE)
    m = closer_re.search(rest)
    if not m:
        return {}, text
    fm_block = rest[: m.start()]
    body = rest[m.end():].lstrip("\n")
    fm = {}
    for line in fm_block.splitlines():
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        key = k.strip().lower()
        val = v.strip().strip('"').strip("'")
        if not key:
            continue
        fm[key] = val
    return fm, body


def _has_clipper_signal(body: str) -> bool:
    """Heuristic: even without frontmatter, a body containing a clear "source:"
    line or several external links suggests a clipping. Used so plain memos
    written in markdown still get parsed (and any deck links surfaced)."""
    if not body:
        return False
    # If there's a hosted deck link anywhere, treat as clipping signal.
    if _DECK_RE.search(body):
        return True
    # If there are 3+ external links, treat as clipping signal.
    links = _LINK_MD_RE.findall(body)
    return len(links) >= 3


def _pick_source_url(fm: dict[str, str], body: str) -> str | None:
    """Frontmatter source URL takes priority; otherwise first body link."""
    for key in ("source", "url", "link", "original", "permalink", "canonical"):
        v = fm.get(key)
        if v and v.startswith("http"):
            return v
    m = _LINK_MD_RE.search(body)
    return m.group(1) if m else None
