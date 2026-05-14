"""Build / refresh the Elad-excerpt corpus.

Scrapes blog.eladgil.com posts and writes them as markdown files with
frontmatter into data/elad_excerpts/. The BM25 retriever picks them up at
first call.

The seed `data/elad_excerpts/*.md` files (high_growth_handbook_principles,
founder_pattern_matching, market_inflection, saas_traction_benchmarks,
coinvestor_lens) ship with the repo so the agent works without running this.

Run with:
  python scripts/build_elad_corpus.py [--max-posts N]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
from pathlib import Path
from urllib.parse import urljoin

import httpx
from selectolax.parser import HTMLParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_elad_corpus")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "elad_excerpts"
BASE = "https://blog.eladgil.com"


async def _fetch(url: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "DD-Agent corpus build"})
            r.raise_for_status()
            return r.text
    except Exception as exc:
        log.warning("fetch failed %s: %s", url, exc)
        return None


def _slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\s-]", "", s).strip().lower()
    return re.sub(r"\s+", "_", s)[:80] or "untitled"


def _parse_post(html: str, url: str) -> tuple[str, str] | None:
    tree = HTMLParser(html)
    title_node = tree.css_first("h1") or tree.css_first("title")
    title = title_node.text(strip=True) if title_node else "Untitled"
    article = tree.css_first("article") or tree.css_first(".post") or tree.body
    if article is None:
        return None
    for sel in ["nav", "header", "footer", "script", "style", "noscript", "aside"]:
        for n in article.css(sel):
            n.decompose()
    text = article.text(separator="\n\n", strip=True)
    if len(text) < 500:
        return None
    return title, text


async def _list_post_urls(max_posts: int) -> list[str]:
    html = await _fetch(BASE)
    if not html:
        return []
    tree = HTMLParser(html)
    urls: list[str] = []
    seen: set[str] = set()
    for a in tree.css("a"):
        href = a.attributes.get("href", "")
        if not href:
            continue
        full = urljoin(BASE, href)
        if BASE not in full:
            continue
        if full in seen:
            continue
        if any(seg in full for seg in ["/tag/", "/category/", "/page/", "#", "/feed"]):
            continue
        if full.rstrip("/") == BASE.rstrip("/"):
            continue
        seen.add(full)
        urls.append(full)
        if len(urls) >= max_posts:
            break
    return urls


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-posts", type=int, default=30)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    urls = await _list_post_urls(args.max_posts)
    log.info("found %d candidate post URLs", len(urls))

    saved = 0
    for url in urls:
        html = await _fetch(url)
        if not html:
            continue
        parsed = _parse_post(html, url)
        if not parsed:
            continue
        title, text = parsed
        slug = _slugify(title)
        out_path = OUT_DIR / f"eladgil_{slug}.md"
        body = f"---\ntitle: {title}\nurl: {url}\nsource_type: blog\n---\n\n# {title}\n\n{text}\n"
        out_path.write_text(body, encoding="utf-8")
        saved += 1
        log.info("saved %s", out_path.name)

    log.info("done — saved %d posts to %s", saved, OUT_DIR)


if __name__ == "__main__":
    asyncio.run(main())
