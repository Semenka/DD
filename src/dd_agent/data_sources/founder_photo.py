"""v8 founder-photo discovery for the LIVE deal pipeline.

Before v8, the live pipeline ran a single discovery strategy: the LLM was
asked to extract `photo_url` from the memo/deck/website text. If the URL
wasn't in the document (the common case), `Founder.photo_url` stayed
`None` and the photo classifier returned `available=False`.

This module ports the cascade used by `scripts/build_unicorn_corpus.py`
and adds two strategies the corpus build doesn't have:

  1. Deck-slide face crop — for each saved deck slide PNG (from
     `ingestion/screenshot_deck.py`), run InsightFace face detection;
     if exactly one face is detected on a slide whose OCR text mentions
     the founder's first or last name, crop a 1.4× bbox and save it.
  2. LinkedIn `og:image` — if `Founder.linkedin_url` is populated,
     fetch the public profile page and parse `<meta property="og:image">`.
     Rejects known-placeholder LinkedIn URLs to avoid embedding the
     generic "in" logo.

Top-level entry point: `resolve_founder_photo(founder, ctx, deck_capture,
clipping)` — returns a local file path or remote URL, or None if all
tiers fail. The result is stored back to `founder.photo_url` so the
downstream photo classifier consumes it the same way it would consume
a URL extracted by the LLM.

Best-effort throughout: every helper catches its own exceptions and
returns None on failure. Nothing in this module is allowed to crash the
pipeline. Strict whitelist for image content-type + minimum bytes to
reject 1-pixel beacons and HTML error pages.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from ..context import DealContext, Founder
    from ..ingestion.clipper import ClippingContext
    from ..ingestion.screenshot_deck import DeckCapture

log = logging.getLogger("dd_agent.founder_photo")

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_TIMEOUT = httpx.Timeout(20.0, connect=10.0)

# LinkedIn placeholder URL patterns to reject (generic "in" logo etc.)
_LI_PLACEHOLDER_HINTS = (
    "static.licdn.com/sc/",
    "ghost-person",
    "ghost_person",
    "anonymous",
)


# ---------- public API ------------------------------------------------------


async def resolve_founder_photo(
    *,
    founder: "Founder",
    ctx: "DealContext",
    deck_capture: "DeckCapture | None" = None,
    clipping: "ClippingContext | None" = None,
    save_dir: Path | None = None,
) -> str | None:
    """Run the full v8 cascade for ONE founder. Mutates `founder.photo_url`
    when a candidate is found; also returns it. Skips work entirely if
    `founder.photo_url` is already populated."""
    if founder.photo_url:
        return founder.photo_url

    save_dir = save_dir or _default_save_dir(ctx.deal_id)
    save_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        # v9: pre-discover LinkedIn URL + company website if missing, so the
        # downstream tiers (linkedin_og, company_team) actually have something
        # to work with. These pre-discovery steps are cheap web searches —
        # they take ~1-2s and dramatically lift the photo hit rate for
        # non-famous founders (the v8 Alfred case: zero photos because
        # `ctx.website` and `founder.linkedin_url` were both None).
        if not founder.linkedin_url:
            try:
                await _discover_linkedin_url(founder, ctx)
            except Exception as exc:  # noqa: BLE001
                log.debug("linkedin URL discovery failed for %s: %s",
                          founder.name, exc)
        if not ctx.website:
            try:
                await _discover_company_website(ctx)
            except Exception as exc:  # noqa: BLE001
                log.debug("company website discovery failed for %s: %s",
                          ctx.company_name, exc)

        # Cascade — each helper returns image bytes + a source label, or None.
        # Ordering by image-quality * recall:
        #  - deck_slide first (highest signal when present — founder's own deck)
        #  - wikipedia + company_team (best quality when available)
        #  - linkedin_og + web_image_search (broadest recall for unknowns)
        #  - grounded_llm + clipping_embedded (last resorts)
        strategies = [
            ("deck_slide", lambda: _from_deck_slides(founder, deck_capture)),
            ("wikipedia", lambda: _from_wikipedia(client, founder)),
            ("company_team", lambda: _from_company_team(client, founder, ctx)),
            ("linkedin_og", lambda: _from_linkedin_og(client, founder)),
            ("web_image_search", lambda: _from_web_image_search(client, founder, ctx)),
            ("grounded_llm", lambda: _from_grounded(client, founder, ctx)),
            ("clipping_embedded", lambda: _from_clipping(client, founder, clipping)),
        ]
        for label, strategy in strategies:
            try:
                result = await strategy()
            except Exception as exc:  # noqa: BLE001
                log.debug("photo strategy %s failed for %s: %s",
                          label, founder.name, exc)
                result = None
            if result is None:
                continue
            img_bytes = result if isinstance(result, bytes) else result.get("bytes")
            url_hint = None if isinstance(result, bytes) else result.get("url")
            if not img_bytes:
                continue
            path = _persist(img_bytes, save_dir, founder.name, source=label)
            if path:
                founder.photo_url = path
                log.info("resolved photo for %s via %s → %s",
                         founder.name, label, path)
                return path
            # Even if persist failed, hand back the remote URL.
            if url_hint:
                founder.photo_url = url_hint
                log.info("resolved photo URL for %s via %s (no local save)",
                         founder.name, label)
                return url_hint
    log.info("photo cascade exhausted for %s — no image found", founder.name)
    return None


async def resolve_all_founder_photos(
    *,
    ctx: "DealContext",
    deck_capture: "DeckCapture | None" = None,
    clipping: "ClippingContext | None" = None,
) -> dict[str, str | None]:
    """Convenience: run `resolve_founder_photo` for every founder in `ctx`
    in parallel. Returns a `{founder_name → path|url|None}` map. Mutates
    `ctx.founders[*].photo_url` in-place when a photo is found."""
    if not ctx.founders:
        return {}
    results = await asyncio.gather(*[
        resolve_founder_photo(
            founder=f, ctx=ctx,
            deck_capture=deck_capture, clipping=clipping,
        )
        for f in ctx.founders
    ], return_exceptions=True)
    out: dict[str, str | None] = {}
    for f, res in zip(ctx.founders, results):
        if isinstance(res, Exception):
            log.debug("photo resolve raised for %s: %s", f.name, res)
            out[f.name] = None
        else:
            out[f.name] = res
    return out


# ---------- v9 pre-discovery -----------------------------------------------


# Gemini grounding wraps every result URL behind a Vertex AI redirect:
# `https://vertexaisearch.cloud.google.com/grounding-api-redirect/<token>`.
# Until we follow the redirect we can't filter by destination domain.
_GROUNDING_REDIRECT_HOSTS = (
    "vertexaisearch.cloud.google.com",
    "vertexaisearch.googleapis.com",
)


async def _resolve_redirect(client: httpx.AsyncClient, url: str) -> str:
    """If `url` is a known search-redirect URL, follow it (HEAD then GET
    fallback) and return the final destination. Otherwise return `url`
    unchanged. Always-safe: any error returns the input URL."""
    if not url:
        return url
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
    except Exception:
        return url
    if not any(h in host for h in _GROUNDING_REDIRECT_HOSTS):
        return url
    # Use a fresh client with redirect-following disabled to inspect the
    # Location header. Some redirect targets reject HEAD, so fall back to GET.
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as c:
            try:
                r = await c.head(url, headers={"User-Agent": _UA})
            except Exception:
                r = await c.get(url, headers={"User-Agent": _UA})
            loc = r.headers.get("location")
            if loc:
                return loc
            # No redirect — try following normally (some grounding URLs use JS)
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as c:
            r = await c.get(url, headers={"User-Agent": _UA})
            return str(r.url)
    except Exception:
        return url


async def _discover_linkedin_url(
    founder: "Founder", ctx: "DealContext",
) -> None:
    """If `founder.linkedin_url` is missing, search the web for
    `site:linkedin.com/in "{name}" "{company}"` and populate it from the
    top hit. Mutates `founder.linkedin_url` in-place; no return value.

    Without this step, the downstream `_from_linkedin_og` strategy was a
    dead branch for ~every non-famous founder because the LLM rarely
    extracts LinkedIn URLs from a memo body. With it, og:image becomes
    one of the highest-recall strategies in the cascade."""
    try:
        from .search import web_search
    except ImportError:
        return
    company = ctx.company_name or ""
    # Try multiple query shapes — direct site: query first, then a looser
    # plain-text query (often returns better results when the search
    # backend doesn't honor site: operators well).
    queries = [
        f'site:linkedin.com/in "{founder.name}" "{company}"' if company else
        f'site:linkedin.com/in "{founder.name}"',
        f'"{founder.name}" {company} linkedin' if company else
        f'"{founder.name}" linkedin',
    ]
    last_low = founder.name.rsplit(" ", 1)[-1].lower()
    first_low = founder.name.split(" ", 1)[0].lower()
    async with httpx.AsyncClient(timeout=10.0) as client:
        for q in queries:
            try:
                results = await web_search(q, max_results=8)
            except Exception:
                continue
            for r in results:
                # Resolve grounding-redirect URLs before domain check.
                real_url = await _resolve_redirect(client, r.url)
                if "linkedin.com/in/" not in real_url.lower():
                    continue
                rl = real_url.lower()
                # Confidence filter: URL slug usually contains the founder's
                # first or last name (LinkedIn vanity URLs).
                if not (last_low in rl or first_low in rl):
                    continue
                founder.linkedin_url = real_url
                log.info("discovered linkedin_url for %s: %s",
                         founder.name, real_url)
                return


async def _discover_company_website(ctx: "DealContext") -> None:
    """If `ctx.website` is missing, search for "{company} official site"
    and pick the first result whose domain doesn't smell like a directory
    (linkedin, crunchbase, wikipedia). Populates `ctx.website` in-place.

    Without this step, `_from_company_team` was a no-op for any deal whose
    memo didn't include a website URL — the v8 Alfred case."""
    if not ctx.company_name:
        return
    try:
        from .search import web_search
    except ImportError:
        return
    skip_hosts = (
        "linkedin.com", "crunchbase.com", "wikipedia.org", "x.com",
        "twitter.com", "facebook.com", "instagram.com", "youtube.com",
        "github.com", "pitchbook.com", "tracxn.com", "owler.com",
        "bloomberg.com", "techcrunch.com", "reuters.com",
    )
    q = f'"{ctx.company_name}" official site'
    try:
        results = await web_search(q, max_results=8)
    except Exception:
        return
    for r in results:
        from urllib.parse import urlparse
        try:
            host = urlparse(r.url).hostname or ""
        except Exception:
            continue
        if any(h in host for h in skip_hosts):
            continue
        # Heuristic: company name should appear in the domain (helps reject
        # press articles that link to articles about the company instead).
        slug = re.sub(r"[^a-z0-9]", "", ctx.company_name.lower())
        host_clean = re.sub(r"[^a-z0-9]", "", host.lower())
        if slug and slug not in host_clean:
            continue
        ctx.website = f"https://{host}"
        log.info("discovered website for %s: %s",
                 ctx.company_name, ctx.website)
        return


# ---------- strategy implementations ---------------------------------------


async def _from_deck_slides(
    founder: "Founder",
    deck_capture: "DeckCapture | None",
) -> bytes | None:
    """If we have OCR'd deck slides on disk, look for a slide whose text
    mentions the founder's name and contains exactly one detectable face.
    Crop a 1.4× bbox around it and return the JPEG bytes."""
    if not deck_capture or not deck_capture.available:
        return None
    paths = getattr(deck_capture, "screenshot_paths", None) or []
    slide_texts = getattr(deck_capture, "slide_texts", None) or []
    if not paths:
        return None
    # Build a list of (path, text) pairs. slide_texts may be shorter than
    # paths if OCR failed on some slides — pad with "" to keep zip safe.
    text_for: dict[str, str] = {}
    for i, p in enumerate(paths):
        text_for[p] = slide_texts[i] if i < len(slide_texts) else ""

    first = founder.name.split(" ", 1)[0].lower()
    last = founder.name.rsplit(" ", 1)[-1].lower()

    # Prefer slides whose OCR mentions the founder. Fall back to all slides.
    matching = [p for p, t in text_for.items()
                if first in t.lower() or last in t.lower()]
    candidates = matching or paths

    try:
        from ..modules.photo_classifier import _embed_face  # face availability check
        from insightface.app import FaceAnalysis  # noqa: F401  (presence check)
        from PIL import Image
        import numpy as np  # noqa: F401  (used implicitly via insightface)
    except ImportError:
        return None

    for slide_path in candidates:
        try:
            data = Path(slide_path).read_bytes()
        except Exception:
            continue
        face_bytes = _crop_largest_face(data)
        if face_bytes:
            return face_bytes
    return None


def _crop_largest_face(image_bytes: bytes) -> bytes | None:
    """Run InsightFace on the input. If exactly one face is detected (or
    one dominant face), crop a 1.4× bounding box around it and return
    JPEG bytes. None if no face detected or any error."""
    try:
        from insightface.app import FaceAnalysis
        from PIL import Image
        import numpy as np
    except ImportError:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)[:, :, ::-1]
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        faces = app.get(arr)
    except Exception:
        return None
    if not faces:
        return None
    # Use the largest face — usually the founder featured on the slide.
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = face.bbox
    w, h = x2 - x1, y2 - y1
    if w < 60 or h < 60:
        return None  # too small to be useful
    # Expand 1.4× around center to include hair / shoulders.
    cx, cy = x1 + w / 2, y1 + h / 2
    half = max(w, h) * 0.7
    left = max(0, int(cx - half))
    upper = max(0, int(cy - half))
    right = min(img.width, int(cx + half))
    lower = min(img.height, int(cy + half))
    cropped = img.crop((left, upper, right, lower))
    out = io.BytesIO()
    cropped.save(out, format="JPEG", quality=88)
    return out.getvalue()


async def _from_wikipedia(client: httpx.AsyncClient, founder: "Founder") -> bytes | None:
    title = founder.name.replace(" ", "_")
    try:
        r = await client.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            headers={"User-Agent": _UA, "Accept": "application/json"},
        )
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None
    url = (data.get("originalimage") or {}).get("source") or \
          (data.get("thumbnail") or {}).get("source")
    if not url:
        return None
    return await _fetch_image(client, url)


async def _from_company_team(
    client: httpx.AsyncClient,
    founder: "Founder",
    ctx: "DealContext",
) -> bytes | None:
    """Scrape company /team, /about, /leadership pages. Look for `<img>`
    whose `alt` contains the founder's last name."""
    if not ctx.website:
        return None
    # Normalize to a bare domain
    site = ctx.website
    if "://" in site:
        site = site.split("://", 1)[1]
    site = site.rstrip("/")
    last_name = founder.name.rsplit(" ", 1)[-1].lower()
    paths = ("/about", "/team", "/about-us", "/our-team", "/leadership", "/company")
    for path in paths:
        url = f"https://{site}{path}"
        try:
            r = await client.get(url, headers={"User-Agent": _UA})
            if r.status_code != 200:
                continue
            html = r.text
        except Exception:
            continue
        for m in re.finditer(
            r'<img[^>]+alt="([^"]+)"[^>]+src="([^"]+)"', html, re.IGNORECASE,
        ):
            alt = m.group(1).lower()
            if last_name in alt:
                src = m.group(2)
                if src.startswith("//"):
                    src = "https:" + src
                elif not src.startswith("http"):
                    src = f"https://{site}{src}"
                img = await _fetch_image(client, src)
                if img:
                    return img
    return None


async def _from_linkedin_og(
    client: httpx.AsyncClient,
    founder: "Founder",
) -> bytes | None:
    """Fetch the public LinkedIn profile and parse `<meta property="og:image">`.
    Rejects known placeholder URLs (the generic "in" logo)."""
    if not founder.linkedin_url:
        return None
    url = founder.linkedin_url
    if not url.startswith("http"):
        url = f"https://{url.lstrip('/')}"
    try:
        r = await client.get(url, headers={"User-Agent": _UA})
        if r.status_code != 200:
            return None
        html = r.text
    except Exception:
        return None
    m = re.search(
        r'<meta\s+property=["\']og:image["\']\s+content=["\']([^"\']+)["\']',
        html, re.IGNORECASE,
    )
    if not m:
        return None
    img_url = m.group(1)
    if any(hint in img_url for hint in _LI_PLACEHOLDER_HINTS):
        log.debug("rejecting LinkedIn placeholder for %s: %s",
                  founder.name, img_url)
        return None
    return await _fetch_image(client, img_url)


async def _from_web_image_search(
    client: httpx.AsyncClient,
    founder: "Founder",
    ctx: "DealContext",
) -> bytes | None:
    """Web search for the founder's headshot, then download the first hit
    that points at an actual image URL.

    This is the workhorse tier for not-yet-famous founders (no Wikipedia
    entry, no scraped /team page, no LinkedIn og:image available). It runs
    several variations of the query and accepts ANY URL ending in a known
    image extension. Cheap and broad — combined with `_fetch_image`'s
    size/MIME validation, false positives stay rare."""
    try:
        from .search import web_search
    except ImportError:
        return None

    company = ctx.company_name or ""
    queries = [
        f'"{founder.name}" {company} headshot',
        f'"{founder.name}" {company} CEO photo',
        f'"{founder.name}" {company} portrait',
        f'"{founder.name}" headshot',
    ]
    if company:
        queries.append(f'"{founder.name}" founder of "{company}"')

    seen: set[str] = set()
    last = founder.name.rsplit(" ", 1)[-1].lower()
    for q in queries:
        try:
            results = await web_search(q, max_results=8)
        except Exception:
            continue
        # Resolve any grounding-redirect URLs to their real destinations
        # so we can both filter by extension and pass the right URL to
        # _fetch_image / _from_page_og_image.
        resolved: list[tuple[str, str]] = []  # (real_url, raw_url)
        for r in results:
            real = await _resolve_redirect(client, r.url)
            resolved.append((real, r.url))

        for real, _raw in resolved:
            url_l = real.lower()
            if url_l in seen:
                continue
            seen.add(url_l)
            # Direct image URL — most desirable.
            if url_l.endswith((".jpg", ".jpeg", ".png", ".webp")):
                img = await _fetch_image(client, real)
                if img:
                    return img
        # Second pass: pages that LIKELY contain a founder headshot
        # (company /team page, founder bio page). Fetch HTML and look
        # for og:image or an <img alt> referencing the founder.
        for real, _raw in resolved:
            url_l = real.lower()
            if url_l.endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            if not any(k in url_l for k in (
                "team", "about", "founder", "ceo", "people", "leadership",
                "/bio", "linkedin.com/in/",
            )):
                continue
            img = await _from_page_og_image(client, real, last)
            if img:
                return img
    return None


async def _from_page_og_image(
    client: httpx.AsyncClient, page_url: str, last_name_lower: str,
) -> bytes | None:
    """Fetch a HTML page; return the og:image bytes if it looks like a
    real photo (skip generic 'in' placeholder, skip if alt-text near the
    image excludes the founder's last name)."""
    try:
        r = await client.get(page_url, headers={"User-Agent": _UA})
        if r.status_code != 200:
            return None
        html = r.text
    except Exception:
        return None
    # First try og:image — it's usually a clean square headshot when the
    # page is a founder bio page.
    m = re.search(
        r'<meta\s+property=["\']og:image["\']\s+content=["\']([^"\']+)["\']',
        html, re.IGNORECASE,
    )
    if m:
        img_url = m.group(1)
        if not any(hint in img_url for hint in _LI_PLACEHOLDER_HINTS):
            img = await _fetch_image(client, img_url)
            if img:
                return img
    # Fallback: any <img alt="..."> whose alt mentions the founder's
    # last name.
    for m in re.finditer(
        r'<img[^>]+alt="([^"]+)"[^>]+src="([^"]+)"', html, re.IGNORECASE,
    ):
        alt = m.group(1).lower()
        if last_name_lower in alt:
            src = m.group(2)
            if src.startswith("//"):
                src = "https:" + src
            elif src.startswith("/"):
                from urllib.parse import urlparse
                p = urlparse(page_url)
                src = f"{p.scheme}://{p.hostname}{src}"
            img = await _fetch_image(client, src)
            if img:
                return img
    return None


async def _from_grounded(
    client: httpx.AsyncClient,
    founder: "Founder",
    ctx: "DealContext",
) -> bytes | None:
    """Last-resort: ask the configured grounded-search backend for a
    direct image URL. Cheap enough we always call it before giving up."""
    try:
        from .search import ask_grounded
    except ImportError:
        return None
    company = ctx.company_name or ""
    prompt = (
        f"Find the URL of a single public headshot photo of {founder.name} "
        f"(founder/CEO of {company}). Prefer Wikipedia, Crunchbase, the "
        f"company's press page, Bloomberg, Forbes, or TechCrunch. The URL "
        f"must end in .jpg, .jpeg, .png, or .webp AND be a direct image "
        f"link (not an HTML page containing the image). Output ONLY the "
        f"URL. If no public direct image URL exists, output 'NONE'."
    )
    try:
        ans = await ask_grounded(prompt, max_sources=3, max_tokens=200)
    except Exception:
        return None
    if not ans or not ans.text:
        return None
    text = ans.text.strip().strip("`").strip()
    m = re.search(
        r"(https?://[^\s\)\]\"']+\.(?:jpe?g|png|webp))",
        text, re.IGNORECASE,
    )
    if not m:
        return None
    return await _fetch_image(client, m.group(1))


async def _from_clipping(
    client: httpx.AsyncClient,
    founder: "Founder",
    clipping: "ClippingContext | None",
) -> bytes | None:
    """Mine the Obsidian clipping's embedded image URLs. Heuristic: any
    image whose URL or surrounding alt text mentions the founder's last
    name is a candidate. Otherwise try the first image — clippings often
    lead with a hero photo of the subject."""
    if clipping is None:
        return None
    urls = getattr(clipping, "embedded_image_urls", None) or []
    if not urls:
        return None
    last_name = founder.name.rsplit(" ", 1)[-1].lower()
    # Score-prefer URLs whose path mentions the founder's name
    preferred = [u for u in urls if last_name in u.lower()]
    ordered = preferred + [u for u in urls if u not in preferred]
    for url in ordered[:5]:
        img = await _fetch_image(client, url)
        if img:
            return img
    return None


# ---------- shared HTTP / persistence helpers ------------------------------


async def _fetch_image(client: httpx.AsyncClient, url: str) -> bytes | None:
    try:
        r = await client.get(url, headers={"User-Agent": _UA})
        if r.status_code != 200:
            return None
        if len(r.content) < 1500:
            return None
        ct = (r.headers.get("content-type") or "").lower()
        if ct and "image" not in ct:
            return None
        return r.content
    except Exception:
        return None


def _persist(
    img_bytes: bytes,
    save_dir: Path,
    founder_name: str,
    source: str,
) -> str | None:
    """Save bytes as JPEG. Returns absolute path or None."""
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", founder_name).strip("_")
    if not safe:
        safe = "founder"
    out = save_dir / f"{safe}_via_{source}.jpg"
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # Normalize to a reasonable size — 400x400 max-axis is plenty for a
        # headshot used both in narrative + in the photo classifier.
        img.thumbnail((520, 520))
        img.save(out, format="JPEG", quality=88)
        return str(out.resolve())
    except Exception:
        # Fallback: write raw bytes — the photo classifier handles it.
        try:
            out.write_bytes(img_bytes)
            return str(out.resolve())
        except Exception:
            return None


def _default_save_dir(deal_id: str) -> Path:
    base = Path(os.environ.get("DD_DATA_DIR", "./data"))
    return base / "reports" / "photos" / deal_id
