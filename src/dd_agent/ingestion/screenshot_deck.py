"""Pitch-deck screenshot + OCR.

When a clipping contains a link to a hosted deck (Pitch.com, Google Slides,
Figma, Notion, etc.) we open the URL in a headless Chromium via Playwright,
detect whether the deck is gated (sign-in / email form), and if not, capture
full-page screenshots of each slide. Then we OCR each screenshot via Gemini
Vision to produce `deck_text` for the rest of the pipeline.

Always best-effort:
  - If Playwright isn't installed → returns DeckCapture(available=False, …)
  - If the page is gated → returns DeckCapture(available=False, gated=True, …)
  - If OCR fails per-slide → that slide's text is empty but capture continues

We do NOT pass cookies, log in, or click "I agree to the terms" — only public
decks load.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger("dd_agent.screenshot_deck")


@dataclass
class DeckCapture:
    available: bool
    deck_url: str
    text: str = ""                          # concatenated OCR across slides
    screenshot_paths: list[str] = field(default_factory=list)
    # v8: per-slide OCR text aligned with `screenshot_paths`. Lets downstream
    # consumers (e.g. founder-photo deck-slide strategy) match a specific
    # slide PNG to its text content for precision filtering.
    slide_texts: list[str] = field(default_factory=list)
    slide_count: int = 0
    gated: bool = False
    note: str | None = None                 # explanation when available=False


# Markers that indicate the page is asking for credentials / email before
# showing the deck. Order matters — most specific first.
_GATE_HINTS = (
    "sign in to view",
    "log in to view",
    "verify your email",
    "request access",
    "this presentation is private",
    "please enter your email",
    "you don't have access",
    "this content requires login",
    "google docs requires you to sign in",
    "docsend requires",
    "please log in to docsend",
)


# Tunable knobs.
_MAX_SLIDES = int(os.environ.get("DD_DECK_MAX_SLIDES", "30"))
_VIEWPORT = {"width": 1440, "height": 900}
_NAV_TIMEOUT_MS = 25_000
_SETTLE_MS = 2_000


def _shots_dir() -> Path:
    base = Path(os.environ.get("DD_DATA_DIR", "./data")) / "deck_shots"
    base.mkdir(parents=True, exist_ok=True)
    return base


async def capture(deck_url: str, deal_id: str | None = None) -> DeckCapture:
    """Capture and OCR a hosted-deck URL. Always returns a DeckCapture."""
    try:
        from playwright.async_api import async_playwright  # noqa: F401
    except ImportError:
        return DeckCapture(
            available=False, deck_url=deck_url,
            note="playwright not installed (`pip install playwright && playwright install chromium`)",
        )

    out_dir = _shots_dir()
    prefix = (deal_id or _slug(deck_url))[:32]

    try:
        screenshots, gated_reason = await _navigate_and_capture(deck_url, out_dir, prefix)
    except Exception as exc:  # noqa: BLE001
        log.warning("playwright capture failed for %s: %s", deck_url, exc)
        return DeckCapture(
            available=False, deck_url=deck_url,
            note=f"playwright error: {exc}",
        )

    if gated_reason:
        return DeckCapture(
            available=False, deck_url=deck_url, gated=True,
            note=f"deck appears gated — {gated_reason}",
        )
    if not screenshots:
        return DeckCapture(
            available=False, deck_url=deck_url,
            note="capture produced no screenshots (page may have failed to render)",
        )

    text, per_slide = await _ocr_screenshots_async(screenshots)
    return DeckCapture(
        available=True,
        deck_url=deck_url,
        text=text,
        screenshot_paths=[str(p) for p in screenshots],
        slide_texts=per_slide,
        slide_count=len(screenshots),
    )


async def _navigate_and_capture(
    deck_url: str, out_dir: Path, prefix: str,
) -> tuple[list[Path], str | None]:
    """Open the URL in headless Chromium. Returns (screenshots, gated_reason)."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        # Use the Playwright Chromium that's already cached on this Mac.
        browser = await pw.chromium.launch(headless=True)
        try:
            ctx = await browser.new_context(
                viewport=_VIEWPORT,
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
                ),
            )
            page = await ctx.new_page()
            try:
                await page.goto(deck_url, timeout=_NAV_TIMEOUT_MS, wait_until="domcontentloaded")
            except Exception as exc:  # noqa: BLE001
                return [], f"navigation failed: {exc}"

            # Let the slide JS render.
            try:
                await page.wait_for_load_state("networkidle", timeout=8_000)
            except Exception:
                pass
            await page.wait_for_timeout(_SETTLE_MS)

            # Gate detection — search visible page text for known gate phrases.
            try:
                body_text = (await page.inner_text("body")).lower()
            except Exception:
                body_text = ""
            for hint in _GATE_HINTS:
                if hint in body_text:
                    return [], f"detected gate phrase '{hint}'"
            # Also look for password / email input fields as gate proxy.
            email_field = await page.query_selector("input[type='email']")
            password_field = await page.query_selector("input[type='password']")
            if email_field or password_field:
                # Could still be a "subscribe to our newsletter" widget — only
                # treat as gate if the rest of the body is suspiciously short.
                if len(body_text.strip()) < 400:
                    return [], "page shows only a sign-in form"

            # Capture per host.
            host = urlparse(deck_url).netloc.lower()
            shots = await _capture_strategy(page, host, out_dir, prefix)
            return shots, None
        finally:
            await browser.close()


async def _capture_strategy(page, host: str, out_dir: Path, prefix: str) -> list[Path]:
    """Choose a per-host capture strategy. Pitch and Google Slides expose
    forward arrows; Figma/Notion/docsend just scroll. Fall back to full-page
    screenshot when slide navigation isn't reliable."""
    if "pitch.com" in host or "pitch.io" in host:
        return await _capture_via_arrow_keys(page, out_dir, prefix)
    if "docs.google.com/presentation" in (host + page.url):
        return await _capture_via_arrow_keys(page, out_dir, prefix)
    if "figma.com" in host:
        # Figma presentations use right-arrow too.
        return await _capture_via_arrow_keys(page, out_dir, prefix)
    if "docsend.com" in host:
        return await _capture_via_arrow_keys(page, out_dir, prefix)
    if "gamma.app" in host:
        return await _capture_via_scroll(page, out_dir, prefix)
    if "notion.so" in host:
        return await _capture_via_scroll(page, out_dir, prefix)
    if "slideshare.net" in host:
        return await _capture_via_scroll(page, out_dir, prefix)
    if "slides.com" in host:
        return await _capture_via_arrow_keys(page, out_dir, prefix)
    return await _capture_via_scroll(page, out_dir, prefix)


async def _capture_via_arrow_keys(page, out_dir: Path, prefix: str) -> list[Path]:
    """Click on the page to focus, then press Right N times capturing between."""
    shots: list[Path] = []
    await page.click("body")
    for i in range(_MAX_SLIDES):
        out = out_dir / f"{prefix}-slide-{i+1:02d}.png"
        try:
            await page.screenshot(path=str(out), full_page=False)
        except Exception as exc:  # noqa: BLE001
            log.debug("screenshot %d failed: %s", i + 1, exc)
            break
        shots.append(out)
        # Hash the file briefly so we can detect "stopped advancing"
        if i >= 1 and shots[-1].stat().st_size == shots[-2].stat().st_size:
            # Same size two in a row is usually end-of-deck. Quick check.
            if _files_equal(shots[-1], shots[-2]):
                shots.pop()
                break
        await page.keyboard.press("ArrowRight")
        await page.wait_for_timeout(700)
    return shots


async def _capture_via_scroll(page, out_dir: Path, prefix: str) -> list[Path]:
    """Single full-page screenshot of a scroll-style deck."""
    out = out_dir / f"{prefix}-fullpage.png"
    try:
        await page.screenshot(path=str(out), full_page=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("full-page screenshot failed: %s", exc)
        return []
    return [out]


def _files_equal(a: Path, b: Path) -> bool:
    try:
        return a.read_bytes() == b.read_bytes()
    except OSError:
        return False


def _slug(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:40] or "deck"


# --- Gemini Vision OCR ------------------------------------------------------


async def _ocr_screenshots_async(paths: list[Path]) -> tuple[str, list[str]]:
    """OCR every screenshot via Gemini Vision. Returns (concatenated_text,
    per_slide_texts) where per_slide_texts is aligned 1:1 with `paths` —
    slides whose OCR failed or returned nothing become "" in that list so
    downstream consumers can still index by slide path."""
    if not os.environ.get("GEMINI_API_KEY"):
        return "", ["" for _ in paths]
    chunks = await asyncio.gather(*(_ocr_one(p) for p in paths))
    per_slide: list[str] = []
    blocks: list[str] = []
    for i, (p, txt) in enumerate(zip(paths, chunks), 1):
        cleaned = (txt or "").strip()
        per_slide.append(cleaned)
        if cleaned:
            blocks.append(f"### Slide {i} ({p.name})\n\n{cleaned}")
    return "\n\n".join(blocks), per_slide


async def _ocr_one(path: Path) -> str:
    """Send one screenshot to Gemini Vision and return the extracted text."""
    import httpx
    try:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        log.warning("could not read %s: %s", path, exc)
        return ""

    model = os.environ.get("DD_GEMINI_VISION_MODEL", "gemini-2.5-flash")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={os.environ['GEMINI_API_KEY']}"
    )
    body = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/png", "data": b64}},
                {"text": (
                    "Extract every word visible on this slide. Preserve list "
                    "structure. If the slide is purely an image, describe what "
                    "you see in 1-2 sentences. Output plain text only — no "
                    "markdown headers, no commentary."
                )},
            ],
        }],
    }
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            r = await client.post(url, json=body)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("Gemini Vision OCR failed for %s: %s", path.name, exc)
        return ""
    cand = (data.get("candidates") or [{}])[0]
    parts = cand.get("content", {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts).strip()
