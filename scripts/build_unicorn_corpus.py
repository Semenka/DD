"""Build data/unicorn_founders.parquet.

Pipeline:
  1. Generate (or load from cache) a list of ~500 founder-led companies
     across three cohorts:
       - public_sp500_nasdaq: S&P 500 + Nasdaq 100 founder-led companies
       - yc_top_100:           Y Combinator top-100 most successful
       - unicorn_private:      notable 1B+ private founder-led companies
     The list is generated via Perplexity (`ask_grounded`) at first run and
     cached at data/unicorn_founders_list.json for reproducible offline builds.
  2. For each founder: resolve a photo via Wikipedia REST API first, then the
     company's `/about` or `/team` page, then a fallback web search.
  3. Run InsightFace buffalo_l → 512-d ArcFace embedding. Skip if no face.
  4. Trait labeling via Gemini Vision (batched, cheap) on 5 traits.
  5. Persist a row per founder with: {founder_id, name, company, cohort,
     photo_url, embedding, resilience, intensity, warmth,
     presentation_polish, energy}.

Run:
  python scripts/build_unicorn_corpus.py
  python scripts/build_unicorn_corpus.py --limit 50          # smoke
  python scripts/build_unicorn_corpus.py --no-llm-traits     # 3.0 for all
  python scripts/build_unicorn_corpus.py --refresh-list      # regenerate from Perplexity
  python scripts/build_unicorn_corpus.py --cohort public_sp500_nasdaq

Existing parquet rows are preserved on incremental runs unless --rebuild.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import re
import sys
from pathlib import Path

import httpx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_unicorn_corpus")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
LIST_CACHE = DATA_DIR / "unicorn_founders_list.json"
OUT_PATH = DATA_DIR / "unicorn_founders.parquet"
LEGACY_SEED = DATA_DIR / "unicorn_founders_seed.json"  # fallback

TRAITS = ("resilience", "intensity", "warmth", "presentation_polish", "energy")
COHORTS = ("public_sp500_nasdaq", "yc_top_100", "unicorn_private")

UA = "DD-Agent/0.1 (https://github.com/Semenka/DD; corpus build)"


# -------------------- founder list generation ------------------------------


_GEN_PROMPT = """List founder-led companies for a startup-pattern-matching \
corpus. I need approximately 500 founders total, split across three cohorts:

1. **public_sp500_nasdaq** (target: ~70 founders): companies currently in the \
S&P 500 or Nasdaq 100 where the founder is still a major shareholder, board \
member, or active operator (CEO/Chair/Exec Chair). Examples: Mark Zuckerberg \
(Meta), Jensen Huang (NVIDIA), Marc Benioff (Salesforce), Eric Yuan (Zoom), \
Reed Hastings (Netflix), Larry Ellison (Oracle), Brian Chesky (Airbnb).

2. **yc_top_100** (target: ~120 founders): founders of Y Combinator's top-100 \
companies by current valuation. Examples: Patrick Collison + John Collison \
(Stripe), Brian Chesky + Joe Gebbia + Nathan Blecharczyk (Airbnb), Drew Houston \
(Dropbox), Henrique Dubugras + Pedro Franceschi (Brex), Tony Xu (DoorDash), \
Cristina Cordova (former Notion, etc.), Sam Altman (OpenAI).

3. **unicorn_private** (target: ~310 founders): notable 1B+ private companies \
where the founder is currently CEO. Examples: Anthropic (Dario + Daniela \
Amodei), SpaceX (Elon Musk), Databricks (Ali Ghodsi + Matei Zaharia), \
Stripe (already counted under YC), Canva (Melanie Perkins + Cliff Obrecht), \
Revolut (Nikolay Storonsky), Deel (Alex Bouaziz), Ramp (Eric Glyman + Karim \
Atiyeh + Gene Lee), Figma (Dylan Field).

Output ONLY a single JSON object with this exact shape (no preamble, no \
markdown fences, start with `{`):

{
  "founders": [
    {
      "founder_id": "patrick_collison",
      "name": "Patrick Collison",
      "company": "Stripe",
      "cohort": "yc_top_100",
      "company_domain": "stripe.com"
    }
  ]
}

Rules:
- founder_id: lowercase, snake_case, ascii-only. Use first_last format.
- name: original spelling including diacritics.
- company: short brand name, no "Inc" / "Corp".
- cohort: one of public_sp500_nasdaq | yc_top_100 | unicorn_private.
- company_domain: just the bare domain (no protocol, no path).
- If a founder qualifies for multiple cohorts, pick the most prominent ONE.
- Multi-founder companies: include each founder as a separate row.
- Target the cohort sizes above; you may overshoot by ~20%.
- Output only the JSON object."""


async def generate_founder_list() -> list[dict]:
    """Generate the 500-founder list via Perplexity, one cohort at a time.

    A single Perplexity call hits its output-token cap well before 500 entries.
    Splitting per cohort (~70-310 each) keeps each call within budget AND lets
    us specify the cohort label deterministically rather than relying on the
    LLM to assign it correctly."""
    from dd_agent.data_sources.search import ask_grounded

    cohorts = (
        ("public_sp500_nasdaq",
         "currently in the S&P 500 or Nasdaq 100 where the founder is still a major "
         "shareholder, board member, or active operator (CEO/Chair/Exec Chair). "
         "Examples: Mark Zuckerberg (Meta), Jensen Huang (NVIDIA), Marc Benioff (Salesforce), "
         "Eric Yuan (Zoom), Reed Hastings (Netflix), Larry Ellison (Oracle), Brian Chesky (Airbnb).",
         "~80 founders"),
        ("yc_top_100",
         "founders of Y Combinator's top-100 alumni companies by current valuation. "
         "Examples: Patrick Collison and John Collison (Stripe), Brian Chesky and Joe Gebbia "
         "and Nathan Blecharczyk (Airbnb), Drew Houston (Dropbox), Henrique Dubugras and "
         "Pedro Franceschi (Brex), Tony Xu (DoorDash), Sam Altman (OpenAI).",
         "~120 founders"),
        ("unicorn_private",
         "notable 1B+ private companies (NOT in S&P 500 or Nasdaq 100, NOT primarily YC-known) "
         "where the founder is currently CEO. "
         "Examples: SpaceX (Elon Musk), Databricks (Ali Ghodsi, Matei Zaharia), "
         "Canva (Melanie Perkins, Cliff Obrecht), Revolut (Nikolay Storonsky), "
         "Deel (Alex Bouaziz), Anthropic (Dario Amodei, Daniela Amodei), "
         "Figma (Dylan Field).",
         "~150 founders"),
    )

    all_founders: list[dict] = []
    for cohort, description, target_size in cohorts:
        prompt = _build_cohort_prompt(cohort, description, target_size)
        log.info("generating founder list for cohort %s (target %s)...",
                 cohort, target_size)
        try:
            # Bump max_tokens — each founder is ~30 tokens, target ~150 = ~4500.
            ans = await ask_grounded(prompt, max_sources=3, max_tokens=8000)
        except Exception as exc:
            log.warning("cohort %s ask_grounded raised: %s", cohort, exc)
            continue
        if ans is None:
            log.warning("cohort %s: no grounded answer (no Perplexity/Gemini key?)", cohort)
            continue
        founders = _parse_founder_response(ans.text, default_cohort=cohort)
        if not founders:
            log.warning("cohort %s: parser yielded 0 founders", cohort)
            continue
        log.info("cohort %s: %d founders", cohort, len(founders))
        all_founders.extend(founders)

    # Dedup by founder_id (some founders appear in multiple lists).
    seen: set[str] = set()
    deduped: list[dict] = []
    for f in all_founders:
        fid = f.get("founder_id")
        if not fid or fid in seen:
            continue
        seen.add(fid)
        deduped.append(f)
    log.info("total: %d founders across all cohorts (deduped from %d)",
             len(deduped), len(all_founders))
    return deduped


def _build_cohort_prompt(cohort: str, description: str, target_size: str) -> str:
    """Per-cohort prompt — much shorter than the original combined prompt so
    Perplexity has token headroom to actually emit the full list."""
    return (
        f'List founder-led companies matching this description ({target_size}): {description}\n\n'
        'Output ONLY a single JSON object, no preamble, no markdown fences. '
        f'Use cohort="{cohort}" for every entry. Schema:\n\n'
        '{\n'
        '  "founders": [\n'
        '    {\n'
        '      "founder_id": "first_last",   // lowercase, snake_case, ascii\n'
        '      "name": "First Last",         // original spelling, diacritics ok\n'
        '      "company": "Company name",    // short brand, no Inc/Corp\n'
        f'      "cohort": "{cohort}",\n'
        '      "company_domain": "company.com" // bare domain, no protocol/path\n'
        '    }\n'
        '  ]\n'
        '}\n\n'
        "Multi-founder companies: include each founder as a separate row. "
        f"Aim for {target_size}. Output ONLY the JSON object — start with `{{`."
    )


def _parse_founder_response(text: str, default_cohort: str) -> list[dict]:
    """Robust JSON parser for the founder-list response.

    Handles: markdown fences, leading/trailing prose, and truncation mid-array
    (clips the last incomplete row + closes the JSON braces)."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    # Find the outermost {...} block.
    start = s.find("{")
    if start < 0:
        return []
    s = s[start:]

    data = _try_parse_with_repair(s)
    if not data:
        log.error("could not parse JSON even after repair (first 300 chars): %s", s[:300])
        return []

    founders = data.get("founders", []) if isinstance(data, dict) else []
    out: list[dict] = []
    for row in founders:
        if not isinstance(row, dict) or not row.get("name"):
            continue
        row.setdefault("cohort", default_cohort)
        row.setdefault("founder_id", _slug_name(row["name"]))
        out.append(row)
    return out


def _try_parse_with_repair(s: str) -> dict | None:
    """Try json.loads; on failure, clip the last incomplete row and close braces."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Truncation repair: find the last `}` that closes a founders array entry,
    # then close the array + outer object.
    last_obj_end = s.rfind("}")
    while last_obj_end > 0:
        candidate = s[: last_obj_end + 1] + "\n]\n}"
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            last_obj_end = s.rfind("}", 0, last_obj_end)
    return None


def _slug_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return s[:60] or "unknown"


async def load_or_generate_list(refresh: bool = False) -> list[dict]:
    """Cached founder-list loader. Falls back to legacy seed if no source."""
    if LIST_CACHE.exists() and not refresh:
        log.info("loading cached founder list from %s", LIST_CACHE)
        return json.loads(LIST_CACHE.read_text()).get("founders", [])
    founders = await generate_founder_list()
    if founders:
        LIST_CACHE.write_text(json.dumps({"founders": founders}, indent=2, ensure_ascii=False))
        log.info("cached %d founders to %s", len(founders), LIST_CACHE)
        return founders
    # Final fallback: legacy seed
    if LEGACY_SEED.exists():
        log.warning("falling back to legacy seed at %s", LEGACY_SEED)
        legacy = json.loads(LEGACY_SEED.read_text()).get("founders", [])
        # Backfill cohort field
        for row in legacy:
            row.setdefault("cohort", "unicorn_private")
        return legacy
    return []


# -------------------- photo resolution -------------------------------------


async def resolve_photo(client: httpx.AsyncClient, founder: dict) -> bytes | None:
    """Try multiple sources to find a usable founder headshot.

    1. Wikipedia REST API summary endpoint (originalimage)
    2. Company `/about`, `/team`, `/leadership` etc. for an `<img>` whose
       alt-text matches the founder's last name
    3. Web search for "{name} headshot"
    4. Grounded LLM lookup (cascades through openclaw→perplexity→gemini per
       data_sources.search.ask_grounded) for an explicit photo URL
    """
    name = founder["name"]
    # 1) Wikipedia
    img = await _wikipedia_photo(client, name)
    if img:
        return img
    # 2) Company team page
    if founder.get("company_domain"):
        img = await _company_team_photo(client, founder)
        if img:
            return img
    # 3) Web search fallback
    img = await _web_search_photo(client, founder)
    if img:
        return img
    # 4) Last-resort: ask a grounded LLM for the URL of a public photo. Very
    #    effective for founders whose Wikipedia page redirects and whose
    #    company site is gated/SPA. Routes through the configured search
    #    cascade (default openclaw→gemini, so quota-safe).
    return await _grounded_photo_lookup(client, founder)


async def _grounded_photo_lookup(client: httpx.AsyncClient, founder: dict) -> bytes | None:
    """Ask the configured grounded-search backend for a public photo URL."""
    try:
        from dd_agent.data_sources.search import ask_grounded
    except ImportError:
        return None
    name = founder["name"]
    company = founder.get("company") or ""
    prompt = (
        f"Find the URL of a single public headshot photo of {name} "
        f"(the founder/CEO of {company}). Prefer Wikipedia, Crunchbase, "
        f"the company's official press page, Bloomberg, Forbes, or "
        f"TechCrunch. The URL must end in .jpg, .jpeg, .png, or .webp "
        f"AND be a direct image link (not an HTML page that contains the "
        f"image). Output ONLY the URL, nothing else. If no public direct "
        f"image URL exists, output exactly 'NONE'."
    )
    try:
        ans = await ask_grounded(prompt, max_sources=3, max_tokens=200)
    except Exception:
        return None
    if not ans or not ans.text:
        return None
    text = ans.text.strip().strip("`").strip()
    # Extract the first http(s) URL ending in an image extension.
    m = re.search(
        r"(https?://[^\s\)\]\"']+\.(?:jpe?g|png|webp))",
        text, re.IGNORECASE,
    )
    if not m:
        return None
    return await _fetch_image(client, m.group(1))


async def _wikipedia_photo(client: httpx.AsyncClient, name: str) -> bytes | None:
    title = name.replace(" ", "_")
    try:
        r = await client.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            headers={"User-Agent": UA, "Accept": "application/json"},
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


async def _company_team_photo(client: httpx.AsyncClient, founder: dict) -> bytes | None:
    """Try `/about` and `/team` on the company's domain; if an img alt or
    nearby text contains the founder's last name, grab it."""
    domain = founder["company_domain"]
    last_name = founder["name"].rsplit(" ", 1)[-1].lower()
    for path in ("/about", "/team", "/about-us", "/our-team", "/leadership"):
        url = f"https://{domain}{path}"
        try:
            r = await client.get(url, headers={"User-Agent": UA}, timeout=12.0)
            if r.status_code != 200:
                continue
            html = r.text
        except Exception:
            continue
        # Look for img tags whose alt mentions the founder.
        for m in re.finditer(r'<img[^>]+alt="([^"]+)"[^>]+src="([^"]+)"', html, re.IGNORECASE):
            alt = m.group(1).lower()
            if last_name in alt:
                src = m.group(2)
                if not src.startswith("http"):
                    src = f"https://{domain}{src}"
                img = await _fetch_image(client, src)
                if img:
                    return img
    return None


async def _web_search_photo(client: httpx.AsyncClient, founder: dict) -> bytes | None:
    """Last-resort: search for a headshot URL."""
    try:
        from dd_agent.data_sources.search import web_search
    except ImportError:
        return None
    domain = founder.get("company_domain", "")
    q = f'"{founder["name"]}" headshot' + (f" site:{domain}" if domain else "")
    try:
        results = await web_search(q, max_results=5)
    except Exception:
        return []
    for r in results:
        if any(ext in r.url.lower() for ext in (".jpg", ".jpeg", ".png", ".webp")):
            img = await _fetch_image(client, r.url)
            if img:
                return img
    return None


async def _fetch_image(client: httpx.AsyncClient, url: str) -> bytes | None:
    try:
        r = await client.get(url, headers={"User-Agent": UA}, timeout=20.0,
                             follow_redirects=True)
        if r.status_code != 200:
            return None
        # Reject anything tiny
        if len(r.content) < 1500:
            return None
        # Reject obvious non-image content
        ct = r.headers.get("content-type", "")
        if ct and "image" not in ct.lower():
            return None
        return r.content
    except Exception:
        return None


# -------------------- face embedding + trait labeling ----------------------


def _embed(image_bytes: bytes, app) -> np.ndarray | None:
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)[:, :, ::-1]
        faces = app.get(arr)
    except Exception as exc:
        log.debug("embed failed: %s", exc)
        return None
    if not faces:
        return None
    face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    e = face.embedding.astype(np.float32)
    n = np.linalg.norm(e)
    return (e / n) if n > 0 else e


async def _gemini_traits(name: str, company: str, image_bytes: bytes) -> dict[str, float]:
    """Score 5 traits via Gemini Vision. Cheaper than codex and we already have the key."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {t: 3.0 for t in TRAITS}
    model = os.environ.get("DD_GEMINI_VISION_MODEL", "gemini-2.5-flash")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    b64 = base64.b64encode(image_bytes).decode("ascii")
    prompt = (
        f"You are scoring a single photo of {name}, a founder of {company}. "
        "Rate the image 1-5 (integer) on each of: resilience, intensity, warmth, "
        "presentation_polish, energy. Base your scores ONLY on what is visible "
        "in the photo — facial expression, framing, attire, gaze, posture. "
        'Output ONLY this exact JSON: {"resilience":N,"intensity":N,"warmth":N,'
        '"presentation_polish":N,"energy":N}'
    )
    body = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                {"text": prompt},
            ],
        }],
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(url, json=body)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        log.debug("Gemini trait scoring failed for %s: %s", name, exc)
        return {t: 3.0 for t in TRAITS}
    cand = (data.get("candidates") or [{}])[0]
    parts = cand.get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    text = text.strip("`")
    if text.lower().startswith("json"):
        text = text[4:].lstrip()
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {t: 3.0 for t in TRAITS}
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return {t: 3.0 for t in TRAITS}
    return {t: float(obj.get(t, 3.0)) for t in TRAITS}


# -------------------- main -------------------------------------------------


async def main_async(args) -> int:
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        log.error('insightface not installed. Run: uv pip install -e ".[photo]"')
        return 1
    import pandas as pd

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    founders = await load_or_generate_list(refresh=args.refresh_list)
    if args.cohort:
        founders = [f for f in founders if f.get("cohort") == args.cohort]
        log.info("filtered to cohort %s: %d founders", args.cohort, len(founders))
    if args.limit:
        founders = founders[: args.limit]
    if not founders:
        log.error("no founders to process")
        return 2

    # Load existing parquet so we can skip already-embedded founders.
    existing: dict[str, dict] = {}
    if OUT_PATH.exists() and not args.rebuild:
        df = pd.read_parquet(OUT_PATH)
        for r in df.to_dict("records"):
            # Backfill cohort on legacy rows that predate the cohort column.
            r.setdefault("cohort", "unicorn_private")
            existing[r["founder_id"]] = r
        log.info("loaded %d existing rows from %s", len(existing), OUT_PATH)

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    rows: list[dict] = list(existing.values())
    fetched = 0
    skipped = 0
    async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
        for i, f in enumerate(founders, 1):
            fid = f["founder_id"]
            if fid in existing and not args.rebuild:
                skipped += 1
                continue
            log.info("[%d/%d] %s (%s) — %s",
                     i, len(founders), f["name"], f["company"], f.get("cohort", "?"))
            img = await resolve_photo(client, f)
            if not img:
                log.warning("  could not fetch photo")
                continue
            emb = _embed(img, app)
            if emb is None:
                log.warning("  no face detected")
                continue
            if args.no_llm_traits:
                traits = {t: 3.0 for t in TRAITS}
            else:
                traits = await _gemini_traits(f["name"], f["company"], img)
            rows.append({
                "founder_id": fid,
                "name": f["name"],
                "company": f["company"],
                "cohort": f.get("cohort", "unicorn_private"),
                "photo_url": f.get("photo_url"),
                "embedding": emb.tolist(),
                **traits,
            })
            fetched += 1
            await asyncio.sleep(0.25)  # be polite

    if not rows:
        log.error("no founders successfully embedded")
        return 3

    df = pd.DataFrame(rows)
    # Dedupe in case --rebuild was off and we re-added the same id.
    df = df.drop_duplicates(subset=["founder_id"], keep="last")
    df.to_parquet(OUT_PATH, index=False)
    log.info("wrote %s — %d founders (%d new this run, %d skipped existing)",
             OUT_PATH, len(df), fetched, skipped)
    if "cohort" in df.columns:
        log.info("cohort breakdown: %s", df["cohort"].value_counts().to_dict())
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N founders.")
    ap.add_argument("--no-llm-traits", action="store_true",
                    help="Skip Gemini Vision trait scoring; use 3.0 for all traits.")
    ap.add_argument("--refresh-list", action="store_true",
                    help="Regenerate the founder list from Perplexity even if cached.")
    ap.add_argument("--cohort", choices=COHORTS, default=None,
                    help="Process only one cohort.")
    ap.add_argument("--rebuild", action="store_true",
                    help="Re-embed all founders, overwriting existing rows.")
    args = ap.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
