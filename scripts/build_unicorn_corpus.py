"""Build data/unicorn_founders.parquet from data/unicorn_founders_seed.json.

For each founder:
  1. Fetch the photo at photo_url
  2. Run InsightFace buffalo_l → 512-d ArcFace embedding
  3. Ask GPT-5.5 (vision) to rate the photo on 5 traits, 1-5 scale
  4. Append a row to the parquet

Run with:
  python scripts/build_unicorn_corpus.py [--limit N] [--no-llm-traits]

If a photo URL is broken, the founder is skipped (logged). If InsightFace is
not installed, the script exits with a clear message. If --no-llm-traits is
passed, traits default to 3.0 for all founders (the orchestrator will still
work but trait scores will be uninformative).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import sys
from pathlib import Path

import httpx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_unicorn_corpus")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SEED_PATH = DATA_DIR / "unicorn_founders_seed.json"
OUT_PATH = DATA_DIR / "unicorn_founders.parquet"

TRAITS = ("resilience", "intensity", "warmth", "presentation_polish", "energy")


def _load_seed() -> list[dict]:
    return json.loads(SEED_PATH.read_text()).get("founders", [])


UA = "DD-Agent/0.1 (https://github.com/semenka/DD; corpus build)"


async def _resolve_via_wiki(client: httpx.AsyncClient, name: str) -> str | None:
    """Resolve a person name to their Wikipedia originalimage URL."""
    title = name.replace(" ", "_")
    try:
        r = await client.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            headers={"User-Agent": UA, "Accept": "application/json"},
        )
        if r.status_code != 200:
            return None
        data = r.json()
        orig = (data.get("originalimage") or {}).get("source")
        if orig:
            return orig
        thumb = (data.get("thumbnail") or {}).get("source")
        return thumb
    except Exception:
        return None


async def _fetch_photo(client: httpx.AsyncClient, url: str | None, name: str) -> bytes | None:
    """Fetch a photo. Try the provided URL first, then resolve via Wikipedia REST API."""
    candidates: list[str] = []
    if url:
        candidates.append(url)
    resolved = await _resolve_via_wiki(client, name)
    if resolved and resolved not in candidates:
        candidates.append(resolved)
    if not candidates:
        return None
    for candidate in candidates:
        try:
            r = await client.get(candidate, headers={"User-Agent": UA})
        except Exception as exc:
            log.debug("fetch error %s: %s", candidate, exc)
            continue
        if r.status_code == 200 and len(r.content) >= 500:
            log.debug("fetched %s (%d bytes)", candidate, len(r.content))
            return r.content
        log.debug("fetch %s returned %s (%d bytes)", candidate, r.status_code, len(r.content))
    return None


def _embed(image_bytes: bytes, app) -> np.ndarray | None:
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)[:, :, ::-1]
        faces = app.get(arr)
    except Exception as exc:
        log.warning("embed failed: %s", exc)
        return None
    if not faces:
        return None
    face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    e = face.embedding.astype(np.float32)
    n = np.linalg.norm(e)
    return (e / n) if n > 0 else e


async def _llm_traits(name: str, image_bytes: bytes) -> dict[str, float]:
    """Score 5 traits via GPT-5.5 vision (codex CLI). Returns 3.0 across the
    board on any failure."""
    import sys
    import tempfile
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from dd_agent.modules._llm import codex_exec, CodexUnavailableError, FAST_MODEL
    except ImportError:
        return {t: 3.0 for t in TRAITS}

    prompt = (
        f"You are scoring a single photo of {name} (a founder of a 1B+ company). "
        "Rate the image on 5 traits, each 1-5 integer scale, based ONLY on what "
        "is visible: resilience, intensity, warmth, presentation_polish, energy. "
        "Output ONLY a JSON object with those 5 keys and integer 1-5 values. "
        "No preamble, no markdown fences."
    )
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        text = await codex_exec(prompt, model=FAST_MODEL, images=[tmp_path], timeout=120.0)
    except CodexUnavailableError:
        log.warning("codex CLI not installed — trait scoring defaults to 3.0 for %s", name)
        return {t: 3.0 for t in TRAITS}
    except Exception as exc:
        log.warning("LLM trait scoring failed for %s: %s", name, exc)
        return {t: 3.0 for t in TRAITS}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    try:
        s = text.strip().strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
        data = json.loads(s)
        return {t: float(data.get(t, 3.0)) for t in TRAITS}
    except Exception:
        return {t: 3.0 for t in TRAITS}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-llm-traits", action="store_true")
    args = ap.parse_args()

    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        log.error(
            "insightface not installed. Run: uv pip install -e \".[photo]\""
        )
        sys.exit(1)
    import pandas as pd

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    seed = _load_seed()
    if args.limit:
        seed = seed[: args.limit]

    rows: list[dict] = []
    async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
        for f in seed:
            log.info("processing %s (%s)", f["name"], f["company"])
            img = await _fetch_photo(client, f.get("photo_url"), f["name"])
            if not img:
                log.warning("  skipped — could not fetch photo")
                continue
            emb = _embed(img, app)
            if emb is None:
                log.warning("  skipped — no face detected")
                continue
            traits = (
                {t: 3.0 for t in TRAITS}
                if args.no_llm_traits
                else await _llm_traits(f["name"], img)
            )
            rows.append({
                "founder_id": f["founder_id"],
                "name": f["name"],
                "company": f["company"],
                "photo_url": f.get("photo_url"),
                "embedding": emb.tolist(),
                **traits,
            })
            await asyncio.sleep(0.3)  # be polite to Wikimedia

    if not rows:
        log.error("no founders successfully embedded — parquet not written")
        sys.exit(2)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    log.info("wrote %s (%d founders)", OUT_PATH, len(df))


if __name__ == "__main__":
    asyncio.run(main())
