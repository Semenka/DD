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
import base64
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


async def _fetch_photo(url: str) -> bytes | None:
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "DD-Agent corpus build"})
            if r.status_code != 200 or len(r.content) < 500:
                return None
            return r.content
    except Exception as exc:
        log.warning("fetch failed %s: %s", url, exc)
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
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {t: 3.0 for t in TRAITS}
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return {t: 3.0 for t in TRAITS}

    client = AsyncOpenAI(api_key=api_key)
    model = os.environ.get("DD_MODEL_FAST", os.environ.get("DD_MODEL", "gpt-5.5"))
    img_b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    prompt = (
        f"You are scoring a single photo of {name} (a founder of a 1B+ company). "
        "Rate the image on 5 traits, each 1-5 scale, based ONLY on what is visible: "
        "resilience, intensity, warmth, presentation_polish, energy. "
        "Output ONLY a JSON object: {\"resilience\": int, \"intensity\": int, "
        "\"warmth\": int, \"presentation_polish\": int, \"energy\": int}."
    )
    try:
        resp = await client.chat.completions.create(
            model=model,
            max_completion_tokens=200,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                        }},
                    ],
                }
            ],
        )
    except Exception as exc:
        log.warning("LLM trait scoring failed for %s: %s", name, exc)
        return {t: 3.0 for t in TRAITS}
    text = resp.choices[0].message.content or ""
    try:
        data = json.loads(text.strip().strip("`").replace("json\n", "", 1))
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
    for f in seed:
        log.info("processing %s (%s)", f["name"], f["company"])
        img = await _fetch_photo(f["photo_url"])
        if not img:
            log.warning("  skipped — could not fetch photo")
            continue
        emb = _embed(img, app)
        if emb is None:
            log.warning("  skipped — no face detected")
            continue
        traits = {t: 3.0 for t in TRAITS} if args.no_llm_traits else await _llm_traits(f["name"], img)
        rows.append({
            "founder_id": f["founder_id"],
            "name": f["name"],
            "company": f["company"],
            "photo_url": f["photo_url"],
            "embedding": emb.tolist(),
            **traits,
        })

    if not rows:
        log.error("no founders successfully embedded — parquet not written")
        sys.exit(2)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    log.info("wrote %s (%d founders)", OUT_PATH, len(df))


if __name__ == "__main__":
    asyncio.run(main())
