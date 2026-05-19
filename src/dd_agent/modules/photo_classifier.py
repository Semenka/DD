"""Founder photo classifier.

Pipeline:
  1. Load seed corpus from data/unicorn_founders.parquet (founder_id, company,
     photo_url, embedding[512], trait scores).
  2. For an input founder photo, compute the 512-d ArcFace embedding via
     InsightFace (buffalo_l).
  3. k-NN over cosine similarity → top-N nearest unicorn-founder matches.
  4. Trait scores = weighted average (by similarity) of nearest-5 neighbors'
     trait scores across resilience / intensity / warmth / presentation_polish / energy.

The module gracefully degrades:
  - If InsightFace is not installed, returns empty results without raising.
  - If the seed corpus is empty, returns empty results.
  - The orchestrator treats this section as best-effort.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TRAITS = ("resilience", "intensity", "warmth", "presentation_polish", "energy")


@dataclass
class FounderMatch:
    founder_id: str
    company: str
    photo_url: str | None
    similarity: float
    cohort: str | None = None   # public_sp500_nasdaq | yc_top_100 | unicorn_private
    traits: dict[str, float] = field(default_factory=dict)


@dataclass
class PhotoAnalysis:
    founder_name: str
    photo_source: str
    nearest: list[FounderMatch]
    trait_scores: dict[str, float]
    available: bool
    cohort_breakdown: dict[str, int] = field(default_factory=dict)  # cohort → count in nearest-k
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "founder_name": self.founder_name,
            "photo_source": self.photo_source,
            "nearest": [m.__dict__ for m in self.nearest],
            "trait_scores": self.trait_scores,
            "cohort_breakdown": self.cohort_breakdown,
            "available": self.available,
            "note": self.note,
        }


def _corpus_path() -> Path:
    return Path(os.environ.get("DD_DATA_DIR", "./data")) / "unicorn_founders.parquet"


def _load_corpus() -> "tuple[list[dict], np.ndarray] | tuple[list, np.ndarray]":
    path = _corpus_path()
    if not path.exists():
        return [], np.zeros((0, 512), dtype=np.float32)
    try:
        import pandas as pd
        df = pd.read_parquet(path)
    except Exception:
        return [], np.zeros((0, 512), dtype=np.float32)
    if df.empty:
        return [], np.zeros((0, 512), dtype=np.float32)
    rows = df.to_dict("records")
    embeddings = np.stack([np.asarray(r["embedding"], dtype=np.float32) for r in rows])
    return rows, embeddings


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _embed_face(image_bytes: bytes) -> np.ndarray | None:
    """Run InsightFace buffalo_l on a single image. Returns 512-d embedding or None."""
    try:
        from insightface.app import FaceAnalysis
        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)[:, :, ::-1]  # PIL RGB → cv2 BGR
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        faces = app.get(arr)
    except Exception:
        return None
    if not faces:
        return None
    face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])  # largest detected face
    return _normalize(face.embedding.astype(np.float32))


async def _fetch_image(url: str) -> bytes | None:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "DD-Agent"})
            r.raise_for_status()
            return r.content
    except Exception:
        return None


async def analyze_founder_photo(
    *,
    founder_name: str,
    photo_url: str | None = None,
    photo_bytes: bytes | None = None,
    k: int = 10,
) -> PhotoAnalysis:
    """Run the full pipeline. Returns PhotoAnalysis with available=False if any
    step fails (no exception)."""
    if os.environ.get("DD_ENABLE_PHOTO_CLASSIFIER", "1") == "0":
        return PhotoAnalysis(
            founder_name=founder_name,
            photo_source=photo_url or "",
            nearest=[],
            trait_scores={t: 0.0 for t in TRAITS},
            available=False,
            note="photo classifier disabled via DD_ENABLE_PHOTO_CLASSIFIER=0",
        )

    if photo_bytes is None and photo_url:
        photo_bytes = await _fetch_image(photo_url)
    if not photo_bytes:
        return PhotoAnalysis(
            founder_name=founder_name,
            photo_source=photo_url or "",
            nearest=[],
            trait_scores={t: 0.0 for t in TRAITS},
            available=False,
            note="no photo bytes available",
        )

    embedding = _embed_face(photo_bytes)
    if embedding is None:
        return PhotoAnalysis(
            founder_name=founder_name,
            photo_source=photo_url or "",
            nearest=[],
            trait_scores={t: 0.0 for t in TRAITS},
            available=False,
            note="insightface unavailable or no face detected",
        )

    corpus, corpus_emb = _load_corpus()
    if not corpus:
        return PhotoAnalysis(
            founder_name=founder_name,
            photo_source=photo_url or "",
            nearest=[],
            trait_scores={t: 0.0 for t in TRAITS},
            available=False,
            note=(
                "unicorn corpus is empty — run scripts/build_unicorn_corpus.py "
                "to populate data/unicorn_founders.parquet"
            ),
        )

    sims = corpus_emb @ embedding
    order = np.argsort(-sims)[:k]
    matches: list[FounderMatch] = []
    cohort_counts: dict[str, int] = {}
    weighted = {t: 0.0 for t in TRAITS}
    weight_sum = 0.0
    for i in order:
        row = corpus[int(i)]
        sim = float(sims[int(i)])
        traits = {t: float(row.get(t, 0.0)) for t in TRAITS}
        cohort = row.get("cohort")  # may be missing on older parquets
        if cohort:
            cohort_counts[str(cohort)] = cohort_counts.get(str(cohort), 0) + 1
        matches.append(FounderMatch(
            founder_id=str(row.get("founder_id", "?")),
            company=str(row.get("company", "?")),
            photo_url=row.get("photo_url"),
            similarity=sim,
            cohort=str(cohort) if cohort else None,
            traits=traits,
        ))
        w = max(sim, 0.0)
        weight_sum += w
        for t, v in traits.items():
            weighted[t] += w * v

    if weight_sum > 0:
        for t in weighted:
            weighted[t] /= weight_sum

    return PhotoAnalysis(
        founder_name=founder_name,
        photo_source=photo_url or "(inline)",
        nearest=matches,
        trait_scores={t: round(weighted[t], 2) for t in TRAITS},
        cohort_breakdown=cohort_counts,
        available=True,
    )
