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
class DistinctiveFeature:
    """A trait where this founder is unusually high or low vs the corpus."""
    trait: str                        # e.g. "intensity"
    value: float                      # the kNN-weighted trait score (1-5 scale)
    corpus_mean: float
    corpus_std: float
    z_score: float                    # how many σ from the corpus mean
    direction: str                    # "high" (z>0) | "low" (z<0)


@dataclass
class Archetype:
    """A cluster of nearest neighbors that share a profile."""
    label: str                        # "Technical visionary" | "Operator-founder" | etc.
    members: list[str]                # founder_id values
    member_companies: list[str]
    centroid_traits: dict[str, float] # mean of the cluster's trait values
    dominant_cohort: str | None       # most common cohort among members


@dataclass
class PhotoAnalysis:
    founder_name: str
    photo_source: str
    nearest: list[FounderMatch]
    trait_scores: dict[str, float]
    available: bool
    cohort_breakdown: dict[str, int] = field(default_factory=dict)  # cohort → count in nearest-k
    # v5 additions — characteristic profile vs the unicorn corpus.
    trait_percentiles: dict[str, float] = field(default_factory=dict)        # 0-100 vs full corpus
    cohort_percentiles: dict[str, dict[str, float]] = field(default_factory=dict)  # cohort → trait → percentile
    distinctive_features: list[DistinctiveFeature] = field(default_factory=list)
    archetypes: list[Archetype] = field(default_factory=list)
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "founder_name": self.founder_name,
            "photo_source": self.photo_source,
            "nearest": [m.__dict__ for m in self.nearest],
            "trait_scores": self.trait_scores,
            "cohort_breakdown": self.cohort_breakdown,
            "trait_percentiles": self.trait_percentiles,
            "cohort_percentiles": self.cohort_percentiles,
            "distinctive_features": [d.__dict__ for d in self.distinctive_features],
            "archetypes": [a.__dict__ for a in self.archetypes],
            "summary_for_prompt": self.summary_for_prompt(),
            "available": self.available,
            "note": self.note,
        }

    def summary_for_prompt(self) -> str:
        """Human-readable summary suitable for direct injection into an LLM
        prompt — saves the LLM from having to parse the raw JSON."""
        if not self.available:
            return f"Photo profile unavailable: {self.note or 'no data'}"
        lines = [f"Founder: {self.founder_name}"]
        if self.trait_percentiles:
            lines.append("Trait percentiles vs unicorn corpus:")
            for trait in ("resilience", "intensity", "warmth", "presentation_polish", "energy"):
                if trait in self.trait_percentiles:
                    pct = self.trait_percentiles[trait]
                    val = self.trait_scores.get(trait, 0)
                    lines.append(f"  {trait}: {pct:.0f}th percentile (value {val})")
        if self.distinctive_features:
            lines.append("Distinctive features (>±1σ from corpus mean):")
            for d in self.distinctive_features:
                lines.append(
                    f"  {d.trait} {d.direction} (z={d.z_score:+.2f}, "
                    f"value {d.value} vs corpus mean {d.corpus_mean})"
                )
        else:
            lines.append("Distinctive features: none (profile is near corpus median)")
        if self.archetypes:
            lines.append("Archetype clusters from nearest matches:")
            for a in self.archetypes:
                companies = ", ".join(a.member_companies[:5])
                lines.append(f"  '{a.label}' ({a.dominant_cohort}): {companies}")
        if self.cohort_breakdown:
            top = max(self.cohort_breakdown.items(), key=lambda x: x[1])
            lines.append(f"Closest cohort: {top[0]} ({top[1]} of nearest-{len(self.nearest)})")
        return "\n".join(lines)


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

    trait_scores = {t: round(weighted[t], 2) for t in TRAITS}

    # v5: characteristic profile vs the unicorn corpus
    trait_percentiles = _trait_percentiles(trait_scores, corpus)
    cohort_percentiles = _cohort_trait_percentiles(trait_scores, corpus)
    distinctive = _distinctive_features(trait_scores, corpus)
    archetypes = _archetype_clusters(matches)

    return PhotoAnalysis(
        founder_name=founder_name,
        photo_source=photo_url or "(inline)",
        nearest=matches,
        trait_scores=trait_scores,
        cohort_breakdown=cohort_counts,
        trait_percentiles=trait_percentiles,
        cohort_percentiles=cohort_percentiles,
        distinctive_features=distinctive,
        archetypes=archetypes,
        available=True,
    )


# --- v5 characteristic-profile helpers --------------------------------------


def _trait_percentiles(trait_scores: dict[str, float], corpus: list[dict]) -> dict[str, float]:
    """For each trait, return the percentile rank of this founder's score
    among all corpus rows (0-100). Higher = more distinctive."""
    if not corpus:
        return {}
    out: dict[str, float] = {}
    for trait, value in trait_scores.items():
        corpus_vals = [float(r.get(trait, 3.0)) for r in corpus]
        if not corpus_vals:
            continue
        # Percentile = fraction of corpus strictly below + half ties (standard rank)
        below = sum(1 for v in corpus_vals if v < value)
        ties = sum(1 for v in corpus_vals if v == value)
        pct = 100.0 * (below + 0.5 * ties) / len(corpus_vals)
        out[trait] = round(pct, 1)
    return out


def _cohort_trait_percentiles(
    trait_scores: dict[str, float], corpus: list[dict],
) -> dict[str, dict[str, float]]:
    """Same percentile calc but partitioned by cohort. Returns
    {cohort_name: {trait: percentile}}. Skips cohorts with fewer than 5 rows."""
    if not corpus:
        return {}
    by_cohort: dict[str, list[dict]] = {}
    for r in corpus:
        c = r.get("cohort")
        if not c:
            continue
        by_cohort.setdefault(str(c), []).append(r)
    out: dict[str, dict[str, float]] = {}
    for cohort_name, rows in by_cohort.items():
        if len(rows) < 5:
            continue
        out[cohort_name] = _trait_percentiles(trait_scores, rows)
    return out


def _distinctive_features(
    trait_scores: dict[str, float], corpus: list[dict],
) -> list[DistinctiveFeature]:
    """Return traits where this founder is more than ±1σ from the corpus mean.
    These are the 'distinctive' parts of the founder's visual profile vs the
    overall unicorn distribution."""
    import statistics
    if not corpus or len(corpus) < 5:
        return []
    out: list[DistinctiveFeature] = []
    for trait, value in trait_scores.items():
        corpus_vals = [float(r.get(trait, 3.0)) for r in corpus]
        if len(corpus_vals) < 5:
            continue
        try:
            mu = statistics.mean(corpus_vals)
            sigma = statistics.stdev(corpus_vals)
        except statistics.StatisticsError:
            continue
        if sigma == 0:
            continue
        z = (value - mu) / sigma
        if abs(z) >= 1.0:
            out.append(DistinctiveFeature(
                trait=trait,
                value=round(value, 2),
                corpus_mean=round(mu, 2),
                corpus_std=round(sigma, 2),
                z_score=round(z, 2),
                direction="high" if z > 0 else "low",
            ))
    # Sort by absolute z so most distinctive features come first
    out.sort(key=lambda d: abs(d.z_score), reverse=True)
    return out


def _archetype_clusters(matches: list[FounderMatch]) -> list[Archetype]:
    """Cluster the nearest matches into 2-3 archetypes by their cohort + the
    centroid of their trait vectors.

    Strategy: group by cohort first (since cohort is a strong prior),
    then within each cohort compute a centroid trait vector + name the
    cluster using a simple rule based on which trait is dominant.

    The 'label' is descriptive — 'Technical visionary' when intensity +
    resilience are dominant, 'Operator-founder' when presentation_polish +
    warmth are dominant, etc."""
    if not matches:
        return []
    by_cohort: dict[str, list[FounderMatch]] = {}
    for m in matches:
        if not m.cohort:
            continue
        by_cohort.setdefault(m.cohort, []).append(m)
    if not by_cohort:
        return []

    archetypes: list[Archetype] = []
    for cohort_name, members in by_cohort.items():
        if len(members) < 1:
            continue
        # Centroid of trait vectors
        centroid: dict[str, float] = {t: 0.0 for t in TRAITS}
        for m in members:
            for t, v in (m.traits or {}).items():
                if t in centroid:
                    centroid[t] += float(v)
        for t in centroid:
            centroid[t] = round(centroid[t] / len(members), 2)
        # Label: pick the dominant trait pair from the centroid
        label = _archetype_label(centroid, cohort_name)
        archetypes.append(Archetype(
            label=label,
            members=[m.founder_id for m in members],
            member_companies=[m.company for m in members],
            centroid_traits=centroid,
            dominant_cohort=cohort_name,
        ))
    # Sort by cluster size descending, take up to 3
    archetypes.sort(key=lambda a: len(a.members), reverse=True)
    return archetypes[:3]


_ARCHETYPE_RULES = (
    # (label, condition over centroid)
    ("Technical visionary",
     lambda c: c.get("intensity", 0) >= 3.5 and c.get("resilience", 0) >= 3.5 and c.get("warmth", 0) < 3.5),
    ("Charismatic operator",
     lambda c: c.get("warmth", 0) >= 3.5 and c.get("presentation_polish", 0) >= 3.5 and c.get("energy", 0) >= 3.5),
    ("Resilient builder",
     lambda c: c.get("resilience", 0) >= 3.8 and c.get("intensity", 0) >= 3.3),
    ("Polished executive",
     lambda c: c.get("presentation_polish", 0) >= 4.0 and c.get("warmth", 0) >= 3.0),
    ("High-energy founder",
     lambda c: c.get("energy", 0) >= 4.0),
)


def _archetype_label(centroid: dict[str, float], cohort: str) -> str:
    """Pick the best descriptive label for a centroid."""
    for label, predicate in _ARCHETYPE_RULES:
        try:
            if predicate(centroid):
                return label
        except Exception:
            continue
    # Fallback: name by cohort
    cohort_labels = {
        "public_sp500_nasdaq": "Public-company founder",
        "yc_top_100": "YC-top alum",
        "unicorn_private": "Unicorn private founder",
    }
    return cohort_labels.get(cohort, "Mixed archetype")
