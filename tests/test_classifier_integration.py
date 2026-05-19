"""Integration test for the photo classifier against the real parquet.

This test runs against `data/unicorn_founders.parquet` (whatever's currently
in there — the build script populates it incrementally). It's a thin
sanity check that:
  1. The parquet loads cleanly
  2. The corpus has at least 50 founders (basic non-empty guard)
  3. All rows have the required columns including `cohort`
  4. The cohort field uses one of the known values
  5. Embedding dimensions are correct (512 for ArcFace)

If the parquet doesn't exist yet (no corpus build run), the test skips
rather than fails — keeping CI green on a fresh checkout.
"""

from pathlib import Path

import pytest

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "unicorn_founders.parquet"
KNOWN_COHORTS = {"public_sp500_nasdaq", "yc_top_100", "unicorn_private"}


def _load_corpus():
    if not CORPUS_PATH.exists():
        pytest.skip(f"no corpus at {CORPUS_PATH} — run scripts/build_unicorn_corpus.py")
    import pandas as pd
    return pd.read_parquet(CORPUS_PATH)


def test_corpus_loads_and_has_min_size():
    df = _load_corpus()
    assert len(df) >= 50, f"corpus is too small ({len(df)} rows) to be useful for kNN"


def test_corpus_has_required_columns():
    df = _load_corpus()
    required = {"founder_id", "name", "company", "cohort", "embedding"}
    missing = required - set(df.columns)
    assert not missing, f"corpus missing columns: {missing}"


def test_corpus_cohort_values_are_known():
    df = _load_corpus()
    seen = set(df["cohort"].dropna().unique())
    unexpected = seen - KNOWN_COHORTS
    assert not unexpected, f"unexpected cohort values: {unexpected}"


def test_corpus_embedding_dimensions():
    """ArcFace embeddings are 512-d. Sample one row and check the length."""
    df = _load_corpus()
    sample = df.iloc[0]
    emb = sample["embedding"]
    assert len(emb) == 512, f"expected 512-d embedding, got {len(emb)}"


def test_corpus_has_all_three_cohorts_represented():
    """A useful corpus has at least one founder in each cohort."""
    df = _load_corpus()
    cohorts_present = set(df["cohort"].dropna().unique())
    missing = KNOWN_COHORTS - cohorts_present
    assert not missing, (
        f"corpus missing cohorts {missing} — re-run "
        "scripts/build_unicorn_corpus.py with cohort coverage"
    )


def test_classifier_loads_corpus_without_error():
    """The photo_classifier module's _load_corpus() should return the
    parquet rows + an (N, 512) embedding matrix."""
    import sys
    sys.path.insert(0, str(CORPUS_PATH.parent.parent / "src"))
    from dd_agent.modules.photo_classifier import _load_corpus
    corpus, embeddings = _load_corpus()
    assert len(corpus) == embeddings.shape[0]
    assert embeddings.shape[1] == 512


def test_corpus_no_duplicate_founder_ids():
    df = _load_corpus()
    assert df["founder_id"].nunique() == len(df), \
        "found duplicate founder_id rows — dedup logic broken"
