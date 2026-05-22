"""v5 photo classifier characteristic profile — percentile, distinctive
features, archetype clustering."""

from dd_agent.modules import photo_classifier as pc


# --- _trait_percentiles ------------------------------------------------------


def test_trait_percentiles_against_known_corpus():
    """Founder at the top of every trait should hit ~100th percentile.
    Founder at the bottom should hit ~0th percentile."""
    corpus = [
        {"resilience": 2.0, "intensity": 2.0, "warmth": 2.0,
         "presentation_polish": 2.0, "energy": 2.0},
        {"resilience": 3.0, "intensity": 3.0, "warmth": 3.0,
         "presentation_polish": 3.0, "energy": 3.0},
        {"resilience": 4.0, "intensity": 4.0, "warmth": 4.0,
         "presentation_polish": 4.0, "energy": 4.0},
        {"resilience": 5.0, "intensity": 5.0, "warmth": 5.0,
         "presentation_polish": 5.0, "energy": 5.0},
    ]
    top_pct = pc._trait_percentiles(
        {"resilience": 5.0, "intensity": 5.0, "warmth": 5.0,
         "presentation_polish": 5.0, "energy": 5.0},
        corpus,
    )
    # 100% trait = top of distribution (3 strictly below + half of 1 tie / 4 = 87.5)
    for trait in pc.TRAITS:
        assert top_pct[trait] > 80, f"{trait} should be high percentile, got {top_pct[trait]}"

    bottom_pct = pc._trait_percentiles(
        {"resilience": 1.0, "intensity": 1.0, "warmth": 1.0,
         "presentation_polish": 1.0, "energy": 1.0},
        corpus,
    )
    for trait in pc.TRAITS:
        assert bottom_pct[trait] < 20, f"{trait} should be low percentile, got {bottom_pct[trait]}"


def test_trait_percentiles_empty_corpus():
    assert pc._trait_percentiles({"resilience": 3.0}, []) == {}


# --- _distinctive_features ---------------------------------------------------


def test_distinctive_features_flags_high_outlier():
    """When a trait is >+1σ above corpus mean, it should appear as distinctive."""
    # 10 founders, mean=3, σ≈1.4 — value of 5 is z≈+1.4
    corpus = [{"intensity": v, "resilience": 3.0, "warmth": 3.0,
               "presentation_polish": 3.0, "energy": 3.0}
              for v in (1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0)]
    distinctive = pc._distinctive_features(
        {"intensity": 5.0, "resilience": 3.0, "warmth": 3.0,
         "presentation_polish": 3.0, "energy": 3.0},
        corpus,
    )
    intensity_features = [d for d in distinctive if d.trait == "intensity"]
    assert len(intensity_features) == 1
    assert intensity_features[0].direction == "high"
    assert intensity_features[0].z_score > 1.0


def test_distinctive_features_flags_low_outlier():
    corpus = [{"warmth": v, "resilience": 3.0, "intensity": 3.0,
               "presentation_polish": 3.0, "energy": 3.0}
              for v in (3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0)]
    distinctive = pc._distinctive_features(
        {"warmth": 2.0, "resilience": 3.0, "intensity": 3.0,
         "presentation_polish": 3.0, "energy": 3.0},
        corpus,
    )
    warmth_features = [d for d in distinctive if d.trait == "warmth"]
    assert len(warmth_features) == 1
    assert warmth_features[0].direction == "low"
    assert warmth_features[0].z_score < -1.0


def test_distinctive_features_empty_when_near_median():
    """If trait values are all near the median, nothing is distinctive."""
    corpus = [{"intensity": 3.0, "resilience": 3.0, "warmth": 3.0,
               "presentation_polish": 3.0, "energy": 3.0} for _ in range(10)]
    distinctive = pc._distinctive_features(
        {"intensity": 3.0, "resilience": 3.0, "warmth": 3.0,
         "presentation_polish": 3.0, "energy": 3.0},
        corpus,
    )
    assert distinctive == []


# --- _archetype_clusters -----------------------------------------------------


def test_archetype_clusters_group_by_cohort():
    """Nearest matches should group into archetypes by their cohort."""
    matches = [
        pc.FounderMatch(founder_id="a", company="A", photo_url=None, similarity=0.9,
                        cohort="yc_top_100",
                        traits={"intensity": 4.5, "resilience": 4.0, "warmth": 2.5,
                                "presentation_polish": 3.5, "energy": 4.0}),
        pc.FounderMatch(founder_id="b", company="B", photo_url=None, similarity=0.8,
                        cohort="yc_top_100",
                        traits={"intensity": 4.0, "resilience": 4.0, "warmth": 2.5,
                                "presentation_polish": 3.5, "energy": 3.8}),
        pc.FounderMatch(founder_id="c", company="C", photo_url=None, similarity=0.75,
                        cohort="public_sp500_nasdaq",
                        traits={"intensity": 3.0, "resilience": 3.0, "warmth": 4.5,
                                "presentation_polish": 4.5, "energy": 4.0}),
    ]
    archetypes = pc._archetype_clusters(matches)
    # Should produce 2 clusters — one per cohort
    assert len(archetypes) == 2
    cohorts = {a.dominant_cohort for a in archetypes}
    assert cohorts == {"yc_top_100", "public_sp500_nasdaq"}


def test_archetype_label_recognizes_technical_visionary():
    """High intensity + resilience + low warmth → 'Technical visionary'."""
    label = pc._archetype_label(
        centroid={"intensity": 4.5, "resilience": 4.0, "warmth": 2.5,
                  "presentation_polish": 3.0, "energy": 4.0},
        cohort="yc_top_100",
    )
    assert label == "Technical visionary"


def test_archetype_label_recognizes_charismatic_operator():
    """High warmth + presentation_polish + energy → 'Charismatic operator'."""
    label = pc._archetype_label(
        centroid={"intensity": 3.5, "resilience": 3.5, "warmth": 4.0,
                  "presentation_polish": 4.5, "energy": 4.0},
        cohort="public_sp500_nasdaq",
    )
    assert label == "Charismatic operator"


# --- summary_for_prompt ------------------------------------------------------


def test_summary_for_prompt_unavailable_path():
    pa = pc.PhotoAnalysis(
        founder_name="X", photo_source="", nearest=[],
        trait_scores={t: 0.0 for t in pc.TRAITS},
        available=False, note="no photo bytes available",
    )
    s = pa.summary_for_prompt()
    assert "unavailable" in s.lower()
    assert "no photo bytes available" in s


def test_summary_for_prompt_includes_percentiles_and_archetypes():
    pa = pc.PhotoAnalysis(
        founder_name="Alfred Founder",
        photo_source="https://x.com/photo.jpg",
        nearest=[pc.FounderMatch(founder_id="a", company="Acme", photo_url=None,
                                 similarity=0.81, cohort="yc_top_100")],
        trait_scores={"resilience": 4.0, "intensity": 4.5, "warmth": 2.5,
                      "presentation_polish": 3.0, "energy": 4.0},
        trait_percentiles={"resilience": 78, "intensity": 92, "warmth": 34,
                           "presentation_polish": 55, "energy": 71},
        distinctive_features=[pc.DistinctiveFeature(
            trait="intensity", value=4.5, corpus_mean=3.0, corpus_std=0.9,
            z_score=1.67, direction="high",
        )],
        archetypes=[pc.Archetype(
            label="Technical visionary", members=["pcollison"],
            member_companies=["Stripe"], centroid_traits={"intensity": 4.5},
            dominant_cohort="yc_top_100",
        )],
        cohort_breakdown={"yc_top_100": 1},
        available=True,
    )
    s = pa.summary_for_prompt()
    assert "Trait percentiles" in s
    assert "intensity: 92" in s or "intensity: 92.0" in s.replace(".", "")
    assert "Technical visionary" in s
    assert "Distinctive features" in s
    assert "intensity high" in s
    assert "Closest cohort" in s
