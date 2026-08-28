"""
Dossier tests: the citation guarantee.

The central assertion is that an uncited claim cannot be constructed. Everything
else in the audit-trail story is downstream of that being true at the type
level rather than by convention.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.dossier_builder import (
    Dossier,
    Evidence,
    build_dossier,
    dossier_corpus_metrics,
    generate_llm_summary,
)
from src.feature_engine import compute_candidate_meta, compute_features
from src.vulcan_integration import VulcanComposer


def _valid_evidence(n: int = 2) -> list[Evidence]:
    """A small list of properly cited evidence."""
    return [
        Evidence(claim=f"claim {i}", citation_type="feature_value",
                 citation_id=f"feature:x{i}", value=i, supporting_citations=[f"feature:y{i}"])
        for i in range(n)
    ]


def _dossier_kwargs(**overrides):
    """Minimal valid Dossier constructor arguments."""
    base = dict(
        alert_id="RF-TEST",
        candidate_cards=["CARD_000001", "CARD_000002"],
        ringfence_score=0.87,
        vulcan_composite_score=0.91,
        threshold_used=0.31,
        recommendation="BLOCK",
        evidence=_valid_evidence(),
        generated_at=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return base


# ── the guarantee ────────────────────────────────────────────────────────


def test_dossier_raises_on_uncited_claim():
    """An empty `citation_id` makes the dossier unconstructible."""
    bad = Evidence(claim="cards look suspicious", citation_type="feature_value",
                   citation_id="", value=None)
    with pytest.raises(ValueError, match="Uncited claim detected"):
        Dossier(**_dossier_kwargs(evidence=[*_valid_evidence(), bad]))


def test_dossier_raises_on_whitespace_only_citation():
    """A whitespace citation is not a citation."""
    bad = Evidence(claim="trust me", citation_type="graph_edge", citation_id="   ", value=1)
    with pytest.raises(ValueError, match="Uncited claim detected"):
        Dossier(**_dossier_kwargs(evidence=[bad]))


def test_dossier_error_names_the_offending_claims():
    """The error identifies which claims failed, so it is actionable."""
    bad = Evidence(claim="the smoking gun", citation_type="graph_edge", citation_id="", value=1)
    with pytest.raises(ValueError, match="the smoking gun"):
        Dossier(**_dossier_kwargs(evidence=[bad]))


def test_valid_dossier_constructs():
    """A fully-cited dossier constructs and reports zero uncited claims."""
    d = Dossier(**_dossier_kwargs())
    assert d.n_uncited_claims == 0
    assert d.refs_per_claim == 2.0


# ── built dossiers ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def built_dossier(positive_candidate, graph, index, request):
    """A dossier built by the real pipeline for a genuine ring candidate."""
    trained = request.getfixturevalue("trained")
    from src.scorer import RingfenceScorer

    scorer = RingfenceScorer()
    cards = list(positive_candidate.cards)
    feats = compute_features(cards, graph, index)
    meta = compute_candidate_meta(cards, index)
    score = scorer.score(feats)
    composite = VulcanComposer().compute_composite(meta.get("vulcan_mean", 0.0), score)
    return build_dossier(cards, feats, meta, graph, score, composite,
                         scorer.threshold, use_llm=False)


def test_built_dossier_has_no_uncited_claims(built_dossier):
    """The generated evidence is fully cited — the metric the bar asks for."""
    assert built_dossier.n_uncited_claims == 0
    assert built_dossier.quality_metrics()["n_uncited_claims"] == 0


def test_built_dossier_meets_citation_density(built_dossier):
    """At least two references per claim on average."""
    assert built_dossier.refs_per_claim >= 2.0, (
        f"only {built_dossier.refs_per_claim:.2f} refs per claim"
    )


def test_every_claim_carries_at_least_one_citation(built_dossier):
    """Per-claim, not just on average."""
    for e in built_dossier.evidence:
        assert e.citation_id.strip()
        assert e.n_references >= 1


def test_dossier_cites_multiple_evidence_types(built_dossier):
    """
    A dossier drawing on one evidence type is a score with extra words.

    A real case file spans the graph, the transactions and the model.
    """
    types = {e.citation_type for e in built_dossier.evidence}
    assert len(types) >= 3, f"only {types} present"
    assert "graph_edge" in types
    assert "vulcan_score" in types


def test_dossier_carries_graph_provenance(built_dossier):
    """Graph claims name the identity value and its network-wide cardinality."""
    graph_claims = [e for e in built_dossier.evidence if e.citation_type == "graph_edge"]
    assert graph_claims
    for e in graph_claims:
        assert isinstance(e.value, dict)
        assert e.value["attribute"] in ("device_fingerprint", "ip_address", "email_hash")
        assert e.value["network_cardinality"] >= e.value["n_cards_in_candidate"]


def test_dossier_records_the_operating_threshold(built_dossier, trained):
    """The dossier states which threshold the decision was made against."""
    assert built_dossier.threshold_used == pytest.approx(trained["artifact"]["threshold"], abs=1e-6)


def test_dossier_snapshot_contains_all_31_features(built_dossier):
    """The feature snapshot is complete, so a decision can be replayed exactly."""
    from src.feature_engine import FEATURE_NAMES

    assert set(built_dossier.feature_snapshot) == set(FEATURE_NAMES)


def test_dossier_round_trips_through_json(built_dossier):
    """Serialisation is lossless — the API stores dossiers as JSON."""
    restored = Dossier.model_validate_json(built_dossier.model_dump_json())
    assert restored.alert_id == built_dossier.alert_id
    assert len(restored.evidence) == len(built_dossier.evidence)
    assert restored.n_uncited_claims == 0


# ── LLM degradation ──────────────────────────────────────────────────────


def test_llm_summary_degrades_without_a_key(monkeypatch):
    """No API key is a skip, not a crash."""
    monkeypatch.setattr("src.dossier_builder.OPENAI_API_KEY", None)
    summary, status = generate_llm_summary(_valid_evidence())
    assert summary is None
    assert status == "skipped:no_api_key"


def test_dossier_falls_back_to_a_deterministic_summary(built_dossier):
    """
    The dossier ships a summary regardless.

    The narrative is a convenience; the evidence is the product. A missing or
    failing LLM must never block an alert.
    """
    assert built_dossier.llm_summary
    assert "template_fallback" in built_dossier.llm_status
    assert str(len(built_dossier.candidate_cards)) in built_dossier.llm_summary


def test_llm_failure_does_not_break_dossier_construction(monkeypatch, positive_candidate, graph, index, trained):
    """An exploding LLM client still yields a complete dossier."""
    monkeypatch.setattr("src.dossier_builder.OPENAI_API_KEY", "sk-fake-key-for-test")

    def boom(*args, **kwargs):
        raise RuntimeError("simulated API outage")

    monkeypatch.setattr("src.dossier_builder.generate_llm_summary",
                        lambda ev, timeout=1.0: (None, "failed:RuntimeError"))

    from src.scorer import RingfenceScorer

    scorer = RingfenceScorer()
    cards = list(positive_candidate.cards)
    feats = compute_features(cards, graph, index)
    meta = compute_candidate_meta(cards, index)
    composite = VulcanComposer().compute_composite(0.3, 0.8)
    d = build_dossier(cards, feats, meta, graph, 0.8, composite, 0.3, use_llm=True)

    assert d.llm_summary
    assert d.n_uncited_claims == 0
    assert "failed" in d.llm_status


def test_corpus_metrics_report_zero_uncited(built_dossier):
    """Corpus-level dossier quality."""
    m = dossier_corpus_metrics([built_dossier, built_dossier])
    assert m["n_dossiers"] == 2
    assert m["uncited_claims"] == 0
    assert m["refs_per_claim"] >= 2.0
