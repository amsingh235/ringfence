"""
API tests: endpoint contracts, Pydantic validation, graceful degradation, and
the defense-only posture.

The degradation tests carry the most weight. A fraud API that 500s on unknown
input is worse than useless — it fails open in practice, because whatever calls
it will time out and let the transaction through.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.config import GATING


@pytest.fixture(scope="module")
def client(trained, graph, index, candidates):
    """A TestClient with the app's lifespan run, so artifacts are loaded."""
    from src.api import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def scored_alert(client, positive_candidate):
    """Score a genuine ring candidate and return the response body."""
    r = client.post("/detect/candidate", json={"cards": list(positive_candidate.cards)})
    assert r.status_code == 200
    return r.json()


# ── contracts ────────────────────────────────────────────────────────────


def test_root_declares_defensive_posture(client):
    """The service says what it is and what it cannot do."""
    body = client.get("/").json()
    assert "defensive" in body["posture"].lower()
    assert "cannot" in body["posture"].lower()


def test_health_reports_component_status(client):
    """Health is per-component, not a single opaque boolean."""
    r = client.get("/health")
    assert r.status_code in (200, 503)
    body = r.json()
    assert set(body["components"]) == {"identity_graph", "transaction_index", "model", "case_memory"}
    assert body["defense_only"] is True
    assert body["model_version"]


def test_detect_returns_score_composite_and_dossier(scored_alert):
    """The core endpoint returns everything a decision needs."""
    assert 0.0 <= scored_alert["ringfence_score"] <= 1.0
    assert scored_alert["composite"]["recommendation"] in ("APPROVE", "REVIEW", "BLOCK")
    assert scored_alert["dossier"]["evidence"]
    assert scored_alert["dossier_quality"]["n_uncited_claims"] == 0


def test_detect_explains_the_composition(scored_alert):
    """
    Every money-relevant number is explained, not just emitted.

    The response carries the composition formula, the gate bands, both input
    scores and the bounded-action rule — an analyst can reconstruct the
    decision without reading our source.
    """
    c = scored_alert["composite"]
    assert "noisy_or" in c["composition"]
    assert set(c["gate_bands"]) == {"APPROVE", "REVIEW", "BLOCK"}
    assert "vulcan" in c["contributing_signals"] and "ringfence" in c["contributing_signals"]
    assert "bounded_action_rule" in c


def test_novel_pattern_never_permits_automated_action(scored_alert):
    """
    Bounded and gated.

    With no Ring Template behind it, no candidate may unlock an automated
    action — regardless of how high it scores.
    """
    assert scored_alert["composite"]["auto_action_allowed"] is False
    assert scored_alert["composite"]["human_review_required"] is True


def test_ingest_accepts_and_acknowledges(client):
    """Ingest buffers and says so, rather than pretending to score inline."""
    r = client.post("/ingest/transaction", json={
        "transaction_id": "TXN_TEST_1", "card_id": "CARD_000001", "merchant_id": "MERCH_00001",
        "amount": 5.0, "timestamp": "2026-01-01T14:23:17", "device_fingerprint": "DEV_000001",
        "ip_address": "10.0.0.1", "email_hash": "EMAIL_000001", "vulcan_score": 0.22,
    })
    assert r.status_code == 200
    assert r.json()["status"] == "accepted"
    assert "batch" in r.json()["note"].lower()


def test_alerts_list_paginates_and_filters(client, scored_alert):
    """Pagination and filters behave."""
    body = client.get("/alerts", params={"limit": 5}).json()
    assert body["total"] >= 1
    assert len(body["alerts"]) <= 5

    rec = scored_alert["composite"]["recommendation"]
    filtered = client.get("/alerts", params={"recommendation": rec}).json()
    assert all(a["recommendation"] == rec for a in filtered["alerts"])


def test_dossier_endpoint_returns_full_evidence(client, scored_alert):
    """The dossier endpoint returns the stored case file with its quality metrics."""
    r = client.get(f"/alerts/{scored_alert['alert_id']}/dossier")
    assert r.status_code == 200
    body = r.json()
    assert body["quality"]["n_uncited_claims"] == 0
    assert body["quality"]["refs_per_claim"] >= 1.0


def test_unknown_alert_returns_404(client):
    """Missing resources 404 — they do not 500 and do not invent a dossier."""
    assert client.get("/alerts/RF-DOESNOTEXIST/dossier").status_code == 404
    assert client.get("/alerts/RF-DOESNOTEXIST/failure-recovery").status_code == 404


def test_metrics_endpoint_is_prometheus_formatted(client):
    """Metrics are scrapeable, with HELP and TYPE lines."""
    text = client.get("/metrics").text
    assert "# HELP ringfence_up" in text
    assert "# TYPE" in text
    assert "ringfence_detect_total" in text


# ── validation ───────────────────────────────────────────────────────────


def test_empty_card_list_is_rejected(client):
    """Pydantic rejects a request that cannot mean anything."""
    assert client.post("/detect/candidate", json={"cards": []}).status_code == 422


def test_oversized_card_list_is_rejected(client):
    """A 5,000-card 'candidate' is a population, and is refused."""
    assert client.post("/detect/candidate",
                       json={"cards": [f"CARD_{i:06d}" for i in range(600)]}).status_code == 422


def test_out_of_range_vulcan_score_is_rejected(client):
    """Vulcan scores are probabilities; 1.4 is not one."""
    assert client.post("/detect/candidate",
                       json={"cards": ["CARD_000001"], "vulcan_score": 1.4}).status_code == 422


def test_invalid_disposition_is_rejected(client, scored_alert):
    """Only the two valid verdicts are accepted."""
    r = client.post(f"/alerts/{scored_alert['alert_id']}/disposition",
                    json={"disposition": "probably_fine"})
    assert r.status_code == 422


def test_duplicate_cards_are_deduplicated(client, positive_candidate):
    """A card listed twice is one card."""
    cards = list(positive_candidate.cards)
    r = client.post("/detect/candidate", json={"cards": cards + cards})
    assert r.status_code == 200
    assert len(r.json()["cards"]) == len(set(cards))


# ── graceful degradation ─────────────────────────────────────────────────


def test_unknown_cards_degrade_rather_than_crash(client):
    """
    An unknown card produces a degraded, safe response — not a 500.

    This is the failure mode that matters operationally: a fraud API that
    errors gets timed out by its caller, and a timed-out fraud check is an
    approval.
    """
    r = client.post("/detect/candidate", json={"cards": ["CARD_DOES_NOT_EXIST"]})
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["degraded_reasons"]
    assert body["composite"]["recommendation"] == GATING.safe_default_recommendation


def test_degraded_response_never_approves(client):
    """A detector that cannot see must escalate, never wave through."""
    body = client.post("/detect/candidate", json={"cards": ["CARD_UNKNOWN_A", "CARD_UNKNOWN_B"]}).json()
    assert body["composite"]["recommendation"] != "APPROVE"
    assert body["composite"]["confidence"] == 0.0


def test_degraded_response_still_carries_a_valid_dossier(client):
    """Even degraded, the audit trail exists and is fully cited."""
    body = client.post("/detect/candidate", json={"cards": ["CARD_UNKNOWN_C"]}).json()
    assert body["dossier"]["degraded"] is True
    assert body["dossier"]["degraded_reason"]
    assert body["dossier_quality"]["n_uncited_claims"] == 0


def test_partially_known_card_set_is_flagged(client, positive_candidate):
    """Mixing known and unknown cards degrades and says which were dropped."""
    cards = list(positive_candidate.cards) + ["CARD_NOT_REAL"]
    body = client.post("/detect/candidate", json={"cards": cards}).json()
    assert body["degraded"] is True
    assert "unknown to the graph" in " ".join(body["degraded_reasons"])
    assert len(body["cards_scored"]) == len(positive_candidate.cards)


# ── the failure-recovery loop ────────────────────────────────────────────


def test_disposition_then_failure_recovery_shows_the_damp(client, negative_candidate):
    """
    The full loop: alert -> analyst rejects -> future score damped, with the
    before/after visible on the failure-recovery endpoint.
    """
    first = client.post("/detect/candidate", json={"cards": list(negative_candidate.cards)}).json()
    alert_id = first["alert_id"]

    disp = client.post(f"/alerts/{alert_id}/disposition",
                       json={"disposition": "false_positive", "notes": "household device, not a ring"})
    assert disp.status_code == 200
    assert "damped" in disp.json()["effect_on_future_scoring"]

    recovery = client.get(f"/alerts/{alert_id}/failure-recovery").json()
    assert recovery["disposition"] == "false_positive"
    assert recovery["matched_false_positive_patterns"]
    assert recovery["after_case_memory"]["fp_damp_applied"] > 0
    assert recovery["score_delta"] < 0
    assert recovery["audit_trail"]


def test_confirmed_fraud_creates_a_precedent(client, positive_candidate):
    """A fraud verdict becomes a precedent for the next similar candidate."""
    first = client.post("/detect/candidate", json={"cards": list(positive_candidate.cards)}).json()
    client.post(f"/alerts/{first['alert_id']}/disposition",
                json={"disposition": "fraud", "notes": "confirmed probe-then-cashout ring"})

    second = client.post("/detect/candidate", json={"cards": list(positive_candidate.cards)}).json()
    assert second["dossier"]["precedents"] or second["dossier"]["false_positive_patterns"], (
        "case memory had no effect on a re-scored identical candidate"
    )


def test_dossier_quality_endpoint_reports_zero_uncited(client, scored_alert):
    """Corpus-level dossier quality across everything stored."""
    body = client.get("/dossier-quality").json()
    assert body["n_dossiers"] >= 1
    assert body["uncited_claims"] == 0


# ── defense-only posture ─────────────────────────────────────────────────


def test_no_offensive_endpoints_exist(client):
    """
    Strictly defense-only: the route table contains nothing offense-capable.

    Asserted against the live OpenAPI schema rather than trusting a README line.
    """
    paths = client.get("/openapi.json").json()["paths"]
    forbidden = ("test", "probe", "attack", "generate", "simulate", "charge", "block", "decline", "capture")
    for path in paths:
        assert not any(word in path.lower() for word in forbidden), f"suspicious route {path}"


def test_no_endpoint_can_move_money(client):
    """Every route is read-only or advisory; none executes a payment action."""
    paths = client.get("/openapi.json").json()["paths"]
    mutating = {p for p, ops in paths.items() if "post" in ops}
    assert mutating <= {"/ingest/transaction", "/detect/candidate", "/alerts/{alert_id}/disposition"}
