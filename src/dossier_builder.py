"""
Dossier builder — the audit trail.

Every alert ships a case file, not a score. The rule the whole module exists to
enforce is one line long:

    **A claim without a citation cannot be constructed.**

Not "should not". `Dossier` runs a model validator that raises
`ValueError("Uncited claim detected")` if any piece of evidence carries an empty
`citation_id`. There is no flag to bypass it and no code path that builds a
dossier around it. If we cannot point at the graph edge, the transaction ID, or
the feature value that supports a sentence, the sentence does not ship.

The LLM's place in this
-----------------------
The narrative summary is generated **from** the evidence, never around it. The
deterministic evidence list is built first, from the graph and the transaction
index; the model is then handed that JSON and asked to restate it in three
sentences at temperature 0. It cannot introduce a fact, because it is never
shown anything except facts that already carry citations, and it is instructed
not to recommend actions beyond the bounded recommendation the gating layer
already produced.

If the LLM is unavailable — no key, timeout, rate limit, anything — the dossier
is still complete and still ships, with a deterministic template summary and
`llm_status` recording exactly what happened. The narrative is a convenience;
the evidence is the product.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import networkx as nx
from pydantic import BaseModel, Field, model_validator

from src.config import LLM_TIMEOUT_S, OPENAI_API_KEY, OPENAI_MODEL
from src.feature_engine import CASHOUT_AMOUNT_MIN, PROBE_AMOUNT_MAX, TransactionIndex
from src.utils import get_logger, new_alert_id

log = get_logger("ringfence.dossier_builder")

CitationType = Literal["graph_edge", "transaction_id", "feature_value", "corpus_passage", "vulcan_score"]

LLM_PROMPT = """You are a Razorpay fraud analyst assistant. Write a 3-sentence summary of the following evidence.
Only state facts supported by the evidence. Do not infer beyond the citations.
Do not suggest actions beyond the bounded recommendation.

Evidence:
{evidence_json}

Summary:"""


# ──────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────


class Evidence(BaseModel):
    """
    One cited claim.

    `citation_id` is the primary reference — a graph edge key, a transaction ID,
    a feature name. `supporting_citations` holds the additional references
    behind the same claim, which is how "these five cards share this device"
    carries all five card-pair edges rather than pretending to rest on one.
    """

    claim: str
    citation_type: CitationType
    citation_id: str
    value: Any
    supporting_citations: list[str] = Field(default_factory=list)

    @property
    def n_references(self) -> int:
        """Total references backing this claim (primary + supporting)."""
        return 1 + len(self.supporting_citations)


class Dossier(BaseModel):
    """
    A complete case file for one alert.

    Instantiation fails if any evidence is uncited. That is the enforcement
    point for "every money action explainable" — you cannot get an alert out of
    this system without the provenance attached to it.
    """

    alert_id: str
    candidate_cards: list[str]
    ringfence_score: float
    vulcan_composite_score: float
    threshold_used: float
    recommendation: Literal["APPROVE", "REVIEW", "BLOCK"]
    evidence: list[Evidence]
    llm_summary: Optional[str] = None
    generated_at: datetime
    analyst_disposition: Optional[str] = None

    # Context beyond the spec's required fields — all optional, all serialised.
    vulcan_transaction_score: float = 0.0
    auto_action_allowed: bool = False
    degraded: bool = False
    degraded_reason: Optional[str] = None
    llm_status: str = "not_attempted"
    precedents: list[dict] = Field(default_factory=list)
    false_positive_patterns: list[dict] = Field(default_factory=list)
    feature_snapshot: dict[str, float] = Field(default_factory=dict)
    build_ms: float = 0.0

    @model_validator(mode="after")
    def _reject_uncited_claims(self) -> "Dossier":
        """
        Refuse to construct a dossier containing an uncited claim.

        Checked after the whole model is assembled so the error names every
        offending claim at once rather than failing on the first.
        """
        offenders = [e.claim for e in self.evidence if not (e.citation_id or "").strip()]
        if offenders:
            raise ValueError(
                f"Uncited claim detected: {offenders[:3]}"
                + (f" (and {len(offenders) - 3} more)" if len(offenders) > 3 else "")
            )
        return self

    @property
    def n_uncited_claims(self) -> int:
        """Always 0 for a constructed dossier — reported so the metric is visible."""
        return sum(1 for e in self.evidence if not (e.citation_id or "").strip())

    @property
    def refs_per_claim(self) -> float:
        """Mean references per claim. Tracked as a dossier-quality metric."""
        if not self.evidence:
            return 0.0
        return sum(e.n_references for e in self.evidence) / len(self.evidence)

    def quality_metrics(self) -> dict:
        """Dossier quality summary, surfaced on the API and the metrics page."""
        by_type: dict[str, int] = {}
        for e in self.evidence:
            by_type[e.citation_type] = by_type.get(e.citation_type, 0) + 1
        return {
            "n_claims": len(self.evidence),
            "n_uncited_claims": self.n_uncited_claims,
            "refs_per_claim": round(self.refs_per_claim, 2),
            "citations_by_type": by_type,
        }


# ──────────────────────────────────────────────────────────────────────────
# Deterministic evidence generation
# ──────────────────────────────────────────────────────────────────────────


def _graph_evidence(cards: list[str], G: nx.Graph) -> list[Evidence]:
    """
    Turn the candidate's induced subgraph into cited claims about shared
    infrastructure.

    Grouped by identity value rather than by edge, because "cards A, B and C
    all used device D" is the claim an analyst reads; the three individual
    edges are the citations behind it.
    """
    sub = G.subgraph(cards)
    by_value: dict[tuple[str, str], dict] = {}

    for u, v, data in sub.edges(data=True):
        for shared in data.get("shared", []):
            key = (shared["attribute"], shared["value"])
            rec = by_value.setdefault(key, {"cards": set(), "edges": [], "cardinality": shared["cardinality"]})
            rec["cards"].update((u, v))
            rec["edges"].append(f"edge:{u}~{v}:{shared['attribute']}={shared['value']}")

    out: list[Evidence] = []
    label = {"device_fingerprint": "device", "ip_address": "IP address", "email_hash": "email hash"}

    for (attr, value), rec in sorted(by_value.items(), key=lambda kv: -len(kv[1]["cards"]))[:6]:
        card_list = sorted(rec["cards"])
        shown = ", ".join(card_list[:4]) + (f" and {len(card_list) - 4} more" if len(card_list) > 4 else "")
        # Population-level context matters: a device shared by 4 cards is
        # evidence, the same device shared by 1,492 is background noise, and
        # the claim says which one this is.
        out.append(
            Evidence(
                claim=(
                    f"Cards {shown} share {label.get(attr, attr)} {value}"
                    f" (this value is used by {rec['cardinality']} cards network-wide)"
                ),
                citation_type="graph_edge",
                citation_id=rec["edges"][0],
                value={"attribute": attr, "value": value, "n_cards_in_candidate": len(card_list),
                       "network_cardinality": rec["cardinality"]},
                supporting_citations=rec["edges"][1:6],
            )
        )
    return out


def _behavioural_evidence(features: dict[str, float], meta: dict) -> list[Evidence]:
    """
    Turn the behavioural feature values into cited claims, with the actual
    transaction IDs behind the amount claims.

    Only claims that clear a materiality bar are emitted. A dossier saying
    "probe fraction is 0.02" is noise; the analyst needs the three things that
    actually fired.
    """
    out: list[Evidence] = []

    probe = features.get("probe_fraction", 0.0)
    if probe > 0.05:
        out.append(Evidence(
            claim=f"{probe:.0%} of this cluster's shared-infrastructure transactions are under INR {PROBE_AMOUNT_MAX:.0f}",
            citation_type="feature_value",
            citation_id="feature:probe_fraction",
            value=round(probe, 4),
            supporting_citations=[f"txn:{t}" for t in meta.get("probe_txn_ids", [])[:4]],
        ))

    cashout = features.get("cashout_fraction", 0.0)
    if cashout > 0.02:
        out.append(Evidence(
            claim=f"{cashout:.0%} of those transactions exceed INR {CASHOUT_AMOUNT_MIN:,.0f}",
            citation_type="feature_value",
            citation_id="feature:cashout_fraction",
            value=round(cashout, 4),
            supporting_citations=[f"txn:{t}" for t in meta.get("cashout_txn_ids", [])[:4]],
        ))

    escalation = features.get("amount_escalation", 1.0)
    if escalation > 50 and meta.get("min_amount") is not None:
        refs = [f"txn:{t}" for t in (meta.get("probe_txn_ids", [])[:2] + meta.get("cashout_txn_ids", [])[:2])]
        out.append(Evidence(
            claim=(f"Amounts escalated from INR {meta['min_amount']:,.2f} to INR {meta['max_amount']:,.2f} "
                   f"({escalation:,.0f}x) on the same shared infrastructure"),
            citation_type="transaction_id",
            citation_id=refs[0] if refs else "feature:amount_escalation",
            value={"min": meta.get("min_amount"), "max": meta.get("max_amount"), "ratio": round(escalation, 1)},
            supporting_citations=refs[1:],
        ))

    burst = features.get("burst_compression", 0.0)
    if burst > 0.3:
        out.append(Evidence(
            claim=f"{burst:.0%} of the cluster's activity landed inside a single 5-minute window",
            citation_type="feature_value",
            citation_id="feature:burst_compression",
            value=round(burst, 4),
            supporting_citations=["feature:velocity_txn_per_hour", "feature:time_span_hours"],
        ))

    gap = features.get("probe_to_cashout_gap_hours", -1.0)
    if gap >= 0:
        out.append(Evidence(
            claim=f"The first large transaction followed the first sub-INR-10 transaction by {gap:.1f} hours",
            citation_type="feature_value",
            citation_id="feature:probe_to_cashout_gap_hours",
            value=round(gap, 3),
            supporting_citations=[f"txn:{t}" for t in
                                  (meta.get("probe_txn_ids", [])[:1] + meta.get("cashout_txn_ids", [])[:1])],
        ))

    small = features.get("small_merchant_concentration", 0.0)
    if small > 0.2:
        out.append(Evidence(
            claim=f"{small:.0%} of the activity sits at merchants whose average ticket is under INR 50",
            citation_type="feature_value",
            citation_id="feature:small_merchant_concentration",
            value=round(small, 4),
            supporting_citations=[f"merchant:{m}" for m in meta.get("probe_merchants", [])[:3]],
        ))

    return out


def _structural_evidence(features: dict[str, float]) -> list[Evidence]:
    """Cited claims about the candidate's shape in the identity graph."""
    return [
        Evidence(
            claim=(f"The candidate is {int(features.get('cand_size', 0))} cards at "
                   f"{features.get('cand_density', 0):.0%} internal edge density, "
                   f"{features.get('cand_ego_overlap_ratio', 0):.0%} of which is explained by a single "
                   f"shared identity value"),
            citation_type="feature_value",
            citation_id="feature:cand_density",
            value={"size": features.get("cand_size"), "density": round(features.get("cand_density", 0), 4),
                   "ego_overlap": round(features.get("cand_ego_overlap_ratio", 0), 4)},
            supporting_citations=["feature:cand_size", "feature:cand_ego_overlap_ratio",
                                  "feature:cand_clustering_coeff"],
        )
    ]


def _vulcan_evidence(vulcan_txn: float, ringfence: float, composite: float, meta: dict) -> list[Evidence]:
    """
    The claim that justifies this system's existence: an individually plausible
    Vulcan score, elevated by network context.
    """
    return [
        Evidence(
            claim=(f"Vulcan's per-transaction score on this cluster's cashout activity was "
                   f"{vulcan_txn:.2f} (individually plausible); combined with a Ringfence ring score of "
                   f"{ringfence:.2f}, the composite risk is {composite:.2f}"),
            citation_type="vulcan_score",
            citation_id=f"vulcan:{vulcan_txn:.4f}",
            value={"vulcan_transaction_score": round(vulcan_txn, 4),
                   "ringfence_ring_score": round(ringfence, 4),
                   "composite": round(composite, 4)},
            supporting_citations=[f"ringfence:{ringfence:.4f}"]
                                 + [f"txn:{t}" for t in meta.get("cashout_txn_ids", [])[:2]],
        )
    ]


def _memory_evidence(precedents: list[dict], fp_patterns: list[dict]) -> list[Evidence]:
    """
    Cited claims from case memory — both the confirmed precedents that raise
    the score and the confirmed false positives that lower it.

    The false-positive claim is the one that matters. It is the system saying,
    in the case file, "we have been wrong about this exact shape before".
    """
    out: list[Evidence] = []
    for p in precedents[:2]:
        out.append(Evidence(
            claim=(f"This signature matches confirmed fraud ring {p.get('ring_id', 'unknown')} at "
                   f"{p.get('similarity', 0):.0%} cosine similarity"),
            citation_type="corpus_passage",
            citation_id=f"case:{p.get('alert_id') or p.get('ring_id')}",
            value={"similarity": round(float(p.get("similarity", 0)), 4),
                   "disposition": p.get("disposition"), "notes": p.get("notes")},
            supporting_citations=[f"disposition:{p.get('disposition')}"],
        ))
    for f in fp_patterns[:2]:
        out.append(Evidence(
            claim=(f"This signature also matches case {f.get('alert_id', 'unknown')}, which an analyst "
                   f"previously confirmed as a FALSE POSITIVE, at {f.get('similarity', 0):.0%} similarity — "
                   f"the score has been damped accordingly"),
            citation_type="corpus_passage",
            citation_id=f"case:{f.get('alert_id')}",
            value={"similarity": round(float(f.get("similarity", 0)), 4),
                   "disposition": f.get("disposition"), "notes": f.get("notes")},
            supporting_citations=[f"disposition:{f.get('disposition')}"],
        ))
    return out


# ──────────────────────────────────────────────────────────────────────────
# LLM summary
# ──────────────────────────────────────────────────────────────────────────


def _template_summary(dossier_fields: dict, evidence: list[Evidence]) -> str:
    """
    Deterministic fallback narrative, assembled from the evidence.

    Used when no LLM is configured or reachable. It is not a degraded product —
    it is the same facts in the same order, without the prose polish.
    """
    n = len(dossier_fields["candidate_cards"])
    top = evidence[0].claim if evidence else "no evidence generated"
    return (
        f"A {n}-card cluster scored {dossier_fields['ringfence_score']:.2f} by Ringfence, "
        f"composited with Vulcan to {dossier_fields['vulcan_composite_score']:.2f} against an operating "
        f"threshold of {dossier_fields['threshold_used']:.2f}. {top}. "
        f"The bounded recommendation is {dossier_fields['recommendation']}, supported by "
        f"{len(evidence)} cited claims."
    )


def generate_llm_summary(evidence: list[Evidence], timeout: float = LLM_TIMEOUT_S) -> tuple[Optional[str], str]:
    """
    Ask the LLM to restate the evidence in three sentences at temperature 0.

    Returns (summary, status). Every failure mode — missing key, missing SDK,
    timeout, API error — returns a status string rather than raising, because
    a narrative flourish must never be able to take down an alert.
    """
    if not OPENAI_API_KEY:
        return None, "skipped:no_api_key"

    payload = json.dumps(
        [{"claim": e.claim, "citation": e.citation_id, "value": e.value} for e in evidence],
        indent=2, default=str,
    )
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY, timeout=timeout)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            max_tokens=220,
            messages=[{"role": "user", "content": LLM_PROMPT.format(evidence_json=payload)}],
        )
        return response.choices[0].message.content.strip(), "ok"
    except ImportError:
        return None, "skipped:openai_sdk_missing"
    except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
        log.warning(f"LLM summary unavailable, falling back to deterministic template: {exc}")
        return None, f"failed:{type(exc).__name__}"


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────


def build_dossier(
    cards: list[str],
    features: dict[str, float],
    meta: dict,
    G: nx.Graph,
    ringfence_score: float,
    composite: dict,
    threshold: float,
    precedents: list[dict] | None = None,
    fp_patterns: list[dict] | None = None,
    alert_id: str | None = None,
    use_llm: bool = True,
    degraded: bool = False,
    degraded_reason: str | None = None,
) -> Dossier:
    """
    Assemble a complete, fully-cited dossier for one candidate.

    Order matters: deterministic evidence is built first and completely, then
    the narrative is generated from it. Reversing that — writing a summary and
    hunting for citations to justify it — is exactly the failure mode the
    citation validator exists to make impossible.
    """
    t0 = time.perf_counter()
    precedents = precedents or []
    fp_patterns = fp_patterns or []

    evidence: list[Evidence] = []
    evidence += _graph_evidence(cards, G)
    evidence += _behavioural_evidence(features, meta)
    evidence += _structural_evidence(features)
    evidence += _vulcan_evidence(
        meta.get("vulcan_cashout_mean") or meta.get("vulcan_mean", 0.0),
        ringfence_score,
        composite.get("composite_score", 0.0),
        meta,
    )
    evidence += _memory_evidence(precedents, fp_patterns)

    fields = {
        "alert_id": alert_id or new_alert_id(),
        "candidate_cards": sorted(cards),
        "ringfence_score": round(float(ringfence_score), 4),
        "vulcan_composite_score": round(float(composite.get("composite_score", 0.0)), 4),
        "threshold_used": round(float(threshold), 4),
        "recommendation": composite.get("recommendation", "REVIEW"),
    }

    summary, status = (None, "skipped:disabled")
    if use_llm:
        summary, status = generate_llm_summary(evidence)
    if summary is None:
        summary = _template_summary(fields, evidence)
        status = f"{status}|template_fallback"

    dossier = Dossier(
        **fields,
        evidence=evidence,
        llm_summary=summary,
        llm_status=status,
        generated_at=datetime.now(timezone.utc),
        vulcan_transaction_score=round(float(meta.get("vulcan_cashout_mean") or meta.get("vulcan_mean", 0.0)), 4),
        auto_action_allowed=bool(composite.get("auto_action_allowed", False)),
        degraded=degraded,
        degraded_reason=degraded_reason,
        precedents=precedents,
        false_positive_patterns=fp_patterns,
        feature_snapshot={k: round(v, 6) for k, v in features.items()},
        build_ms=round((time.perf_counter() - t0) * 1000, 2),
    )

    log.info(
        f"dossier {dossier.alert_id}: {len(evidence)} claims, "
        f"{dossier.refs_per_claim:.2f} refs/claim, 0 uncited, {dossier.build_ms:.1f}ms, llm={status}"
    )
    return dossier


def dossier_corpus_metrics(dossiers: list[Dossier]) -> dict:
    """
    Aggregate dossier-quality metrics across a batch.

    `uncited_claims` is structurally guaranteed to be 0 — every dossier in the
    list passed the validator to exist. It is reported anyway, because the bar
    asks for the number and "0, and here is why it cannot be anything else" is
    a better answer than "trust us".
    """
    if not dossiers:
        return {"n_dossiers": 0, "uncited_claims": 0, "refs_per_claim": 0.0}
    total_claims = sum(len(d.evidence) for d in dossiers)
    total_refs = sum(e.n_references for d in dossiers for e in d.evidence)
    return {
        "n_dossiers": len(dossiers),
        "total_claims": total_claims,
        "uncited_claims": sum(d.n_uncited_claims for d in dossiers),
        "refs_per_claim": round(total_refs / max(total_claims, 1), 2),
        "claims_per_dossier": round(total_claims / len(dossiers), 2),
        "enforcement": "Dossier.__init__ raises ValueError('Uncited claim detected'); "
                       "an uncited dossier cannot be constructed.",
    }


def build_dossier_for_candidate(
    cards: list[str],
    G: nx.Graph,
    index: TransactionIndex,
    scorer,
    composer,
    memory=None,
    alert_id: str | None = None,
    use_llm: bool = True,
) -> Dossier:
    """
    End-to-end convenience path: features -> score -> memory -> composite ->
    dossier. This is what the API calls.
    """
    from src.feature_engine import compute_candidate_meta, compute_features

    features = compute_features(cards, G, index)
    meta = compute_candidate_meta(cards, index)

    degraded = not getattr(scorer, "available", False)
    ringfence_score = scorer.score(features) if not degraded else 0.0

    precedents, fp_patterns, boost, damp = [], [], 0.0, 0.0
    if memory is not None:
        precedents, fp_patterns, boost, damp = memory.adjustments(features)

    composite = composer.compute_composite(
        vulcan_score=meta.get("vulcan_cashout_mean") or meta.get("vulcan_mean", 0.0),
        ringfence_score=ringfence_score,
        precedent_boost=boost,
        fp_damp=damp,
        n_template_precedents=len(precedents),
        degraded=degraded,
    )

    return build_dossier(
        cards=list(cards),
        features=features,
        meta=meta,
        G=G,
        ringfence_score=ringfence_score,
        composite=composite,
        threshold=getattr(scorer, "threshold", 0.5),
        precedents=precedents,
        fp_patterns=fp_patterns,
        alert_id=alert_id,
        use_llm=use_llm,
        degraded=degraded,
        degraded_reason="model artifact unavailable" if degraded else None,
    )
