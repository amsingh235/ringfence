"""
Shared pytest fixtures.

The whole suite runs against a **test-scale universe** generated once per
session into a temporary directory: 300 cards, 24k transactions, 25 rings, five
worlds. Same code paths as the full 737k-transaction run, roughly 1/30th the
wall clock.

Nothing here touches the developer's real `data/` directory. `RINGFENCE_DATA_DIR`
is redirected before `src.config` is imported, so a test run cannot clobber the
artifacts a demo is about to use.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Redirect all artifact paths BEFORE src.config is imported anywhere.
_TMP = Path(tempfile.mkdtemp(prefix="ringfence_tests_"))
os.environ["RINGFENCE_DATA_DIR"] = str(_TMP)
os.environ.setdefault("RINGFENCE_LOG_LEVEL", "WARNING")
os.environ.pop("OPENAI_API_KEY", None)  # tests must never make a network call


@pytest.fixture(scope="session")
def gen_config():
    """Test-scale generation config."""
    from src.config import GenerationConfig

    return GenerationConfig.small()


@pytest.fixture(scope="session")
def universe(gen_config):
    """Generate and persist the synthetic universe once for the whole session."""
    from src.data_generator import generate_universe, save_universe

    u = generate_universe(gen_config, seed=42)
    save_universe(u)
    return u


@pytest.fixture(scope="session")
def transactions(universe):
    """The generated transaction table."""
    return universe.transactions


@pytest.fixture(scope="session")
def cards_df(universe):
    """The generated card table."""
    return universe.cards


@pytest.fixture(scope="session")
def merchants_df(universe):
    """The generated merchant table."""
    return universe.merchants


@pytest.fixture(scope="session")
def rings_df(universe):
    """Planted ring ground truth, with `card_id_list` exploded."""
    df = universe.rings.copy()
    df["card_id_list"] = df["card_ids"].apply(lambda s: s.split("|"))
    return df


@pytest.fixture(scope="session")
def graph(transactions, cards_df):
    """The collision-capped identity graph."""
    from src.graph_builder import build_identity_graph, save_graph

    G = build_identity_graph(transactions, cards_df=cards_df)
    save_graph(G)
    return G


@pytest.fixture(scope="session")
def candidates(graph):
    """Deduplicated candidate sets from all three generators."""
    from src.candidate_generator import generate_candidates

    return generate_candidates(graph)


@pytest.fixture(scope="session")
def labels(candidates, rings_df):
    """Candidate labels joined to planted rings."""
    from src.candidate_generator import label_candidates
    from src.config import PROCESSED_DIR

    df = label_candidates(candidates, rings_df)
    df.to_csv(PROCESSED_DIR / "candidate_labels.csv", index=False)
    return df


@pytest.fixture(scope="session")
def index(transactions, merchants_df):
    """Columnar transaction index."""
    from src.feature_engine import build_index

    return build_index(transactions, merchants_df)


@pytest.fixture(scope="session")
def features(candidates, graph, index):
    """Feature store for every candidate."""
    from src.config import FEATURES_CSV
    from src.feature_engine import compute_features_batch

    df = compute_features_batch(candidates, graph, index)
    df.to_csv(FEATURES_CSV, index=False)
    return df


@pytest.fixture(scope="session")
def trained(features, labels):
    """A trained model artifact plus its metrics and out-of-fold predictions."""
    from src.scorer import save_artifact, train

    artifact, metrics, oof = train(features, labels)
    save_artifact(artifact)
    return {"artifact": artifact, "metrics": metrics, "oof": oof}


@pytest.fixture
def memory(tmp_path):
    """A fresh, isolated case-memory database per test."""
    from src.case_memory import CaseMemory

    m = CaseMemory(db_path=tmp_path / "case_memory.sqlite")
    yield m
    m.close()


@pytest.fixture(scope="session")
def positive_candidate(candidates, labels):
    """A candidate that genuinely covers a planted ring."""
    ids = set(labels.loc[labels["is_fraud_ring"] == 1, "candidate_id"])
    match = next((c for c in candidates if c.candidate_id in ids), None)
    assert match is not None, "no positive candidate in the test universe"
    return match


@pytest.fixture(scope="session")
def negative_candidate(candidates, labels):
    """A candidate that does not cover any planted ring."""
    ids = set(labels.loc[labels["is_fraud_ring"] == 0, "candidate_id"])
    match = next((c for c in candidates if c.candidate_id in ids), None)
    assert match is not None, "no negative candidate in the test universe"
    return match
