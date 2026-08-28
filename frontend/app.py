"""
Ringfence demo dashboard.

Five pages, in the order a panel should see them:

  1. Live Alert Queue     — what an analyst opens in the morning
  2. Dossier Viewer       — the audit trail, per alert
  3. Network Explorer     — the identity graph behind a candidate
  4. Metrics              — recall, the cost curve, feature gain, latency
  5. Failure Recovery     — one false positive, handled, with the numbers

Data comes from the FastAPI service when it is reachable and falls back to
reading the local artifacts directly when it is not. That is deliberate: the
demo has to survive a dead API on someone else's laptop, and the fallback path
exercises the same functions the API calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import API_BASE_URL, METRICS_JSON, PROCESSED_DIR  # noqa: E402
from frontend.components.alert_card import render_alert_queue  # noqa: E402
from frontend.components.dossier_viewer import render_dossier_page  # noqa: E402
from frontend.components.failure_demo import render_failure_recovery  # noqa: E402
from frontend.components.metrics_panel import render_metrics_page  # noqa: E402
from frontend.components.network_explorer import render_network_page  # noqa: E402

st.set_page_config(
    page_title="Ringfence — Abuse-Ring Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark-mode-friendly accents that also survive light mode: we set colours on
# semantic classes only, and let Streamlit's own theme own the background.
st.markdown(
    """
    <style>
      .rf-badge { display:inline-block; padding:2px 10px; border-radius:12px;
                  font-size:0.78rem; font-weight:600; letter-spacing:0.02em; }
      .rf-block   { background:#7f1d1d; color:#fecaca; }
      .rf-review  { background:#78350f; color:#fde68a; }
      .rf-approve { background:#14532d; color:#bbf7d0; }
      .rf-degraded{ background:#334155; color:#e2e8f0; }
      .rf-kpi { font-size:2.0rem; font-weight:700; line-height:1.1; }
      .rf-kpi-label { font-size:0.8rem; opacity:0.7; text-transform:uppercase;
                      letter-spacing:0.06em; }
      .rf-cite { font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
                 font-size:0.78rem; opacity:0.85; }
      div[data-testid="stMetricValue"] { font-size:1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────
# Data access
# ──────────────────────────────────────────────────────────────────────────


class Backend:
    """
    Talks to the API when it is up, reads artifacts from disk when it is not.

    `online` is surfaced in the sidebar so nobody demos an offline dashboard
    while believing they are exercising the service.
    """

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.online = False
        self.health: dict = {}
        self._probe()

    def _probe(self) -> None:
        """Check whether the API answers /health."""
        try:
            import httpx

            r = httpx.get(f"{self.base_url}/health", timeout=2.0)
            self.health = r.json()
            self.online = True
        except Exception:  # noqa: BLE001 - offline is an expected state
            self.online = False

    def get(self, path: str, **params):
        """GET from the API, or None when offline."""
        if not self.online:
            return None
        try:
            import httpx

            r = httpx.get(f"{self.base_url}{path}", params=params, timeout=15.0)
            return r.json() if r.status_code < 400 else None
        except Exception:  # noqa: BLE001
            return None

    def post(self, path: str, payload: dict):
        """POST to the API, or None when offline."""
        if not self.online:
            return None
        try:
            import httpx

            r = httpx.post(f"{self.base_url}{path}", json=payload, timeout=30.0)
            return r.json() if r.status_code < 400 else {"_error": r.text, "_status": r.status_code}
        except Exception as exc:  # noqa: BLE001
            return {"_error": str(exc)}


@st.cache_resource(show_spinner="Loading identity graph and transaction index…")
def load_local_artifacts() -> dict:
    """
    Load the pipeline artifacts directly.

    Cached as a resource so the graph and the 700k-row transaction index are
    built once per Streamlit session rather than once per rerun.
    """
    from src.candidate_generator import load_candidates
    from src.data_generator import load_ground_truth, load_transactions
    from src.feature_engine import build_index
    from src.graph_builder import load_graph
    from src.scorer import RingfenceScorer
    from src.config import MERCHANTS_CSV

    out: dict = {"errors": {}}
    for name, loader in (
        ("graph", load_graph),
        ("candidates", load_candidates),
        ("rings", load_ground_truth),
        ("transactions", load_transactions),
    ):
        try:
            out[name] = loader()
        except Exception as exc:  # noqa: BLE001
            out[name] = None
            out["errors"][name] = str(exc)

    try:
        merchants = pd.read_csv(MERCHANTS_CSV) if MERCHANTS_CSV.exists() else None
        out["index"] = build_index(out["transactions"], merchants) if out["transactions"] is not None else None
    except Exception as exc:  # noqa: BLE001
        out["index"] = None
        out["errors"]["index"] = str(exc)

    out["scorer"] = RingfenceScorer()

    try:
        out["metrics"] = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        out["metrics"] = {}

    for name, fname in (
        ("oof", "oof_predictions.csv"),
        ("labels", "candidate_labels.csv"),
        ("features", "features.csv"),
    ):
        path = PROCESSED_DIR / fname
        out[name] = pd.read_csv(path) if path.exists() else None

    for name, fname in (("candidate_metrics", "candidate_metrics.json"),
                        ("collision_stats", "collision_stats.json"),
                        ("feature_latency", "feature_latency.json")):
        path = PROCESSED_DIR / fname
        out[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    return out


def badge(recommendation: str, degraded: bool = False) -> str:
    """Render a recommendation as a coloured HTML pill."""
    if degraded:
        return '<span class="rf-badge rf-degraded">DEGRADED · REVIEW</span>'
    cls = {"BLOCK": "rf-block", "REVIEW": "rf-review", "APPROVE": "rf-approve"}.get(recommendation, "rf-degraded")
    return f'<span class="rf-badge {cls}">{recommendation}</span>'


# ──────────────────────────────────────────────────────────────────────────
# Shell
# ──────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Render the sidebar and dispatch to the selected page."""
    backend = Backend()
    artifacts = load_local_artifacts()

    if "selected_alert" not in st.session_state:
        st.session_state.selected_alert = None

    with st.sidebar:
        st.markdown("## 🛡️ Ringfence")
        st.caption("Abuse-Ring Sentinel · Track 02 — AI Risk Manager")

        if backend.online:
            status = backend.health.get("status", "unknown")
            st.success(f"API online · {status}") if status == "ok" else st.warning(f"API online · {status}")
            st.caption(f"model `{backend.health.get('model_version', '?')}`")
        else:
            st.info("API offline — reading local artifacts")
            st.caption(f"expected at `{backend.base_url}`")

        page = st.radio(
            "Page",
            ["1 · Alert Queue", "2 · Dossier Viewer", "3 · Network Explorer",
             "4 · Metrics", "5 · Failure Recovery"],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption(
            "**Defense-only.** This system observes and recommends. "
            "It cannot block a payment, decline a card, or move money."
        )
        if artifacts["errors"]:
            with st.expander("Artifact warnings"):
                for k, v in artifacts["errors"].items():
                    st.caption(f"`{k}`: {v}")

    if page.startswith("1"):
        render_alert_queue(backend, artifacts, badge)
    elif page.startswith("2"):
        render_dossier_page(backend, artifacts, badge)
    elif page.startswith("3"):
        render_network_page(backend, artifacts)
    elif page.startswith("4"):
        render_metrics_page(backend, artifacts)
    else:
        render_failure_recovery(backend, artifacts, badge)


if __name__ == "__main__":
    main()
