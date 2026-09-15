"""
Populate the alert queue through the API.

`clean-memory` deletes alerts.sqlite along with case memory, which is correct —
it is the only way to re-arm the failure-recovery demo. But it also empties the
alert queue, and nothing refills it on its own: an alert row is only written
when something is actually scored (src/api.py save_alert). A fresh clone has the
same problem.

Without this, page 1 silently falls back to scoring 40 candidates in-process and
labels them "scored locally", which is true but reads as a broken demo next to a
green "API online" badge.

    python demo/seed_alerts.py              # every candidate
    python demo/seed_alerts.py --limit 150  # the first 150, if you are in a hurry

Idempotent: alert_id is derived per call, so re-running adds new rows. Run it
once after clean-memory, not repeatedly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Make `src` importable from any shell, without relying on PYTHONPATH.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API = "http://localhost:8000"


def _post(path: str, payload: dict, timeout: float = 60.0) -> dict:
    """POST JSON and return the decoded response."""
    req = urllib.request.Request(
        API + path,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    """Score every persisted candidate through the API so the queue has rows."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="stop after N candidates (0 = all)")
    args = ap.parse_args()

    try:
        with urllib.request.urlopen(API + "/health", timeout=10) as r:
            health = json.load(r)
    except urllib.error.URLError as exc:
        print(f"API unreachable at {API} — start it first (.\\run.ps1 run).  {exc}")
        return 1

    if health.get("status") != "ok":
        print(f"warning: API status={health.get('status')!r} errors={health.get('errors')} — seeding anyway")

    from src.candidate_generator import load_candidates

    candidates = load_candidates()
    if args.limit:
        candidates = candidates[: args.limit]
    total = len(candidates)
    if not total:
        print("no candidates on disk — run the pipeline first (.\\run.ps1 all)")
        return 1

    print(f"seeding {total} candidates through {API}/detect/candidate …")
    t0 = time.perf_counter()
    counts: dict[str, int] = {}
    failed = 0

    for i, cand in enumerate(candidates, 1):
        try:
            resp = _post("/detect/candidate", {"cards": list(cand.cards)})
            rec = resp.get("composite", {}).get("recommendation", "?")
            counts[rec] = counts.get(rec, 0) + 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if failed <= 3:
                print(f"  ! {cand.candidate_id}: {type(exc).__name__}: {exc}")
        if i % 50 == 0 or i == total:
            print(f"  {i}/{total}  ({time.perf_counter() - t0:.0f}s)")

    print()
    print(f"done in {time.perf_counter() - t0:.0f}s — {total - failed} alerts, {failed} failed")
    for rec in ("BLOCK", "REVIEW", "APPROVE"):
        if rec in counts:
            print(f"  {rec:8s} {counts[rec]}")
    print()
    print("Reload the dashboard. Page 1 should now say '… alerts from the API'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
