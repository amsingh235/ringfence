"""
Put the demo back to a clean recording state, without losing the alert queue.

`clean-memory` deletes case_memory.sqlite *and* alerts.sqlite. That re-arms the
failure demo, but it also empties the queue, and refilling it means running
seed_alerts.py again — minutes you do not have between takes.

Everything that makes the demo look "already used" lives in two places:

  1. case_memory.sqlite — stored analyst verdicts. Any row here damps the
     failure-recovery candidate, so page 5 shows the post-click numbers before
     the button is pressed and clicking it changes nothing on camera.
  2. alerts.disposition — set when you press the button. Non-null values make
     page 1 report "Dispositioned: N" and annotate rows "analyst: false positive".

This clears both and leaves the alert rows alone.

    python demo/reset_demo.py

Safe to run between takes. Restart the dashboard afterwards so Streamlit drops
its cached queue.
"""
from __future__ import annotations

import sqlite3
import sys

from src.config import ALERTS_DB, CASE_MEMORY_DB


def main() -> int:
    """Clear analyst verdicts and dispositions, keeping every seeded alert."""
    # Delete the rows rather than the file. On Windows the running API holds an
    # open handle to both databases, and unlink() fails with WinError 32 — so a
    # reset would mean stopping the services, which is the opposite of the point.
    if CASE_MEMORY_DB.exists():
        con = sqlite3.connect(str(CASE_MEMORY_DB))
        try:
            tables = [
                r[0]
                for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%'"
                )
            ]
            before = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
            for t in tables:
                con.execute(f"DELETE FROM {t}")
            con.commit()
        finally:
            con.close()
        cleared = ", ".join(f"{t} {n}" for t, n in before.items() if n)
        print(f"case memory cleared ({cleared or 'already empty'}) — failure demo re-armed")
    else:
        print(f"{CASE_MEMORY_DB.name} not present — nothing to clear")

    if not ALERTS_DB.exists():
        print(f"{ALERTS_DB.name} not present — run demo/seed_alerts.py to build a queue")
        return 0

    con = sqlite3.connect(str(ALERTS_DB))
    try:
        total = con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        n = con.execute(
            "SELECT COUNT(*) FROM alerts WHERE disposition IS NOT NULL"
        ).fetchone()[0]
        con.execute("UPDATE alerts SET disposition = NULL WHERE disposition IS NOT NULL")
        con.commit()
    finally:
        con.close()

    print(f"cleared {n} disposition(s); {total} alerts kept")
    print()
    print("Restart the dashboard so Streamlit drops its cached queue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
