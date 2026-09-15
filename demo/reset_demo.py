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

Works from any shell with no PYTHONPATH. If the compose stack is running, it
resets the CONTAINER's databases (the ones the dashboard at :8501 actually
reads) and restarts the dashboard; pass --local to force the host copy.

Safe to run between takes. Restart the dashboard afterwards so Streamlit drops
its cached queue.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

# Make `src` importable no matter where this is launched from, so the script
# does not depend on PYTHONPATH being set in the caller's shell.
# When piped into a container over stdin there is no __file__; the image sets
# PYTHONPATH=/app and runs from /app, so the cwd is the project root there.
ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ALERTS_DB, CASE_MEMORY_DB  # noqa: E402


def _reset_inside_docker() -> bool:
    """
    If the demo is running in Docker, reset THAT copy of the databases.

    The compose stack keeps its data in a named volume, not in the host's
    data/ directory. Resetting the host files while the dashboard at :8501 is
    served from a container would print "cleared" and change nothing on
    screen. So when the api container is up, re-run this script inside it.

    Returns True if the reset was delegated to the container.
    """
    if "--local" in sys.argv or os.environ.get("RINGFENCE_RESET_LOCAL") == "1":
        return False
    docker = _find_docker()
    if docker is None:
        return False
    try:
        probe = subprocess.run(
            [docker, "compose", "ps", "--status", "running", "--services"],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if probe.returncode != 0 or "api" not in probe.stdout.split():
        return False

    print("demo is running in Docker -> resetting the container's databases")
    # Pipe this script's own source in, so it works even if the image was built
    # before this file existed. RINGFENCE_RESET_LOCAL stops the inner run from
    # trying to delegate to Docker again.
    source = Path(__file__).read_text(encoding="utf-8") if "__file__" in globals() else ""
    result = subprocess.run(
        [docker, "compose", "exec", "-T", "-e", "RINGFENCE_RESET_LOCAL=1", "api", "python", "-"],
        cwd=ROOT, input=source, text=True, encoding="utf-8",
    )
    if result.returncode == 0:
        print("restarting the dashboard container so Streamlit drops its cache")
        subprocess.run([docker, "compose", "restart", "demo"], cwd=ROOT, capture_output=True)
    return True


def _find_docker() -> str | None:
    """
    Locate the docker CLI. PATH first; then Docker Desktop's install location,
    because some shells (Git Bash launching a venv python) do not expose it.
    """
    found = shutil.which("docker")
    if found:
        return found
    for candidate in (
        Path(r"C:/Program Files/Docker/Docker/resources/bin/docker.exe"),
        Path("/usr/local/bin/docker"),
        Path("/usr/bin/docker"),
    ):
        if candidate.exists():
            return str(candidate)
    return None


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
    if _reset_inside_docker():
        sys.exit(0)
    sys.exit(main())
