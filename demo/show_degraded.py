"""
Demo helper — ask the API about a card that does not exist.

The point of the exhibit: a fraud API that returns 500 is worse than useless,
because the caller times out and the transaction goes through anyway. Ringfence
degrades to REVIEW and says so. Kept as a script so the on-camera command is one
short line instead of a curl pipeline with escaping.

    python demo/show_degraded.py
"""
import json
import sys
import urllib.error
import urllib.request

URL = "http://localhost:8000/detect/candidate"
BODY = json.dumps({"cards": ["CARD_NOT_REAL"]}).encode()


def main() -> int:
    """Call /detect/candidate with an unknown card and print the safe fallback."""
    req = urllib.request.Request(
        URL, data=BODY, headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            status, payload = r.status, json.load(r)
    except urllib.error.URLError as e:
        print(f"API unreachable at {URL} — is `make run` up?  ({e})")
        return 1

    print(f"HTTP status     : {status}          <- not a 500")
    print(f"degraded        : {payload.get('degraded')}")
    print(f"recommendation  : {payload['composite']['recommendation']}")
    print(f"auto_action     : {payload['composite'].get('auto_action_allowed')}")
    print()
    print("A detector that cannot see escalates to a human. It never quietly approves.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
