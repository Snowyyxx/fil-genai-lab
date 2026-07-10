"""
Stage 1 — Extract placement records from the live site to a CSV.

WHY THIS FILE EXISTS
--------------------
The website https://placements25-26.vercel.app is a React single-page app.
If you `curl` the page you get almost-empty HTML: the data is NOT in the page.
The React code fetches it at runtime from Google Firestore (a NoSQL cloud DB),
project id "tiet-placement2526".

There are two ways to reach that data:
  1. The site's own proxy endpoint:  GET /api/placements
  2. Firestore's public REST API directly:
     https://firestore.googleapis.com/v1/projects/<proj>/databases/(default)/documents/<collection>

We try the proxy first (it returns clean JSON), and fall back to the raw
Firestore REST API (which is paginated and more verbose). Both are read-only
public endpoints — the same ones your browser hits when you open the site.

REAL-WORLD LESSON: public data sources rate-limit you. Firestore's free tier
caps daily reads, so you will see HTTP 429 ("quota exceeded"). The answer is
*polite retry with exponential backoff*, which is built in below. If the daily
quota is fully exhausted it resets at midnight US-Pacific — just run again then.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config — these values were read straight out of the site's JS bundle.
# (Firebase web config is public by design; it is not a secret.)
# ---------------------------------------------------------------------------
PROXY_URL = "https://placements25-26.vercel.app/api/placements"
FIREBASE_API_KEY = "AIzaSyBj-BRayJ6tifplF37gIWiwoNKtR5wUBMo"
FIREBASE_PROJECT = "tiet-placement2526"
COLLECTION = "placements"

OUT_CSV = Path(__file__).parent / "placements.csv"

# Backoff schedule: how many attempts and how long to wait between them.
MAX_ATTEMPTS = 8
BASE_DELAY_SECONDS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fetch_via_proxy() -> list[dict] | None:
    """Try the site's own /api/placements endpoint. Returns list of dicts or None."""
    r = requests.get(PROXY_URL, timeout=30)
    if r.status_code != 200:
        return None
    body = r.json()
    # The proxy wraps data as {"ok": true, "placements": [...], "branchStats": [...]}
    if not body.get("ok"):
        return None
    return body.get("placements") or body.get("data") or []


def _decode_firestore_value(v: dict):
    """Firestore REST wraps every value in a type tag, e.g. {"stringValue": "x"}
    or {"integerValue": "42"}. This unwraps it back to a plain Python value."""
    if "stringValue" in v:
        return v["stringValue"]
    if "integerValue" in v:
        return int(v["integerValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "booleanValue" in v:
        return v["booleanValue"]
    if "timestampValue" in v:
        return v["timestampValue"]
    if "nullValue" in v:
        return None
    if "arrayValue" in v:
        return [_decode_firestore_value(x) for x in v["arrayValue"].get("values", [])]
    if "mapValue" in v:
        return {k: _decode_firestore_value(x)
                for k, x in v["mapValue"].get("fields", {}).items()}
    return None


def fetch_via_firestore() -> list[dict] | None:
    """Fallback: page through the raw Firestore REST API. Returns list of dicts."""
    base = (f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT}"
            f"/databases/(default)/documents/{COLLECTION}")
    rows: list[dict] = []
    page_token = None
    while True:
        params = {"pageSize": 300, "key": FIREBASE_API_KEY}
        if page_token:
            params["pageToken"] = page_token
        r = requests.get(base, params=params, timeout=30)
        if r.status_code != 200:
            return None  # let the caller retry the whole thing
        body = r.json()
        for doc in body.get("documents", []):
            fields = {k: _decode_firestore_value(v)
                      for k, v in doc.get("fields", {}).items()}
            rows.append(fields)
        page_token = body.get("nextPageToken")
        if not page_token:
            break
    return rows


def fetch_with_backoff() -> list[dict]:
    """Try proxy, then firestore, retrying with exponential backoff on failure."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        for name, fn in (("proxy", fetch_via_proxy), ("firestore", fetch_via_firestore)):
            try:
                rows = fn()
            except requests.RequestException as e:
                print(f"  attempt {attempt} via {name}: network error {e}")
                rows = None
            if rows:
                print(f"  ✓ got {len(rows)} records via {name}")
                return rows
        delay = BASE_DELAY_SECONDS * (2 ** (attempt - 1))
        print(f"attempt {attempt}/{MAX_ATTEMPTS} failed (likely HTTP 429). "
              f"waiting {delay}s...")
        time.sleep(delay)
    raise SystemExit(
        "Could not fetch data after retries. The Firestore daily read quota is "
        "probably exhausted — it resets at midnight US-Pacific. Try again later."
    )


def write_csv(rows: list[dict], path: Path) -> None:
    """Flatten the records into a CSV. We compute the union of all keys so that
    records with missing fields still line up under a stable header."""
    # Stable, human-friendly column order; any extra keys get appended.
    preferred = ["company", "role", "type", "ctc", "package", "stipend",
                 "branch", "cgpa", "status", "date", "location", "name"]
    all_keys: list[str] = []
    seen = set()
    for k in preferred:
        if any(k in row for row in rows):
            all_keys.append(k)
            seen.add(k)
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Serialize nested dict/list values so the CSV cell stays a string.
            clean = {k: (v if not isinstance(v, (dict, list)) else str(v))
                     for k, v in row.items()}
            writer.writerow(clean)
    print(f"✓ wrote {len(rows)} rows × {len(all_keys)} cols -> {path}")


def main() -> None:
    print("Fetching placement records...")
    rows = fetch_with_backoff()
    write_csv(rows, OUT_CSV)
    # Show a tiny preview so you can eyeball the data.
    print("\nPreview (first record):")
    for k, v in list(rows[0].items())[:12]:
        print(f"  {k:12} = {v}")


if __name__ == "__main__":
    main()
