#!/usr/bin/env python3
"""Backfill existing submissions into the central collection repo.

Queries the Workers API for submissions and triggers GitHub Actions
workflow_dispatch for each one.

Usage:
    python backfill_submissions.py --week 1                # Backfill week 1
    python backfill_submissions.py --week 1 --dry-run      # Preview only
    python backfill_submissions.py                          # All weeks
    python backfill_submissions.py --delay 5                # 5s between triggers

Environment variables (or .env file):
    WORKERS_API_BASE  - Workers API URL
    ADMIN_API_KEY     - Admin API key for fetching submissions
    GITHUB_PAT        - GitHub PAT for triggering workflow_dispatch
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

OWNER = "iamseungpil"
REPO = "claude-code-study-submissions"
WORKFLOW = "collect-submission.yml"


def load_env():
    """Load .env file from project root if it exists."""
    env_file = SCRIPT_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


load_env()

API_BASE = os.environ.get(
    "WORKERS_API_BASE",
    "https://claude-code-study-api.iamseungpil.workers.dev",
)
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
GITHUB_PAT = os.environ.get("GITHUB_PAT", "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fetch_submissions(week: int) -> list[dict]:
    """Fetch submissions for a given week from the Workers API."""
    url = f"{API_BASE}/api/submissions/{week}"
    req = urllib.request.Request(url)
    req.add_header("X-Admin-Key", ADMIN_API_KEY)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"  [ERROR] Failed to fetch week {week}: HTTP {e.code}")
        return []


def trigger_dispatch(week: int, user_id: str, github_url: str) -> bool:
    """Trigger the collection workflow via GitHub API."""
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/workflows/{WORKFLOW}/dispatches"
    payload = json.dumps({
        "ref": "main",
        "inputs": {
            "week": str(week),
            "user_id": user_id,
            "github_url": github_url,
        },
    }).encode()

    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"Bearer {GITHUB_PAT}")
    req.add_header("User-Agent", "backfill-submissions")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status == 204
    except urllib.error.HTTPError as e:
        print(f"  [ERROR] Dispatch failed for {user_id} week {week}: HTTP {e.code}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Backfill submissions to collection repo")
    parser.add_argument("--week", type=int, help="Specific week to backfill (default: all weeks 1-5)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without triggering")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between triggers (default: 3)")
    args = parser.parse_args()

    if not ADMIN_API_KEY:
        print("ERROR: ADMIN_API_KEY not set. Add it to .env or environment.")
        sys.exit(1)
    if not GITHUB_PAT and not args.dry_run:
        print("ERROR: GITHUB_PAT not set. Add it to .env or environment.")
        sys.exit(1)

    weeks = [args.week] if args.week else list(range(1, 6))

    total = 0
    triggered = 0

    for week in weeks:
        print(f"\n=== Week {week} ===")
        submissions = fetch_submissions(week)

        if not submissions:
            print("  No submissions found.")
            continue

        for sub in submissions:
            user_id = sub.get("participant_id", "")
            github_url = sub.get("github_url", "")

            if not user_id or not github_url:
                continue

            total += 1

            if args.dry_run:
                print(f"  [DRY-RUN] week{week}/{user_id} <- {github_url}")
                continue

            print(f"  Triggering week{week}/{user_id} <- {github_url}")
            ok = trigger_dispatch(week, user_id, github_url)
            if ok:
                triggered += 1
                print(f"    -> dispatched")
            else:
                print(f"    -> FAILED")

            time.sleep(args.delay)

    print(f"\nDone. Total: {total}, Triggered: {triggered}")


if __name__ == "__main__":
    main()
