"""
Fetch and clean several seasons by running the existing, tested scripts.
Safe to rerun: stored pages are reused and the cleaning scripts use upserts.

Usage, from the project root with the virtual environment active:
    python pipeline/run_seasons.py 2014 2025
"""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

PIPELINE_DIR = Path(__file__).resolve().parent
load_dotenv(PIPELINE_DIR.parent / ".env")

FIRST_ALLOWED, LAST_ALLOWED, MAX_SEASONS = 2014, 2025, 15


def run(script, argument):
    """Run one of our scripts in the same Python environment."""
    print(f"\n>>> {script} {argument}", flush=True)
    result = subprocess.run([sys.executable, str(PIPELINE_DIR / script), argument])
    if result.returncode != 0:
        sys.exit(
            f"\nStopped: {script} {argument} failed (exit code {result.returncode}).\n"
            "Fix the error shown above, then rerun the same command. "
            "Work already finished is kept."
        )


def print_summary(first, last):
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB_NAME", "f1_intelligence")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        db = client[db_name]
        counts = {}
        for name in ("races", "race_results", "qualifying_results"):
            grouped = db[name].aggregate([{"$group": {"_id": "$season", "n": {"$sum": 1}}}])
            counts[name] = {doc["_id"]: doc["n"] for doc in grouped}

        print("\nSeason  races  race_results  qualifying_results")
        for season in range(first, last + 1):
            print(
                f"{season}   {counts['races'].get(season, 0):>5}  "
                f"{counts['race_results'].get(season, 0):>12}  "
                f"{counts['qualifying_results'].get(season, 0):>18}"
            )
    except PyMongoError as error:
        print(f"Summary failed: {type(error).__name__}: {str(error)[:300]}")
    finally:
        client.close()


def main():
    if len(sys.argv) != 3 or not (sys.argv[1].isdigit() and sys.argv[2].isdigit()):
        sys.exit("Usage: python pipeline/run_seasons.py 2014 2025")
    first, last = int(sys.argv[1]), int(sys.argv[2])
    if not (FIRST_ALLOWED <= first <= last <= LAST_ALLOWED):
        sys.exit(f"Seasons must satisfy {FIRST_ALLOWED} <= first <= last <= {LAST_ALLOWED}")
    if last - first + 1 > MAX_SEASONS:
        sys.exit(f"At most {MAX_SEASONS} seasons per run, to respect the API rate limit.")

    for season in range(first, last + 1):
        for endpoint in ("races", "results", "qualifying"):
            run("ingest_raw.py", f"{season}/{endpoint}")
        run("clean_results.py", str(season))
        run("clean_qualifying.py", str(season))

    print_summary(first, last)
    print("\nAll seasons finished.")


if __name__ == "__main__":
    main()