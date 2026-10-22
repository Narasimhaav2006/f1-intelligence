"""
Build the qualifying_results collection from raw pages already in MongoDB.
Makes no API requests. Safe to run more than once (it uses upserts).

Usage, from the project root with the virtual environment active:
    python pipeline/clean_qualifying.py 2023

Requires raw pages from:
    python pipeline/ingest_raw.py 2023/qualifying
Run pipeline/clean_results.py first so the drivers collection exists.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Reuse the helpers written for the race results (same folder).
from clean_results import to_int, upsert

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

log = logging.getLogger("clean_qualifying")


def lap_time_to_seconds(text):
    """Convert '1:31.295' to 91.295. Returns None if empty or unreadable."""
    if not text:
        return None
    try:
        if ":" in text:
            minutes, seconds = text.split(":", 1)
            return round(int(minutes) * 60 + float(seconds), 3)
        return round(float(text), 3)
    except ValueError:
        return None


def build_qualifying_doc(season, round_number, row):
    driver = row["Driver"]
    team = row["Constructor"]
    q1 = lap_time_to_seconds(row.get("Q1"))
    q2 = lap_time_to_seconds(row.get("Q2"))
    q3 = lap_time_to_seconds(row.get("Q3"))
    times = [t for t in (q1, q2, q3) if t is not None]
    return {
        "season": season,
        "round": round_number,
        "driver_id": driver["driverId"],
        "constructor_id": team["constructorId"],
        "driver_name": f"{driver['givenName']} {driver['familyName']}",
        "constructor_name": team["name"],
        "position": to_int(row["position"]),
        "q1_seconds": q1,
        "q2_seconds": q2,
        "q3_seconds": q3,
        "best_seconds": min(times) if times else None,
    }


def main():
    season_text = sys.argv[1] if len(sys.argv) > 1 else "2023"
    if not season_text.isdigit():
        sys.exit("Season must be a number such as 2023")
    season = int(season_text)

    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB_NAME", "f1_intelligence")
    if not uri or uri.startswith("replace-with"):
        sys.exit("MONGODB_URI is missing or still a placeholder. Check your .env file.")

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        db = client[db_name]

        pages = list(db["raw_jolpica"].find({"endpoint": f"{season}/qualifying"}))
        if not pages:
            sys.exit(
                f"No raw pages for {season}/qualifying. "
                f"Run: python pipeline/ingest_raw.py {season}/qualifying"
            )

        # Group by (season, round) so a race split across pages is still handled.
        grouped = {}
        for page in pages:
            for race in page["payload"]["MRData"]["RaceTable"]["Races"]:
                key = (int(race["season"]), int(race["round"]))
                grouped.setdefault(key, []).extend(race["QualifyingResults"])

        docs, skipped = [], 0
        for (row_season, round_number), rows in grouped.items():
            for row in rows:
                try:
                    docs.append(build_qualifying_doc(row_season, round_number, row))
                except KeyError as error:
                    skipped += 1
                    log.warning("Skipped a row in round %d: missing field %s", round_number, error)

        expected = int(pages[0]["payload"]["MRData"]["total"])
        inserted, changed = upsert(
            db["qualifying_results"], ["season", "round", "driver_id"], docs
        )
        log.info(
            "qualifying_results prepared=%-4d inserted=%-4d changed=%d",
            len(docs), inserted, changed,
        )

        stored = db["qualifying_results"].count_documents({"season": season})
        log.info("Check: stored for %d = %d (API total %d)", season, stored, expected)
        log.info("Check: rows with no Q1 time: %d", sum(d["q1_seconds"] is None for d in docs))
        log.info("Check: rows with no Q2 time: %d", sum(d["q2_seconds"] is None for d in docs))
        log.info("Check: rows with no Q3 time: %d", sum(d["q3_seconds"] is None for d in docs))
        log.info("Check: rows with no valid position: %d", sum(d["position"] is None for d in docs))

        known = set(db["drivers"].distinct("driver_id"))
        unknown = sorted({d["driver_id"] for d in docs} - known)
        log.info("Check: drivers not in the drivers collection: %s", unknown or "none")
        log.info("Skipped rows: %d", skipped)

    except PyMongoError as error:
        log.error("Database error: %s: %s", type(error).__name__, str(error)[:300])
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()