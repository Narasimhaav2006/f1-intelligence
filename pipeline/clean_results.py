"""
Build the clean collections from raw pages already stored in MongoDB.
Makes no API requests. Safe to run more than once (it uses upserts).

Usage, from the project root with the virtual environment active:
    python pipeline/clean_results.py 2023

Requires the raw pages from:
    python pipeline/ingest_raw.py 2023/races
    python pipeline/ingest_raw.py 2023/results
"""

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("clean")


def to_int(value):
    """Convert text like '57' to 57. Returns None if it is not a whole number."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_float(value):
    """Convert text like '25' to 25.0. Returns None if it is not a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_race_doc(race):
    circuit = race.get("Circuit") or {}
    location = circuit.get("Location") or {}
    return {
        "season": int(race["season"]),
        "round": int(race["round"]),
        "race_name": race["raceName"],
        "date": race["date"],
        "time": race.get("time"),
        "circuit_id": circuit.get("circuitId"),
        "circuit_name": circuit.get("circuitName"),
        "locality": location.get("locality"),
        "country": location.get("country"),
        "latitude": to_float(location.get("lat")),
        "longitude": to_float(location.get("long")),
        "url": race.get("url"),
    }


def build_driver_doc(driver):
    return {
        "driver_id": driver["driverId"],
        "code": driver.get("code"),
        "permanent_number": to_int(driver.get("permanentNumber")),
        "given_name": driver["givenName"],
        "family_name": driver["familyName"],
        "date_of_birth": driver.get("dateOfBirth"),
        "nationality": driver.get("nationality"),
        "url": driver.get("url"),
    }


def build_constructor_doc(team):
    return {
        "constructor_id": team["constructorId"],
        "name": team["name"],
        "nationality": team.get("nationality"),
        "url": team.get("url"),
    }


def build_result_doc(season, round_number, row):
    driver = row["Driver"]
    team = row["Constructor"]
    position_text = row["positionText"]
    fastest = row.get("FastestLap") or {}
    return {
        "season": season,
        "round": round_number,
        "driver_id": driver["driverId"],
        "constructor_id": team["constructorId"],
        "driver_name": f"{driver['givenName']} {driver['familyName']}",
        "constructor_name": team["name"],
        "grid": to_int(row["grid"]),
        "position": to_int(row["position"]),
        "position_text": position_text,
        "points": to_float(row["points"]),
        "laps": to_int(row["laps"]),
        "status": row["status"],
        # 'W' means Did not start or Withdrew (see docs/label_definition.md)
        "started": position_text != "W",
        "finish_time_millis": to_int((row.get("Time") or {}).get("millis")),
        "fastest_lap_rank": to_int(fastest.get("rank")),
    }


def collect_results(pages):
    """Group result rows by (season, round), even if a race spans two pages."""
    grouped = {}
    for page in pages:
        for race in page["payload"]["MRData"]["RaceTable"]["Races"]:
            key = (int(race["season"]), int(race["round"]))
            grouped.setdefault(key, []).extend(race["Results"])
    return grouped


def upsert(collection, key_fields, docs):
    """Insert new documents and update changed ones. Returns (inserted, changed)."""
    if not docs:
        return 0, 0
    now = datetime.now(timezone.utc)
    operations = []
    for doc in docs:
        key = {field: doc[field] for field in key_fields}
        operations.append(
            UpdateOne(
                key,
                {"$set": doc, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )
        )
    result = collection.bulk_write(operations, ordered=False)
    return result.upserted_count, result.modified_count


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
        raw = db["raw_jolpica"]

        race_pages = list(raw.find({"endpoint": f"{season}/races"}))
        result_pages = list(raw.find({"endpoint": f"{season}/results"}))
        if not race_pages or not result_pages:
            sys.exit(
                f"Raw pages for {season}/races and {season}/results are needed. "
                "Run pipeline/ingest_raw.py first."
            )

        races = [
            build_race_doc(race)
            for page in race_pages
            for race in page["payload"]["MRData"]["RaceTable"]["Races"]
        ]

        drivers, constructors, results = {}, {}, []
        skipped = 0
        for (row_season, round_number), rows in collect_results(result_pages).items():
            for row in rows:
                try:
                    result_doc = build_result_doc(row_season, round_number, row)
                    driver_doc = build_driver_doc(row["Driver"])
                    team_doc = build_constructor_doc(row["Constructor"])
                except KeyError as error:
                    skipped += 1
                    log.warning("Skipped a row in round %d: missing field %s", round_number, error)
                    continue
                results.append(result_doc)
                drivers[driver_doc["driver_id"]] = driver_doc
                constructors[team_doc["constructor_id"]] = team_doc

        expected = int(result_pages[0]["payload"]["MRData"]["total"])
        if len(results) + skipped != expected:
            log.warning(
                "Row count mismatch: processed %d, API reported %d",
                len(results) + skipped, expected,
            )

        plan = [
            ("races", ["season", "round"], races),
            ("drivers", ["driver_id"], list(drivers.values())),
            ("constructors", ["constructor_id"], list(constructors.values())),
            ("race_results", ["season", "round", "driver_id"], results),
        ]
        for name, key_fields, docs in plan:
            inserted, changed = upsert(db[name], key_fields, docs)
            log.info("%-13s prepared=%-4d inserted=%-4d changed=%d", name, len(docs), inserted, changed)

        not_started = sum(1 for doc in results if not doc["started"])
        stored = db["race_results"].count_documents({"season": season})
        log.info("Check: race_results stored for %d = %d (API total %d)", season, stored, expected)
        log.info("Check: rows with started = false: %d", not_started)
        log.info("Skipped rows: %d", skipped)

    except PyMongoError as error:
        log.error("Database error: %s: %s", type(error).__name__, str(error)[:300])
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()