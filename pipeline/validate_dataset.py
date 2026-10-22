"""
Read-only validation of the clean collections.
Makes no API requests and changes nothing.

Usage, from the project root with the virtual environment active:
    python pipeline/validate_dataset.py 2014 2025
"""

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def main():
    if len(sys.argv) != 3 or not (sys.argv[1].isdigit() and sys.argv[2].isdigit()):
        sys.exit("Usage: python pipeline/validate_dataset.py 2014 2025")
    first, last = int(sys.argv[1]), int(sys.argv[2])

    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB_NAME", "f1_intelligence")
    if not uri or uri.startswith("replace-with"):
        sys.exit("MONGODB_URI is missing or still a placeholder. Check your .env file.")

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    issues = []

    try:
        client.admin.command("ping")
        db = client[db_name]
        in_range = {"season": {"$gte": first, "$lte": last}}

        races = list(db["races"].find(in_range, {"_id": 0, "season": 1, "round": 1, "date": 1}))
        results = list(
            db["race_results"].find(
                in_range,
                {"_id": 0, "season": 1, "round": 1, "driver_id": 1, "constructor_id": 1,
                 "grid": 1, "position": 1, "points": 1, "started": 1},
            )
        )
        quals = list(
            db["qualifying_results"].find(
                in_range, {"_id": 0, "season": 1, "round": 1, "driver_id": 1, "position": 1}
            )
        )
        known_drivers = set(db["drivers"].distinct("driver_id"))
        known_teams = set(db["constructors"].distinct("constructor_id"))

        # Duplicate keys (the unique indexes should make this impossible)
        for name, rows in (("race_results", results), ("qualifying_results", quals)):
            counts = Counter((r["season"], r["round"], r["driver_id"]) for r in rows)
            for key, n in counts.items():
                if n > 1:
                    issues.append(f"{name}: duplicate key {key} appears {n} times")

        race_keys = {(r["season"], r["round"]) for r in races}
        results_by_race = defaultdict(list)
        for row in results:
            results_by_race[(row["season"], row["round"])].append(row)

        for key in sorted(set(results_by_race) - race_keys):
            issues.append(f"results exist for {key} but no race is stored")
        for key in sorted(race_keys - set(results_by_race)):
            issues.append(f"race {key} has no results")

        # Per-race checks
        for key, rows in sorted(results_by_race.items()):
            positions = [r["position"] for r in rows]
            if None in positions:
                issues.append(f"race {key}: a result has no numeric position")
            numeric = [p for p in positions if p is not None]
            repeated = [p for p, n in Counter(numeric).items() if n > 1]
            if repeated:
                issues.append(f"race {key}: repeated positions {repeated}")
            missing_podium = {1, 2, 3} - set(numeric)
            if missing_podium:
                issues.append(f"race {key}: positions {sorted(missing_podium)} are missing")
            for r in rows:
                if not r["started"] and r["position"] in (1, 2, 3):
                    issues.append(f"race {key}: {r['driver_id']} did not start but is in the top 3")
                if r["started"] and r["grid"] is None:
                    issues.append(f"race {key}: {r['driver_id']} has no grid value")
                if r["driver_id"] not in known_drivers:
                    issues.append(f"race {key}: unknown driver {r['driver_id']}")
                if r["constructor_id"] not in known_teams:
                    issues.append(f"race {key}: unknown constructor {r['constructor_id']}")

        for row in quals:
            if row["position"] is None:
                issues.append(
                    f"qualifying ({row['season']}, {row['round']}): "
                    f"{row['driver_id']} has no numeric position"
                )

        # Per-season table
        result_keys = {(r["season"], r["round"], r["driver_id"]) for r in results}
        qual_keys = {(r["season"], r["round"], r["driver_id"]) for r in quals}

        print("Season  races  rounds_ok  results  started  qualifying  no_qual  no_result  max_points")
        for season in range(first, last + 1):
            season_races = sorted((r for r in races if r["season"] == season), key=lambda r: r["round"])
            rounds = [r["round"] for r in season_races]
            rounds_ok = bool(rounds) and rounds == list(range(1, len(rounds) + 1))
            dates = [r["date"] for r in season_races]
            if not rounds:
                issues.append(f"season {season}: no races stored")
            elif not rounds_ok:
                issues.append(f"season {season}: rounds are not 1..N without gaps: {rounds}")
            if dates != sorted(dates):
                issues.append(f"season {season}: race dates are not in round order")

            season_results = [r for r in results if r["season"] == season]
            started = sum(1 for r in season_results if r["started"])
            season_result_keys = {k for k in result_keys if k[0] == season}
            season_qual_keys = {k for k in qual_keys if k[0] == season}
            max_points = max((r["points"] for r in season_results if r["points"] is not None), default=0)
            print(
                f"{season}   {len(rounds):>5}  {str(rounds_ok):>9}  {len(season_results):>7}  "
                f"{started:>7}  {len(season_qual_keys):>10}  "
                f"{len(season_result_keys - season_qual_keys):>7}  "
                f"{len(season_qual_keys - season_result_keys):>9}  {max_points:>10g}"
            )

        started_rows = [r for r in results if r["started"]]
        podium = sum(1 for r in started_rows if r["position"] in (1, 2, 3))
        if started_rows:
            print(
                f"\nStarters: {len(started_rows)}. Podium finishes: {podium} "
                f"({100 * podium / len(started_rows):.1f}% of starters)."
            )

        print(f"\nIssues found: {len(issues)}")
        for line in issues[:40]:
            print(f"  - {line}")
        if len(issues) > 40:
            print(f"  ... and {len(issues) - 40} more")

    except PyMongoError as error:
        print(f"Database error: {type(error).__name__}")
        print(str(error)[:400])
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()