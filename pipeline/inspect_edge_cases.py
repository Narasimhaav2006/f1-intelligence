"""
Read-only: list unusual result rows in stored raw data for one season.
Makes no API requests and changes nothing.

Shows:
  A) rows with a numeric positionText but a status other than Finished or Lapped
  B) rows with grid = 0

Usage, from the project root:
    python pipeline/inspect_edge_cases.py 2023
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

season = sys.argv[1] if len(sys.argv) > 1 else "2023"
if not season.isdigit():
    sys.exit("Season must be a number such as 2023")

uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB_NAME", "f1_intelligence")
if not uri or uri.startswith("replace-with"):
    sys.exit("MONGODB_URI is missing or still a placeholder. Check your .env file.")

client = MongoClient(uri, serverSelectionTimeoutMS=5000)


def describe(round_number, result, max_laps):
    laps = int(result.get("laps") or 0)
    share = 100 * laps / max_laps if max_laps else 0
    return (
        f"  round {round_number:>2}  {result['Driver']['driverId']:<18} "
        f"grid={result['grid']:>2}  position={result['position']:>2}  "
        f"positionText={result['positionText']:<2}  "
        f"laps={laps}/{max_laps} ({share:.0f}%)  status={result['status']}"
    )


try:
    raw = client[db_name]["raw_jolpica"]

    # Group by round first, so a race split across pages would still be handled.
    by_round = defaultdict(list)
    for page in raw.find({"endpoint": f"{season}/results"}):
        for race in page["payload"]["MRData"]["RaceTable"]["Races"]:
            by_round[int(race["round"])].extend(race["Results"])

    if not by_round:
        sys.exit(f"No stored pages for {season}/results. Run the ingest step first.")

    group_a = []
    group_b = []
    for round_number in sorted(by_round):
        results = by_round[round_number]
        max_laps = max(int(r.get("laps") or 0) for r in results)
        for result in results:
            if result["positionText"].isdigit() and result["status"] not in (
                "Finished",
                "Lapped",
            ):
                group_a.append(describe(round_number, result, max_laps))
            if result["grid"] == "0":
                group_b.append(describe(round_number, result, max_laps))

    print(f"A) Numeric positionText but status is not Finished or Lapped: {len(group_a)}")
    print("\n".join(group_a) if group_a else "  none")
    print(f"\nB) Rows with grid = 0: {len(group_b)}")
    print("\n".join(group_b) if group_b else "  none")

except PyMongoError as error:
    print(f"Database error: {type(error).__name__}")
    print(str(error)[:400])
    sys.exit(1)
finally:
    client.close()