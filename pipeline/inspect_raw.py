"""
Read-only summary of stored raw race results for one season.
Makes no API requests and changes nothing.

Usage, from the project root:
    python pipeline/inspect_raw.py 2023
"""

import os
import sys
from collections import Counter
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

try:
    raw = client[db_name]["raw_jolpica"]
    pages = list(raw.find({"endpoint": f"{season}/results"}))
    if not pages:
        sys.exit(f"No stored pages for {season}/results. Run the ingest step first.")

    total_rows = 0
    pages_per_round = Counter()
    rows_per_round = Counter()
    status_counts = Counter()
    non_numeric_position = Counter()
    grid_zero = 0

    for page in pages:
        races = page["payload"]["MRData"]["RaceTable"]["Races"]
        for race in races:
            round_number = int(race["round"])
            pages_per_round[round_number] += 1
            for result in race["Results"]:
                total_rows += 1
                rows_per_round[round_number] += 1
                status_counts[result["status"]] += 1
                if not result["positionText"].isdigit():
                    non_numeric_position[result["positionText"]] += 1
                if result["grid"] == "0":
                    grid_zero += 1

    reported_total = pages[0]["payload"]["MRData"]["total"]
    split_rounds = sorted(r for r, n in pages_per_round.items() if n > 1)

    print(f"Season {season}")
    print(f"  Pages stored:               {len(pages)}")
    print(f"  Result rows counted:        {total_rows}")
    print(f"  Total reported by the API:  {reported_total}")
    print(f"  Distinct rounds:            {len(pages_per_round)}")
    print(f"  Rounds split across pages:  {split_rounds if split_rounds else 'none'}")
    print(f"  Fewest / most rows in a round: "
          f"{min(rows_per_round.values())} / {max(rows_per_round.values())}")
    print(f"  Rows with grid = 0:         {grid_zero}")
    print("\n  Non-numeric positionText values:")
    for value, count in non_numeric_position.most_common():
        print(f"    {value!r}: {count}")
    print("\n  Status values:")
    for value, count in status_counts.most_common():
        print(f"    {value}: {count}")

except PyMongoError as error:
    print(f"Database error: {type(error).__name__}")
    print(str(error)[:400])
    sys.exit(1)
finally:
    client.close()