"""
Build the model features from the clean collections and store them in the
training_features collection (upsert, so it is safe to rerun).
Makes no API requests.

Run from the project root with the virtual environment active:
    python ml/03_build_training_features.py
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError

from features import ALL_FEATURE_COLUMNS, LABEL_COLUMN, build_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

FIRST_SEASON, LAST_SEASON = 2014, 2025
DATASET_VERSION = "v1"

pd.set_option("display.width", 200)


def load(db, name, fields):
    projection = {field: 1 for field in fields}
    projection["_id"] = 0
    query = {"season": {"$gte": FIRST_SEASON, "$lte": LAST_SEASON}}
    return pd.DataFrame(list(db[name].find(query, projection)), columns=fields)


def to_records(df):
    """DataFrame -> list of dicts with plain Python values (empty cells -> None)."""
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out = out.astype(object).where(out.notna(), None)
    return out.to_dict("records")


uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB_NAME", "f1_intelligence")
if not uri or uri.startswith("replace-with"):
    sys.exit("MONGODB_URI is missing or still a placeholder. Check your .env file.")

client = MongoClient(uri, serverSelectionTimeoutMS=5000)
try:
    client.admin.command("ping")
    db = client[db_name]

    results = load(
        db, "race_results",
        ["season", "round", "driver_id", "constructor_id", "grid", "position", "started"],
    )
    quals = load(
        db, "qualifying_results",
        ["season", "round", "driver_id", "position", "q1_seconds", "q2_seconds", "q3_seconds"],
    )
    races = load(db, "races", ["season", "round", "date", "circuit_id"])

    features = build_features(results, quals, races)

    # Checks before anything is written
    expected = int(results["started"].sum())
    if len(features) != expected:
        sys.exit(f"Row count problem: built {len(features)}, expected {expected} starters.")
    if features.duplicated(["season", "round", "driver_id"]).any():
        sys.exit("Duplicate (season, round, driver_id) rows in the feature table.")

    records = to_records(features)
    now = datetime.now(timezone.utc)
    collection = db["training_features"]
    collection.create_index(
        [("season", 1), ("round", 1), ("driver_id", 1)],
        unique=True, name="uniq_season_round_driver",
    )
    operations = []
    for record in records:
        key = {k: record[k] for k in ("season", "round", "driver_id")}
        document = dict(record, dataset_version=DATASET_VERSION, built_at=now)
        operations.append(UpdateOne(key, {"$set": document}, upsert=True))
    written = collection.bulk_write(operations, ordered=False)
    # Anything not rewritten just now is left over from an older build.
    removed = collection.delete_many({"built_at": {"$lt": now}}).deleted_count

    print(f"Dataset {DATASET_VERSION}: {len(features)} rows "
          f"(inserted {written.upserted_count}, updated {written.modified_count}, "
          f"removed stale {removed})")
    print(f"Date range: {features['date'].min().date()} to {features['date'].max().date()}")
    print(f"Podium rate: {100 * features[LABEL_COLUMN].mean():.1f}%")

    print("\nMissing values per feature:")
    print(features[ALL_FEATURE_COLUMNS].isna().sum().to_string())

    first_rows = features["driver_podium_rate_5"].isna().sum()
    drivers = features["driver_id"].nunique()
    print(f"\nRows with no driver history: {first_rows}. Distinct drivers: {drivers}.")
    print("(These two numbers should be equal: only each driver's first start has no history.)")

    print("\nRows per season:")
    print(features.groupby("season").size().to_string())

except PyMongoError as error:
    print(f"Database error: {type(error).__name__}")
    print(str(error)[:400])
    sys.exit(1)
finally:
    client.close()