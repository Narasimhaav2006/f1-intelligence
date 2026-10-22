"""
First look at the data in pandas.
Read-only: makes no API requests and changes nothing in MongoDB.

Run from the project root with the virtual environment active:
    python ml/01_load_and_inspect.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

FIRST_SEASON, LAST_SEASON = 2014, 2025

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def load(db, collection_name, fields):
    """Read one collection into a DataFrame, keeping only the listed fields."""
    projection = {field: 1 for field in fields}
    projection["_id"] = 0  # we do not need MongoDB's internal id here
    query = {"season": {"$gte": FIRST_SEASON, "$lte": LAST_SEASON}}
    rows = list(db[collection_name].find(query, projection))
    return pd.DataFrame(rows, columns=fields)


def summarize(name, df):
    """Print the size, column types, missing values and first rows of a table."""
    print(f"\n=== {name}: {df.shape[0]} rows, {df.shape[1]} columns ===")
    overview = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum()})
    print(overview.to_string())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())


uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB_NAME", "f1_intelligence")
if not uri or uri.startswith("replace-with"):
    sys.exit("MONGODB_URI is missing or still a placeholder. Check your .env file.")

client = MongoClient(uri, serverSelectionTimeoutMS=5000)
try:
    client.admin.command("ping")
    db = client[db_name]
    races = load(db, "races", ["season", "round", "race_name", "date", "circuit_id", "country"])
    results = load(
        db,
        "race_results",
        ["season", "round", "driver_id", "constructor_id", "grid",
         "position", "points", "laps", "status", "started"],
    )
    quals = load(
        db,
        "qualifying_results",
        ["season", "round", "driver_id", "position",
         "q1_seconds", "q2_seconds", "q3_seconds", "best_seconds"],
    )
except PyMongoError as error:
    print(f"Database error: {type(error).__name__}")
    print(str(error)[:400])
    sys.exit(1)
finally:
    client.close()

summarize("races", races)
summarize("race_results", results)
summarize("qualifying_results", quals)

# A first real question: how often does each starting position finish on the podium?
starters = results[results["started"]].copy()
starters["podium"] = starters["position"].isin([1, 2, 3]).astype(int)
by_grid = starters.groupby("grid")["podium"].agg(starters="count", podium_rate="mean")
print("\nPodium rate by starting grid (grid 0 = see docs/label_definition.md):")
print(by_grid.head(12).round(3).to_string())