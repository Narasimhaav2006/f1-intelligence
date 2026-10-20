"""
Create the MongoDB indexes for F1 Intelligence.

Safe to run more than once: MongoDB ignores an index that already exists.
Run it from the backend folder:  python scripts\\create_indexes.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient
from pymongo.errors import PyMongoError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB_NAME", "f1_intelligence")

if not uri or uri.startswith("replace-with"):
    sys.exit("MONGODB_URI is missing or still a placeholder. Check your .env file.")

# Each entry: collection name -> list of (fields, is_unique, index_name).
# A unique index on several fields means the COMBINATION must be unique.
INDEXES = {
    "raw_jolpica": [
        (["endpoint", "params_key"], True, "uniq_endpoint_params"),
    ],
    "drivers": [
        (["driver_id"], True, "uniq_driver_id"),
    ],
    "constructors": [
        (["constructor_id"], True, "uniq_constructor_id"),
    ],
    "races": [
        (["season", "round"], True, "uniq_season_round"),
    ],
    "race_results": [
        (["season", "round", "driver_id"], True, "uniq_season_round_driver"),
        (["driver_id"], False, "idx_driver"),
        (["constructor_id"], False, "idx_constructor"),
    ],
    "qualifying_results": [
        (["season", "round", "driver_id"], True, "uniq_season_round_driver"),
    ],
}

client = MongoClient(uri, serverSelectionTimeoutMS=5000)

try:
    client.admin.command("ping")
    db = client[db_name]
    print(f"Connected. Creating indexes in database '{db_name}'...\n")

    for collection_name, specs in INDEXES.items():
        for fields, is_unique, index_name in specs:
            keys = [(field, ASCENDING) for field in fields]
            db[collection_name].create_index(keys, unique=is_unique, name=index_name)
            kind = "unique" if is_unique else "regular"
            print(f"  {collection_name}: {kind} index '{index_name}' on {fields}")

    print("\nAll indexes created.")

except PyMongoError as error:
    print(f"\nDatabase error: {type(error).__name__}")
    print(str(error)[:400])
    sys.exit(1)
finally:
    client.close()