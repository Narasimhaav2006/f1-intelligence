"""
Smoke test: connect to MongoDB Atlas and try Create, Read, Update, Delete.

A "smoke test" is a quick check that the basics work before we build on them.
Run it from the backend folder:  python scripts\\db_smoke_test.py
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# This file is at backend/scripts/db_smoke_test.py, so the project root
# (where .env lives) is two folders above this file's folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB_NAME", "f1_intelligence")

if not uri or uri.startswith("replace-with"):
    sys.exit("MONGODB_URI is missing or still a placeholder. Check your .env file.")

# serverSelectionTimeoutMS: give up after 5 seconds instead of hanging.
client = MongoClient(uri, serverSelectionTimeoutMS=5000)

try:
    # "ping" is a tiny request that only proves the server answered.
    client.admin.command("ping")
    print("1. Connected to Atlas: ping succeeded")

    collection = client[db_name]["_connection_test"]

    # CREATE: insert one document. This is a made-up test record, not F1 data.
    result = collection.insert_one(
        {"label": "smoke test", "value": 1, "created_at": datetime.now(timezone.utc)}
    )
    print(f"2. Create: inserted a document with _id {result.inserted_id}")

    # READ: find the document we just inserted.
    found = collection.find_one({"_id": result.inserted_id})
    print(f"3. Read: found label={found['label']!r}, value={found['value']}")

    # UPDATE: change the value field from 1 to 2.
    update = collection.update_one({"_id": result.inserted_id}, {"$set": {"value": 2}})
    updated = collection.find_one({"_id": result.inserted_id})
    print(f"4. Update: matched {update.matched_count}, new value={updated['value']}")

    # DELETE: remove the document.
    delete = collection.delete_one({"_id": result.inserted_id})
    remaining = collection.count_documents({})
    print(f"5. Delete: removed {delete.deleted_count}, documents left: {remaining}")

    # Remove the empty test collection so nothing is left behind.
    collection.drop()
    print("6. Cleanup: test collection dropped")
    print("\nAll checks passed.")

except PyMongoError as error:
    print(f"\nDatabase error: {type(error).__name__}")
    print(str(error)[:400])
    sys.exit(1)
finally:
    client.close()