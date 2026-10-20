"""
Fetch one Jolpica-F1 endpoint (all pages) and store each raw page in MongoDB.

Usage, from the project root with the virtual environment active:
    python pipeline/ingest_raw.py 2023/1/results
    python pipeline/ingest_raw.py 2023/1/results --refresh

Pages already stored in the raw_jolpica collection are NOT downloaded again.
Use --refresh only when you want to replace stored pages (for example, for a
season that is still in progress).
"""

import argparse
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from jolpica_client import PAGE_LIMIT, JolpicaClient, JolpicaError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")


def get_or_fetch(raw, client, endpoint, offset, refresh):
    """Return one page of data, using the stored copy when we have it."""
    params_key = f"limit={PAGE_LIMIT}&offset={offset}"
    key = {"endpoint": endpoint, "params_key": params_key}

    if not refresh:
        stored = raw.find_one(key)
        if stored:
            log.info("cache hit, no request made: %s offset=%d", endpoint, offset)
            return stored["payload"]

    payload = client.get_json(
        f"{endpoint}.json", params={"limit": PAGE_LIMIT, "offset": offset}
    )
    raw.update_one(
        key,
        {
            "$set": {
                "source": "jolpica-f1",
                "url": payload["MRData"].get("url"),
                "fetched_at": datetime.now(timezone.utc),
                "payload": payload,
            }
        },
        upsert=True,
    )
    log.info(
        "fetched and stored: %s offset=%d (total=%s)",
        endpoint, offset, payload["MRData"]["total"],
    )
    return payload


def ingest_endpoint(raw, client, endpoint, refresh):
    """Walk through every page of an endpoint. Returns the number of pages."""
    offset = 0
    pages = 0
    while True:
        payload = get_or_fetch(raw, client, endpoint, offset, refresh)
        pages += 1
        total = int(payload["MRData"]["total"])
        offset += PAGE_LIMIT
        if offset >= total:
            return pages


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("endpoint", help="for example: 2023/1/results")
    parser.add_argument("--refresh", action="store_true", help="re-download stored pages")
    args = parser.parse_args()

    endpoint = args.endpoint.strip("/")
    if not re.fullmatch(r"[A-Za-z0-9_/-]+", endpoint):
        sys.exit("Endpoint may only contain letters, numbers, _ - and /")

    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB_NAME", "f1_intelligence")
    if not uri or uri.startswith("replace-with"):
        sys.exit("MONGODB_URI is missing or still a placeholder. Check your .env file.")

    mongo = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        mongo.admin.command("ping")
        raw = mongo[db_name]["raw_jolpica"]
        with JolpicaClient() as client:
            pages = ingest_endpoint(raw, client, endpoint, args.refresh)
        log.info("Done: %d page(s) for %s", pages, endpoint)
    except JolpicaError as error:
        log.error("API problem: %s", error)
        sys.exit(1)
    except PyMongoError as error:
        log.error("Database error: %s: %s", type(error).__name__, str(error)[:300])
        sys.exit(1)
    finally:
        mongo.close()


if __name__ == "__main__":
    main()