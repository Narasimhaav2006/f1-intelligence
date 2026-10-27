"""One shared MongoDB client for the whole app (connection pooling)."""

from fastapi import HTTPException
from pymongo import MongoClient

from . import config

_client = None


def get_client():
    global _client
    if _client is None:
        if not config.MONGODB_URI or config.MONGODB_URI.startswith("replace-with"):
            raise RuntimeError("MONGODB_URI is missing or still a placeholder.")
        _client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=5000)
    return _client


def get_db():
    """FastAPI dependency: gives an endpoint the database."""
    try:
        return get_client()[config.MONGODB_DB_NAME]
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Database is not configured.")


def close_client():
    global _client
    if _client is not None:
        _client.close()
        _client = None