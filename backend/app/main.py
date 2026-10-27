"""F1 Intelligence API."""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo.errors import PyMongoError

from . import config
from .db import close_client, get_db

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app):
    yield  # the app runs here
    close_client()  # on shutdown, close the database connection


app = FastAPI(title="F1 Intelligence API", version="0.1.0", lifespan=lifespan)

# CORS: only the websites listed in ALLOWED_ORIGINS may call this API from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health(db=Depends(get_db)):
    try:
        db.client.admin.command("ping")
    except PyMongoError as error:
        log.error("Database ping failed: %s", type(error).__name__)
        raise HTTPException(status_code=503, detail="Database is not reachable.")
    return {"status": "ok", "database": "reachable"}


@app.get("/api/models/status")
def model_status(db=Depends(get_db)):
    try:
        record = db["model_registry"].find_one({}, {"_id": 0}, sort=[("created_at", -1)])
    except PyMongoError as error:
        log.error("Model registry read failed: %s", type(error).__name__)
        raise HTTPException(status_code=503, detail="Database is not reachable.")
    if record is None:
        raise HTTPException(status_code=404, detail="No trained model is recorded yet.")
    return record