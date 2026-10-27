"""API tests that use a fake database, so they need no network."""

import pytest
from fastapi.testclient import TestClient
from pymongo.errors import ServerSelectionTimeoutError

from app.db import get_db
from app.main import app


class FakeAdmin:
    def __init__(self, fail):
        self.fail = fail

    def command(self, name):
        if self.fail:
            raise ServerSelectionTimeoutError("database down")
        return {"ok": 1}


class FakeClient:
    def __init__(self, fail):
        self.admin = FakeAdmin(fail)


class FakeCollection:
    def __init__(self, record):
        self.record = record

    def find_one(self, *args, **kwargs):
        return self.record


class FakeDb:
    def __init__(self, fail=False, record=None):
        self.client = FakeClient(fail)
        self.record = record

    def __getitem__(self, name):
        return FakeCollection(self.record)


def make_client(db):
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_health_ok():
    response = make_client(FakeDb()).get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "reachable"}


def test_health_returns_503_when_database_is_down():
    response = make_client(FakeDb(fail=True)).get("/api/health")
    assert response.status_code == 503


def test_model_status_returns_the_recorded_model():
    record = {"model_version": "podium_rf_v1", "features": ["grid_effective"]}
    response = make_client(FakeDb(record=record)).get("/api/models/status")
    assert response.status_code == 200
    assert response.json()["model_version"] == "podium_rf_v1"


def test_model_status_is_404_when_no_model_exists():
    response = make_client(FakeDb(record=None)).get("/api/models/status")
    assert response.status_code == 404


def test_cors_allows_the_local_frontend():
    origin = "http://localhost:5173"
    response = make_client(FakeDb()).get("/api/health", headers={"Origin": origin})
    assert response.headers.get("access-control-allow-origin") == origin


def test_cors_does_not_allow_other_sites():
    response = make_client(FakeDb()).get(
        "/api/health", headers={"Origin": "http://evil.example"}
    )
    assert "access-control-allow-origin" not in response.headers