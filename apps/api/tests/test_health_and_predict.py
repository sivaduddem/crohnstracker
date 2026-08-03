import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from crohnstracker_shared.schemas import HealthResponse, HybridPredictResponse  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # Ensure demo fixtures exist
    from ml.data.generate_demo_fixtures import main as gen

    gen()
    from apps.api.app.main import app

    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = HealthResponse.model_validate(r.json())
    assert body.status == "ok"
    assert body.mode.value == "demo"
    assert body.demo_banner is not None


def test_predict_hybrid_schema(client):
    r = client.post("/predict", json={"case_id": "demo_ulcer_01", "top_k": 3})
    assert r.status_code == 200
    body = HybridPredictResponse.model_validate(r.json())
    assert body.is_mock is True
    assert body.mode.value == "demo"
    assert body.prediction.label in {"normal", "erythema", "erosion", "ulcer"}
    assert len(body.similar_cases) <= 3
    assert "Research" in body.disclaimer


def test_similar_threshold(client):
    r = client.get("/similar", params={"case_id": "demo_normal_01", "top_k": 5, "similarity_threshold": 0.99})
    assert r.status_code == 200
    data = r.json()
    assert data["is_mock"] is True
    assert all(c["similarity"] >= 0.99 for c in data["similar_cases"])


def test_metrics_summary(client):
    r = client.get("/metrics/summary")
    assert r.status_code == 200
    assert "classification" in r.json()
    assert "retrieval" in r.json()
