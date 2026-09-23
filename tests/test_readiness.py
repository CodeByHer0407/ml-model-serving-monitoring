from fastapi.testclient import TestClient

from app.main import app


def test_readiness():
    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "ready"
    assert result["model_version"] == "v1"
    assert result["dataset"] == "NASA C-MAPSS FD001"
    assert result["feature_count"] == 18