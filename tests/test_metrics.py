from fastapi.testclient import TestClient

from app.main import app


def test_metrics_endpoint():
    with TestClient(app) as client:
        client.get("/health")

        response = client.get("/metrics")

    assert response.status_code == 200

    metrics = response.text

    assert "http_requests_total" in metrics
    assert "http_request_duration_seconds" in metrics
    assert "model_predictions_total" in metrics
    assert "model_prediction_duration_seconds" in metrics