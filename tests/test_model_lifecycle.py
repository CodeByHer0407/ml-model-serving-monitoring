import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app, registry


FIXTURE_PATH = Path(
    "tests/fixtures/engine_sample.json"
)


def load_sample():
    with FIXTURE_PATH.open() as file:
        return json.load(file)


def test_model_activation_and_rollback():
    payload = load_sample()

    original_version = registry.get_active_version()

    with TestClient(app) as client:

        # Start from V1 for a deterministic test
        if registry.get_active_version() != "v1":
            response = client.post(
                "/models/v1/activate"
            )
            assert response.status_code == 200

        # Verify V1 is active
        response = client.get(
            "/models/active"
        )

        assert response.status_code == 200
        assert (
            response.json()["active_version"]
            == "v1"
        )

        # Activate V2
        response = client.post(
            "/models/v2/activate"
        )

        assert response.status_code == 200

        result = response.json()

        assert result["active_version"] == "v2"
        assert result["previous_version"] == "v1"

        # Verify inference actually uses V2
        response = client.post(
            "/predict",
            json=payload,
        )

        assert response.status_code == 200

        prediction = response.json()

        assert prediction["model_version"] == "v2"
        assert prediction["predicted_rul"] >= 0

        # Roll back to V1
        response = client.post(
            "/models/rollback"
        )

        assert response.status_code == 200

        result = response.json()

        assert result["active_version"] == "v1"
        assert result["previous_version"] == "v2"

        # Verify inference now uses V1 again
        response = client.post(
            "/predict",
            json=payload,
        )

        assert response.status_code == 200

        prediction = response.json()

        assert prediction["model_version"] == "v1"

        # Restore original state if needed
        if original_version != "v1":
            response = client.post(
                f"/models/{original_version}/activate"
            )
            assert response.status_code == 200

def test_activate_unknown_model():
    with TestClient(app) as client:
        response = client.post(
            "/models/v999/activate"
        )

    assert response.status_code == 404