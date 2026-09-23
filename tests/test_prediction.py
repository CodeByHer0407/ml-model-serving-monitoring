import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app, model_service


FIXTURE_PATH = Path("tests/fixtures/engine_sample.json")


def load_sample():
    with FIXTURE_PATH.open() as file:
        return json.load(file)


def test_real_model_prediction():
    payload = load_sample()

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

        assert response.status_code == 200

        result = response.json()

        assert result["model_version"] == "v1"
        assert result["unit"] == "cycles"
        assert result["predicted_rul"] >= 0

        X = pd.DataFrame(
            [[payload[name] for name in model_service.feature_names]],
            columns=model_service.feature_names,
        )

        expected = max(
            float(model_service.model.predict(X)[0]),
            0.0,
        )

        assert result["predicted_rul"] == pytest.approx(expected)


def test_missing_feature():
    payload = load_sample()
    payload.pop("sensor_21")

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

    assert response.status_code == 422


def test_extra_feature():
    payload = load_sample()
    payload["unknown_sensor"] = 123.45

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

    assert response.status_code == 422


def test_invalid_cycle():
    payload = load_sample()
    payload["cycle"] = 0

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

    assert response.status_code == 422