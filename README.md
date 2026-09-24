# ML Model Serving & Monitoring Platform

A production-style machine learning serving platform for versioned model inference, runtime model activation and rollback, observability, automated validation, and containerized deployment.

The platform serves LightGBM Remaining Useful Life (RUL) models trained on the NASA C-MAPSS FD001 turbofan engine degradation dataset. It demonstrates how multiple ML model versions can be registered, switched at runtime, monitored, and rolled back without restarting the API.

---

## Key Features

- FastAPI-based ML inference service
- Versioned LightGBM model artifacts
- Runtime model activation
- Rollback to the previously active model
- Model metadata and feature-schema validation
- Health and readiness endpoints
- Strict request validation using Pydantic
- Prometheus metrics for:
  - HTTP request counts
  - HTTP request duration
  - Prediction counts by model version
  - Model inference latency
- Grafana dashboard for operational monitoring
- Dockerized FastAPI application
- Full FastAPI + Prometheus + Grafana stack using Docker Compose
- Automated unit and integration tests
- Reproducible local deployment with a single Docker Compose command

---

## Architecture

```text
                         ┌────────────────────┐
                         │       Client       │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │      FastAPI       │
                         │    REST Service    │
                         └─────────┬──────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
          ┌──────────────────┐          ┌──────────────────┐
          │  Model Registry  │          │     /metrics     │
          │                  │          │    Prometheus    │
          │   V1        V2   │          └─────────┬────────┘
          └────────┬─────────┘                    │
                   │                              ▼
                   ▼                    ┌──────────────────┐
          ┌──────────────────┐          │     Grafana      │
          │     LightGBM     │          │    Dashboard     │
          │    Inference     │          └──────────────────┘
          └──────────────────┘
```

---

## Model Lifecycle

The platform supports multiple versioned model artifacts and keeps track of both the active model and the previously active model.

Example lifecycle:

```text
V1 active
   │
   ▼
Activate V2
   │
   ▼
Predictions served by V2
   │
   ▼
Rollback
   │
   ▼
V1 active again
```

Model switching happens at runtime without rebuilding or restarting the API.

---

## Model Versions

Both models use the same 18-feature inference contract.

### Model V1

LightGBM configuration:

```text
n_estimators = 300
learning_rate = 0.05
num_leaves = 15
random_state = 42
```

### Model V2

LightGBM configuration:

```text
n_estimators = 500
learning_rate = 0.03
num_leaves = 31
random_state = 42
```

V2 is an independently trained candidate version used to demonstrate model lifecycle management. It is not presented as inherently better than V1 without a separate model-quality comparison.

---

## Input Features

The API expects the following 18 features:

```text
cycle
setting_1
setting_2
sensor_2
sensor_3
sensor_4
sensor_6
sensor_7
sensor_8
sensor_9
sensor_11
sensor_12
sensor_13
sensor_14
sensor_15
sensor_17
sensor_20
sensor_21
```

The serving layer reconstructs the request in the feature order stored inside the selected model artifact before inference.

Additional validation includes:

- exact feature-schema enforcement
- numeric input validation
- finite-value validation
- positive integer validation for `cycle`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Checks whether the API process is running |
| `GET` | `/ready` | Checks whether the ML model is loaded and ready |
| `POST` | `/predict` | Generates a Remaining Useful Life prediction |
| `GET` | `/models` | Lists registered model versions |
| `GET` | `/models/active` | Returns the active and previous model versions |
| `POST` | `/models/{version}/activate` | Activates a selected model version |
| `POST` | `/models/rollback` | Rolls back to the previously active model |
| `GET` | `/metrics` | Exposes Prometheus-compatible metrics |

Interactive Swagger documentation:

```text
http://localhost:8000/docs
```

Readiness endpoint:

```text
http://localhost:8000/ready
```

---

## Example Prediction

### Request

```json
{
  "cycle": 31,
  "setting_1": -0.0006,
  "setting_2": 0.0004,
  "sensor_2": 642.58,
  "sensor_3": 1581.22,
  "sensor_4": 1398.91,
  "sensor_6": 21.61,
  "sensor_7": 554.42,
  "sensor_8": 2388.08,
  "sensor_9": 9056.4,
  "sensor_11": 47.23,
  "sensor_12": 521.79,
  "sensor_13": 2388.06,
  "sensor_14": 8130.11,
  "sensor_15": 8.4024,
  "sensor_17": 393.0,
  "sensor_20": 38.81,
  "sensor_21": 23.3552
}
```

### Example response

```json
{
  "predicted_rul": 228.58607282951172,
  "model_version": "v1",
  "unit": "cycles"
}
```

---

## Monitoring & Observability

The FastAPI service exposes Prometheus-compatible metrics through:

```text
/metrics
```

Custom metrics include:

```text
http_requests_total
http_request_duration_seconds
model_predictions_total
model_prediction_duration_seconds
```

Prediction metrics include the active model version as a label, allowing V1 and V2 traffic and latency to be monitored independently.

### Prometheus

```text
http://localhost:9090
```

### Grafana

```text
http://localhost:3000
```

The Grafana dashboard currently includes:

- HTTP request rate
- Prediction traffic by model version
- Average inference latency by model version
- HTTP error rate

---

## Running with Docker

The full stack can be started with:

```bash
docker compose up --build -d
```

Check container status:

```bash
docker compose ps
```

Expected services:

| Service | Port |
|---|---:|
| FastAPI | `8000` |
| Prometheus | `9090` |
| Grafana | `3000` |

Stop the stack with:

```bash
docker compose down
```

---

## Running Locally

Create or activate a Python 3.11 environment.

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Start the API:

```bash
python -m uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Testing

Run the full test suite:

```bash
pytest -v
```

Current automated tests cover:

- API health
- model readiness
- real model inference
- missing feature validation
- unexpected feature validation
- invalid cycle validation
- model activation
- model rollback
- Prometheus metrics endpoint

Current status:

```text
9 tests passed
```

---

## Project Structure

```text
ml-model-serving-monitoring/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── metrics.py
│   ├── model_registry.py
│   ├── model_service.py
│   └── schemas.py
│
├── models/
│   ├── rul_model_v1.joblib
│   └── rul_model_v2.joblib
│
├── monitoring/
│   └── prometheus.yml
│
├── scripts/
│   └── train_v2.py
│
├── tests/
│   ├── fixtures/
│   │   └── engine_sample.json
│   ├── __init__.py
│   ├── test_health.py
│   ├── test_metrics.py
│   ├── test_model_lifecycle.py
│   ├── test_prediction.py
│   └── test_readiness.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Technology Stack

### Machine Learning

- LightGBM
- Scikit-learn
- Pandas
- NumPy
- Joblib

### Backend

- Python
- FastAPI
- Pydantic
- Uvicorn

### Monitoring

- Prometheus
- Grafana

### Deployment

- Docker
- Docker Compose

### Testing

- Pytest
- FastAPI TestClient

---

## Dataset

The demonstration models use the NASA C-MAPSS FD001 turbofan engine degradation dataset.

The raw training dataset is intentionally not stored in this repository.

The serialized model artifacts contain the metadata and feature schema required for inference.

---

## Design Decisions

### Why keep V1 and V2 under the same feature contract?

The platform reuses the feature schema stored in V1 when training V2. This ensures that model versions remain interchangeable at serving time and prevents accidental schema drift between versions.

### Why separate `/health` and `/ready`?

`/health` verifies that the API process is alive.

`/ready` verifies that a valid model artifact has been loaded and the service is ready to perform inference.

### Why load the candidate model before changing registry state?

A newly selected model is validated and loaded first. The registry state is updated only after the candidate model is confirmed usable, reducing the risk of marking a broken model version as active.

### Why keep the Grafana dashboard separate from the API?

Prometheus and Grafana are operational components, while FastAPI remains focused on serving inference and exposing metrics. Docker Compose connects the three services into a reproducible local stack.

---

## Future Improvements

Planned improvements include:

- GitHub Actions CI
- automated Docker image build validation
- SBOM generation
- container vulnerability scanning
- model-quality comparison before promotion
- data and feature drift detection
- automated promotion policies
- authentication and authorization for model-management endpoints
- persistent external model registry
- automated Grafana dashboard provisioning

---

## Status

The current implementation supports:

```text
Real model inference          ✅
V1 / V2 model versions       ✅
Runtime activation           ✅
Rollback                     ✅
Health / readiness checks    ✅
Prometheus metrics           ✅
Grafana monitoring           ✅
Docker Compose deployment    ✅
Automated tests              ✅
```

CI/CD and security scanning are the next planned implementation steps.
