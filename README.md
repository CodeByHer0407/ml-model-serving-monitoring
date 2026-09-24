# ML Model Serving & Monitoring Platform

A production-style machine learning serving platform for **versioned model inference, runtime model activation and rollback, observability, automated testing, containerized deployment, and CI security checks**.

The platform serves LightGBM Remaining Useful Life (RUL) models trained on the **NASA C-MAPSS FD001 turbofan engine degradation dataset**. It demonstrates how multiple ML model versions can be registered, switched at runtime, monitored, rolled back, tested, containerized, and validated through CI.

---

## Key Features

- FastAPI-based ML inference service
- Two versioned LightGBM model artifacts: **V1** and **V2**
- Runtime model activation without restarting the API
- Rollback to the previously active model
- Model metadata and feature-schema validation
- Health and readiness endpoints
- Strict request validation using Pydantic
- Prometheus metrics for:
  - HTTP request counts
  - HTTP request duration
  - Prediction counts by model version
  - Model inference latency by model version
- Grafana dashboard for model-serving observability
- Automatic Grafana datasource and dashboard provisioning
- Dockerized FastAPI application
- Full FastAPI + Prometheus + Grafana stack using Docker Compose
- 9 automated tests covering serving, validation, lifecycle management, and metrics
- GitHub Actions CI
- Automated Docker image build validation
- SBOM generation
- Container vulnerability scanning

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

                         GitHub Push / PR
                                   │
                                   ▼
                         ┌────────────────────┐
                         │   GitHub Actions   │
                         └─────────┬──────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
           Run Tests         Docker Build       Security Scan
                                                   │
                                                   ├── SBOM
                                                   └── Vulnerabilities
```

---

## Model Lifecycle

The platform supports multiple versioned model artifacts and tracks both the active model and the previously active model.

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

A candidate model is loaded and validated before the registry state is updated. This prevents an invalid model artifact from being marked active.

Model switching happens at runtime without rebuilding or restarting the API.

---

## Model Versions

Both models use the same **18-feature inference contract**.

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

V2 is an independently trained candidate model used to demonstrate lifecycle management. It is not presented as inherently better than V1 without a separate model-quality comparison.

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

The serving layer reconstructs requests in the feature order stored inside the selected model artifact before inference.

Validation includes:

- exact feature-schema enforcement
- numeric input validation
- finite-value validation
- positive integer validation for `cycle`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Checks whether the API process is running |
| `GET` | `/ready` | Checks whether a valid ML model is loaded and ready |
| `POST` | `/predict` | Generates a Remaining Useful Life prediction |
| `GET` | `/models` | Lists registered model versions |
| `GET` | `/models/active` | Returns the active and previous model versions |
| `POST` | `/models/{version}/activate` | Activates a selected model version |
| `POST` | `/models/rollback` | Rolls back to the previously active model |
| `GET` | `/metrics` | Exposes Prometheus-compatible application metrics |

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

### Example Response

```json
{
  "predicted_rul": 228.58607282951172,
  "model_version": "v1",
  "unit": "cycles"
}
```

The response reports the model version that served the request, which makes it possible to trace predictions across model activations and rollbacks.

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

Prediction metrics include `model_version` labels, allowing V1 and V2 traffic and latency to be monitored independently.

### Prometheus

```text
http://localhost:9090
```

Prometheus scrapes the FastAPI service from the internal Docker Compose network.

### Grafana

```text
http://localhost:3000
```

The dashboard includes:

- **Predictions by Model Version**
- **Average Inference Latency by Model Version**
- **HTTP Request Rate**
- **HTTP Error Rate**

The Prometheus datasource and Grafana dashboard are provisioned automatically from files stored in the repository, so a fresh Docker Compose deployment recreates the monitoring setup without manual dashboard import.

### Grafana Dashboard

The final validation run demonstrates traffic for both V1 and V2, per-version inference latency, API request activity, and a deliberately generated `404` used to verify HTTP error monitoring.

![Grafana Monitoring Dashboard](docs/images/grafana-dashboard.png)

---

## Quick Start with Docker

### 1. Start the complete stack

```bash
docker compose up --build -d
```

### 2. Check running containers

```bash
docker compose ps
```

Expected services:

| Service | Port | Purpose |
|---|---:|---|
| FastAPI | `8000` | ML inference and model lifecycle API |
| Prometheus | `9090` | Metrics collection |
| Grafana | `3000` | Monitoring dashboards |

### 3. Open the services

```text
FastAPI Swagger UI: http://localhost:8000/docs
Prometheus:          http://localhost:9090
Grafana:             http://localhost:3000
```

### 4. Stop the stack

```bash
docker compose down
```

To also remove the Grafana data volume:

```bash
docker compose down -v
```

The Grafana datasource and dashboard will be recreated automatically the next time the stack starts.

---

## Running the API Locally

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

The automated tests cover:

- API health
- model readiness
- real LightGBM inference
- missing feature validation
- unexpected feature validation
- invalid `cycle` validation
- model activation
- model rollback
- Prometheus metrics exposure

Verified status:

```text
9 tests passed
```

---

## CI/CD & Security

GitHub Actions validates the project on pushes and pull requests to `main`.

The pipeline performs:

```text
Run Tests
   │
   ▼
Build Docker Image
   │
   ▼
Generate SBOM
   │
   ▼
Vulnerability Scan
```

The workflow includes:

- Python environment setup
- dependency installation
- automated test execution
- Docker image build validation
- SBOM generation
- container vulnerability scanning

The SBOM and vulnerability scan use Anchore tooling.

The current vulnerability scan reports high-severity findings without blocking the build. This keeps security findings visible while maintaining a usable CI baseline for the project.

---

## Reproducible Grafana Provisioning

Grafana is configured entirely through files stored in the repository.

The project provisions:

```text
monitoring/grafana/provisioning/datasources/prometheus.yml
monitoring/grafana/provisioning/dashboards/dashboards.yml
monitoring/grafana/dashboards/ml-serving-dashboard.json
```

This was validated by removing the existing Grafana Docker volume and recreating the complete stack. The dashboard and Prometheus datasource were restored automatically from repository configuration.

---

## Project Structure

```text
ml-model-serving-monitoring/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── metrics.py
│   ├── model_registry.py
│   ├── model_service.py
│   └── schemas.py
│
├── docs/
│   └── images/
│       └── grafana-dashboard.png
│
├── models/
│   ├── rul_model_v1.joblib
│   └── rul_model_v2.joblib
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboards/
│       │   └── ml-serving-dashboard.json
│       └── provisioning/
│           ├── dashboards/
│           │   └── dashboards.yml
│           └── datasources/
│               └── prometheus.yml
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
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
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

### CI/CD & Security

- GitHub Actions
- Anchore SBOM Action
- Anchore vulnerability scanning

---

## Dataset

The demonstration models use the **NASA C-MAPSS FD001 turbofan engine degradation dataset**.

The raw training dataset is intentionally not stored in this repository.

The serialized model artifacts contain the metadata and feature schema required by the inference service.

---

## Design Decisions

### Why keep V1 and V2 under the same feature contract?

V2 is trained using the feature schema stored in V1. This keeps model versions interchangeable at serving time and prevents accidental schema drift between versions.

### Why separate `/health` and `/ready`?

`/health` verifies that the API process is alive.

`/ready` verifies that a valid model artifact is loaded and the service is ready to perform inference.

### Why validate a candidate model before activation?

A newly selected model is loaded and validated before the registry state changes. This reduces the risk of marking a broken or incompatible model artifact as active.

### Why separate monitoring from the API?

FastAPI focuses on model serving and metrics exposure. Prometheus collects those metrics, while Grafana handles visualization. Docker Compose connects the services while keeping their responsibilities separate.

### Why provision Grafana from files?

A manually configured dashboard only exists in a local Grafana database or Docker volume. File-based provisioning makes the dashboard and datasource reproducible for anyone cloning the repository.

### Why keep the security scan non-blocking?

The current CI pipeline reports high-severity vulnerabilities without failing the entire workflow. This creates visibility into container risk while keeping the initial security baseline observable and actionable.

---

## Verified End-to-End Flow

The project has been validated through the following lifecycle:

```text
Start Docker Compose stack
        │
        ▼
V1 model loaded and ready
        │
        ▼
Serve V1 prediction
        │
        ▼
Activate V2 at runtime
        │
        ▼
Serve V2 prediction
        │
        ▼
Rollback
        │
        ▼
Serve V1 prediction again
        │
        ▼
Prometheus captures metrics
        │
        ▼
Grafana visualizes V1/V2 traffic,
latency, request rate, and HTTP errors
```

A clean test run completes successfully with all **9 tests passing**.

---

## Future Improvements

Potential extensions include:

- model-quality comparison gates before promotion
- data and feature drift detection
- automated model promotion policies
- authentication and authorization for model-management endpoints
- persistent external model registry
- cloud deployment
- stricter CI security gates after vulnerability remediation

These are intentionally left as future extensions rather than requirements for the current portfolio version.

---

## Project Status

```text
Real model inference             ✅
V1 / V2 model versions          ✅
Runtime model activation        ✅
Rollback                        ✅
Health / readiness checks       ✅
Input and schema validation     ✅
Prometheus metrics              ✅
Grafana monitoring              ✅
Grafana auto-provisioning       ✅
Dockerized API                  ✅
Docker Compose deployment       ✅
Automated tests                 ✅
GitHub Actions CI               ✅
Docker build validation         ✅
SBOM generation                 ✅
Vulnerability scanning          ✅
Clean end-to-end validation     ✅
```

The core project is complete. Remaining work is limited to portfolio presentation and resume integration.
