from contextlib import asynccontextmanager
from pathlib import Path
from app.model_registry import ModelRegistry

from fastapi import FastAPI, HTTPException, status

from app.model_service import ModelService
from app.schemas import PredictionRequest, PredictionResponse


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


registry = ModelRegistry(
    models_dir=PROJECT_ROOT / "models",
    state_path=(
        PROJECT_ROOT
        / "registry"
        / "model_state.json"
    ),
)


model_service = ModelService(
    model_path=registry.get_active_model_path()
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load_model()

    yield

    model_service.model = None


app = FastAPI(
    title="ML Model Serving & Monitoring Platform",
    description="A platform for serving and monitoring ML models.",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ml-model-serving-monitoring"
    }


@app.get("/ready")
def readiness_check():
    if model_service.is_ready:
        return {
            "status": "ready",
            "model_version": model_service.model_version,
            "dataset": model_service.dataset,
            "feature_count": len(
                model_service.feature_names
            ),
        }

    return {
        "status": "not_ready",
        "error": model_service.load_error,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):
    if not model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is not available."
        )

    try:
        sensor_data = request.model_dump()

        prediction = model_service.predict(
            sensor_data
        )

        return PredictionResponse(
            predicted_rul=prediction,
            model_version=model_service.model_version,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(exc)}",
        )

@app.get("/models")
def list_models():
    return {
        "models": registry.list_models()
    }

@app.get("/models/active")
def get_active_model():
    return {
        "active_version": (
            registry.get_active_version()
        ),
        "previous_version": (
            registry.get_previous_version()
        ),
    }

@app.post(
    "/models/{version}/activate"
)
def activate_model(version: str):

    try:
        model_path = registry.get_model_path(
            version
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    # Load candidate before changing registry state.
    model_service.load_model(
        model_path
    )

    if (
        not model_service.is_ready
        or model_service.model_version
        != version
    ):
        raise HTTPException(
            status_code=500,
            detail=(
                "Candidate model could not "
                "be loaded safely."
            ),
        )

    state = registry.activate(
        version
    )

    return {
        "status": "activated",
        "active_version": state[
            "active_version"
        ],
        "previous_version": state[
            "previous_version"
        ],
    }

@app.post("/models/rollback")
def rollback_model():

    previous_version = (
        registry.get_previous_version()
    )

    if previous_version is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "No previous model version "
                "available for rollback."
            ),
        )

    try:
        model_path = registry.get_model_path(
            previous_version
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    model_service.load_model(
        model_path
    )

    if (
        not model_service.is_ready
        or model_service.model_version
        != previous_version
    ):
        raise HTTPException(
            status_code=500,
            detail=(
                "Rollback model could not "
                "be loaded safely."
            ),
        )

    state = registry.rollback()

    return {
        "status": "rollback_successful",
        "active_version": state[
            "active_version"
        ],
        "previous_version": state[
            "previous_version"
        ],
    }