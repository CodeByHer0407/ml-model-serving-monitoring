from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status

from app.model_service import ModelService
from app.schemas import PredictionRequest, PredictionResponse


model_service = ModelService(
    model_path=Path("models/rul_model_v1.joblib"),
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