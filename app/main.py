from fastapi import FastAPI

app = FastAPI(
    title="ML Model Serving & Monitoring Platform",
    description="A platform for serving and monitoring ML models.",
    version="0.1.0"
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ml-model-serving-monitoring"
    }