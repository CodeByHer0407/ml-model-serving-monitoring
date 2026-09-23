from prometheus_client import Counter, Histogram


HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "route", "status_code"],
)


HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "route"],
)


MODEL_PREDICTIONS = Counter(
    "model_predictions_total",
    "Total number of model predictions",
    ["model_version", "status"],
)


MODEL_PREDICTION_DURATION = Histogram(
    "model_prediction_duration_seconds",
    "Model inference duration in seconds",
    ["model_version"],
)

