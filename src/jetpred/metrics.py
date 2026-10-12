from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter(
    "jetpred_http_requests_total",
    "HTTP requests by route and status code",
    ["method", "path", "status"],
)

HTTP_LATENCY = Histogram(
    "jetpred_http_request_duration_seconds",
    "End-to-end HTTP request latency",
    ["method", "path"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

INFERENCE_LATENCY = Histogram(
    "jetpred_inference_seconds",
    "Model inference time, excluding HTTP and validation",
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

BATCH_SIZE = Histogram(
    "jetpred_batch_size",
    "Number of jets per prediction request",
    buckets=(1, 2, 5, 10, 25, 50, 100, 250, 500, 1000),
)

PREDICTIONS = Counter(
    "jetpred_predictions_total",
    "Predictions served, by predicted class",
    ["predicted_class", "model_version"],
)

PREDICTION_CONFIDENCE = Histogram(
    "jetpred_prediction_confidence",
    "Probability of the predicted class",
    buckets=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)