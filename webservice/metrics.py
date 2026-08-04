"""Custom Prometheus metrics for model predictions.

The FastAPI instrumentator already exposes the HTTP golden signals on /metrics.
This module adds model-level ("model_*") metrics so we can alert on model
behaviour rather than only HTTP health -- for example a model that always
predicts a single class, a collapse in prediction confidence, or a spike in
prediction errors.

prometheus_client is pulled in transitively by prometheus-fastapi-instrumentator,
so no extra dependency is required.
"""

from prometheus_client import Counter, Histogram


# Count of predictions returned, split by model and the class it predicted.
# A model stuck on one predicted_class over time is a strong drift signal.
PREDICTION_TOTAL = Counter(
    "model_predictions_total",
    "Number of predictions returned, labelled by model and predicted class.",
    ["model", "predicted_class"],
)

# Count of failed predictions, labelled by the exception type that caused them.
PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Number of failed predictions, labelled by model and error type.",
    ["model", "error_type"],
)

# Probability the model assigned to the class it predicted. A confidence
# distribution that collapses towards low values often precedes bad output.
PREDICTION_PROBABILITY = Histogram(
    "model_prediction_probability",
    "Probability the model assigned to the class it predicted.",
    ["model"],
)

# Wall-clock time spent inside the model prediction call.
PREDICTION_DURATION = Histogram(
    "model_prediction_duration_seconds",
    "Time spent inside the model prediction call.",
    ["model"],
)


def record_success(
    model: str,
    predicted_class: int,
    probability: float,
    duration: float,
) -> None:
    """Record metrics for a successful prediction."""
    PREDICTION_TOTAL.labels(model=model, predicted_class=str(predicted_class)).inc()
    PREDICTION_PROBABILITY.labels(model=model).observe(probability)
    PREDICTION_DURATION.labels(model=model).observe(duration)


def record_error(model: str, error: Exception) -> None:
    """Record a failed prediction, labelled by the exception type."""
    PREDICTION_ERRORS.labels(model=model, error_type=type(error).__name__).inc()
