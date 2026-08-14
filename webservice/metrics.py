"""Custom Prometheus metrics for model predictions.

The FastAPI instrumentator already exposes the HTTP golden signals on /metrics.
This module adds model-level ("model_*") metrics so we can alert on model
behaviour rather than only HTTP health -- for example a model that always
predicts a single class, a collapse in prediction confidence, or a spike in
prediction errors.

prometheus_client is pulled in transitively by prometheus-fastapi-instrumentator,
so no extra dependency is required.
"""

import contextlib

from prometheus_client import Counter, Gauge, Histogram


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


# The offline F1 of the model version currently served, read from its MLflow run.
#
# This exists because the only F1 Grafana had came from the Prefect backtest, and that
# one is in-sample: the key models are refit on every labelled row, so the backtest
# replays rows the model trained on and reads far too high (~0.99 accuracy for fk_model
# against an honest 0.81). A number that looks like quality but is not gets read as
# quality, so the honest one is published next to it.
#
# The `estimator` label says how it was measured, which is not cosmetic: fk_model and
# composite_pk_model report a 5-fold grouped cross-validation mean, while the models
# that have not been migrated yet still report a single-fold holdout score whose
# fold-to-fold spread was measured at up to 0.21 F1. Same gauge, very different
# trustworthiness, so the panel can show which is which.
MODEL_OFFLINE_F1 = Gauge(
    "model_offline_f1",
    "Offline F1 of the served model version, from its MLflow run. See estimator label.",
    ["model", "estimator"],
)

# Spread of the offline estimate. Only set for models evaluated by cross-validation -
# a single holdout has no spread to report, which is precisely its weakness.
MODEL_OFFLINE_F1_STD = Gauge(
    "model_offline_f1_std",
    "Standard deviation of the served model's per-fold offline F1.",
    ["model"],
)

# Which registry version each alias currently resolves to. Lets a quality change on the
# dashboard be lined up against a deployment instead of guessed at.
MODEL_SERVED_VERSION = Gauge(
    "model_served_version",
    "Registry version number the serving alias currently points at.",
    ["model"],
)


def record_offline_quality(
    model: str,
    f1: float,
    estimator: str,
    version: str | None = None,
    f1_std: float | None = None,
) -> None:
    """Publish one served model's offline quality, as logged by its training run."""
    MODEL_OFFLINE_F1.labels(model=model, estimator=estimator).set(f1)
    if f1_std is not None:
        MODEL_OFFLINE_F1_STD.labels(model=model).set(f1_std)
    if version is not None:
        # Registry versions are numeric strings today; a non-numeric one is not worth
        # failing a metrics export over.
        with contextlib.suppress(TypeError, ValueError):
            MODEL_SERVED_VERSION.labels(model=model).set(float(version))


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
