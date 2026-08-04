"""Helpers for loading the registered MLflow model and serving predictions."""

import os
from functools import lru_cache

import mlflow
import pandas as pd
import requests
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient


# Every registered model the API serves, and the alias it serves them under. The
# health endpoint reports on exactly this set, so a new model has to be listed here
# to show up in readiness.
MODEL_NAMES = (
    "pk_model",
    "composite_pk_model",
    "fk_model",
    "composite_fk_model",
    "denormalization_model",
)
MODEL_ALIAS = "dev"

# Budget for the reachability check below. Short on purpose: it exists to keep a
# readiness probe inside a probe-sized deadline, not to wait out a slow server.
REGISTRY_PROBE_TIMEOUT_SECONDS = 2


def _configure_tracking() -> str:
    """Point the MLflow client at the configured tracking server, and return its URI."""
    load_dotenv()
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_tracking_uri:
        msg_tracking_uri = "MLFLOW_TRACKING_URI is not set."
        raise RuntimeError(msg_tracking_uri)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    return mlflow_tracking_uri


def _registry_reachable(tracking_uri: str) -> bool:
    """
    Cheap check that the tracking server answers at all.

    Without this a readiness probe against a stopped MLflow does not fail, it
    *hangs*: the MLflow client retries connection errors with exponential backoff,
    once per model, and the request outlives any probe timeout worth setting. Ask
    once with a short deadline instead, and treat silence as "nothing to serve".
    """
    try:
        requests.get(
            f"{tracking_uri.rstrip('/')}/health",
            timeout=REGISTRY_PROBE_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return False

    return True


@lru_cache(maxsize=5)
def load_model(model_name: str) -> PyFuncModel:
    model_uri = f"models:/{model_name}@{MODEL_ALIAS}"

    model = mlflow.pyfunc.load_model(model_uri)
    return model


def resolve_model_versions() -> dict[str, str | None]:
    """
    Ask the registry which version the serving alias points at, per model.

    Deliberately does not download artifacts: a readiness probe has to stay cheap,
    and it must not warm the lru_cache in load_model as a side effect. It answers
    "is there something to serve", not "is it already in memory".
    """
    tracking_uri = _configure_tracking()
    if not _registry_reachable(tracking_uri):
        return dict.fromkeys(MODEL_NAMES)

    client = MlflowClient()

    versions: dict[str, str | None] = {}
    for model_name in MODEL_NAMES:
        try:
            versions[model_name] = client.get_model_version_by_alias(
                model_name,
                MODEL_ALIAS,
            ).version
        except (MlflowException, OSError):
            # Model not registered, alias not set, or the registry is unreachable.
            # All three mean the same thing to a caller: nothing to serve here.
            versions[model_name] = None

    return versions


def _align_to_signature(
    model: PyFuncModel,
    model_input: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """
    Reorder the request columns to match the model's logged input signature.

    Raises a message naming the offending fields when the request and the model
    disagree on *which* features exist - that is a schema drift between the Pydantic
    model and the registered model, and the bare sklearn error does not say so.
    """
    input_schema = model.metadata.get_input_schema()
    if input_schema is None:
        # No signature logged - nothing to align against, let the model decide.
        return model_input

    expected = list(input_schema.input_names())
    provided = set(model_input.columns)

    missing = []
    for name in expected:
        if name not in provided:
            missing.append(name)

    extra = []
    for name in model_input.columns:
        if name not in expected:
            extra.append(name)

    if missing or extra:
        msg = (
            f"Feature mismatch for '{model_name}': "
            f"the model expects {len(expected)} features but the request does not "
            f"match. Missing: {missing or 'none'}. Unexpected: {extra or 'none'}. "
            f"The Pydantic schema and the registered model are out of sync - retrain "
            f"the model or update the request schema."
        )
        raise ValueError(msg)

    return model_input[expected]


def predict_domain(model_name: str, data: pd.DataFrame) -> tuple[str, float]:
    """Predict domain from text features and return (label, confidence).

    Separate from `predict` below because that one casts every column to float for
    the five statistics-based models. The subject-area model is a TF-IDF pipeline
    over `table_name` and `columns`, so the same cast would turn its only inputs into
    NaN. Everything else is identical: align to the logged signature, predict, then
    read the confidence off the raw estimator's predict_proba.

    The label is whatever class the model was trained on - here a subject-area name -
    so this returns a str rather than the int the numeric models return.
    """
    _configure_tracking()

    if not isinstance(data, pd.DataFrame):
        msg_type_error_dataframe = f"Expected DataFrame, got {type(data)}"
        raise TypeError(msg_type_error_dataframe)

    model_input = data.copy()
    for col in model_input.columns:
        model_input[col] = model_input[col].astype(str)

    model = load_model(model_name)
    model_input = _align_to_signature(model, model_input, model_name)

    prediction = model.predict(model_input)

    raw_model = model._model_impl.get_raw_model()
    probabilities = raw_model.predict_proba(model_input)

    return str(prediction[0]), float(probabilities[0].max())


def predict(model_name: str, data: pd.DataFrame) -> tuple[int, float]:
    _configure_tracking()

    # Ensure data is a DataFrame and convert to proper dtypes
    if not isinstance(data, pd.DataFrame):
        msg_type_error_dataframe = f"Expected DataFrame, got {type(data)}"
        raise TypeError(msg_type_error_dataframe)

    model_input = data.copy()
    for col in model_input.columns:
        if model_input[col].dtype == bool:
            model_input[col] = model_input[col].astype(int)

    model_input = model_input.astype(float)

    print("Load model...")

    model = load_model(model_name)

    # Aligns columns to model's signature before predicting
    # Feature names should match those passed during fit in naming and order
    model_input = _align_to_signature(model, model_input, model_name)

    print("Making prediction with data: ", model_input.head())
    prediction = model.predict(model_input)

    raw_sklearn_model = model._model_impl.get_raw_model()
    probabilities = raw_sklearn_model.predict_proba(model_input)

    # Confidence = probability of the PREDICTED class. Since the models predict via
    # argmax, this is max(probabilities). Works for both binary (pk/fk/cpk/cfk) and
    # the multiclass normalform model. Previously this returned probabilities[0][1]
    # (hardcoded class index 1), which is meaningless for a >2-class model.
    return float(prediction[0]), float(probabilities[0].max())
