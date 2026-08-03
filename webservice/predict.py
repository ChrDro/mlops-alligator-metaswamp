"""Helpers for loading the registered MLflow model and serving predictions."""

import os
from functools import lru_cache

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.pyfunc import PyFuncModel


@lru_cache(maxsize=5)
def load_model(model_name: str) -> PyFuncModel:
    alias = "dev"
    model_uri = f"models:/{model_name}@{alias}"

    model = mlflow.pyfunc.load_model(model_uri)
    return model


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


def _set_tracking_uri() -> None:
    load_dotenv()
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_tracking_uri:
        msg_tracking_uri = "MLFLOW_TRACKING_URI is not set."
        raise RuntimeError(msg_tracking_uri)

    mlflow.set_tracking_uri(mlflow_tracking_uri)


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
    _set_tracking_uri()

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
    _set_tracking_uri()

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
