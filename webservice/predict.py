"""Helpers for loading the registered MLflow model and serving predictions."""

import os
from functools import lru_cache

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.pyfunc import PyFuncModel


@lru_cache(maxsize=1)
def load_model(model_name: str) -> PyFuncModel:
    alias = "dev"
    model_uri = f"models:/{model_name}@{alias}"

    model = mlflow.pyfunc.load_model(model_uri)
    return model


def predict(model_name: str, data: pd.DataFrame) -> tuple[int, float]:
    load_dotenv()
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_tracking_uri:
        msg_tracking_uri = "MLFLOW_TRACKING_URI is not set."
        raise RuntimeError(msg_tracking_uri)

    mlflow.set_tracking_uri(mlflow_tracking_uri)

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
    print("Making prediction with data: ", model_input.head())
    prediction = model.predict(model_input)

    raw_sklearn_model = model._model_impl.get_raw_model()
    probabilities = raw_sklearn_model.predict_proba(model_input)

    return float(prediction[0]), float(probabilities[0][1])

