"""
Unit tests for webservice/predict.py.

Both halves of this module have already been the cause of a production outage, and
each bug was invisible until predictions came back wrong or NULL:

* **27.07.** - `predict_proba` ran on the raw sklearn estimator with the request's
  column order, while `model.predict()` reordered by name through pyfunc. As soon as
  the Pydantic field order diverged from the training order, every call to that
  endpoint failed. `_align_to_signature()` is the fix, so its behaviour is pinned
  here in both directions (reordering *and* the error message on real drift).

* **28.07.** - the returned confidence was `probabilities[0][1]`, a hardcoded class
  index. For the 4-class normalform model that silently returned "probability of
  class 1" no matter what was predicted. The multiclass test below is built so that
  the old expression and the correct one give *different* numbers - it fails if
  anyone reintroduces a fixed index.

No MLflow is involved: `load_model` is replaced by the fake from conftest.
"""

import numpy as np
import pandas as pd
import predict as predict_module
import pytest

from .conftest import FakePyFuncModel


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """`load_model` is an lru_cache; leaking entries between tests hides bugs."""
    predict_module.load_model.cache_clear()
    yield
    predict_module.load_model.cache_clear()


@pytest.fixture
def tracking_uri(monkeypatch):
    """Satisfy predict()'s env requirement without reading the real .env."""
    monkeypatch.setattr(predict_module, "load_dotenv", lambda: None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://test-mlflow:5000")


# --- _align_to_signature ------------------------------------------------------


def test_align_reorders_columns_into_signature_order():
    model = FakePyFuncModel(["b", "c", "a"])
    frame = pd.DataFrame([{"a": 1.0, "b": 2.0, "c": 3.0}])

    aligned = predict_module._align_to_signature(model, frame, "pk_model")

    assert list(aligned.columns) == ["b", "c", "a"]
    # Values must travel with their column, not just the labels.
    assert aligned.iloc[0].tolist() == [2.0, 3.0, 1.0]


def test_align_passes_frame_through_when_model_has_no_signature():
    model = FakePyFuncModel(None)
    frame = pd.DataFrame([{"a": 1.0, "b": 2.0}])

    aligned = predict_module._align_to_signature(model, frame, "pk_model")

    assert list(aligned.columns) == ["a", "b"]


def test_align_error_names_the_missing_features():
    model = FakePyFuncModel(["a", "b", "missing_one"])
    frame = pd.DataFrame([{"a": 1.0, "b": 2.0}])

    with pytest.raises(ValueError, match="missing_one") as error:
        predict_module._align_to_signature(model, frame, "fk_model")

    message = str(error.value)
    assert "fk_model" in message
    assert "Missing: ['missing_one']" in message
    assert "Unexpected: none" in message


def test_align_error_names_the_unexpected_features():
    model = FakePyFuncModel(["a"])
    frame = pd.DataFrame([{"a": 1.0, "stowaway": 2.0}])

    with pytest.raises(ValueError, match="stowaway") as error:
        predict_module._align_to_signature(model, frame, "fk_model")

    assert "Unexpected: ['stowaway']" in str(error.value)


# --- predict ------------------------------------------------------------------


def test_predict_returns_probability_of_the_predicted_class(monkeypatch, tracking_uri):
    """
    Regression test for the 28.07. multiclass bug.

    Class 3 is predicted with p=0.55, while class index 1 has p=0.20. The old
    `probabilities[0][1]` would return 0.20 here - a number that belongs to a class
    the model did not predict.
    """
    model = FakePyFuncModel(["a"], prediction=3, probabilities=(0.10, 0.20, 0.15, 0.55))
    monkeypatch.setattr(predict_module, "load_model", lambda _name: model)

    prediction, probability = predict_module.predict(
        "denormalization_model",
        pd.DataFrame([{"a": 1.0}]),
    )

    assert prediction == 3.0
    assert probability == pytest.approx(0.55)


def test_predict_returns_probability_of_the_predicted_class_for_binary(monkeypatch, tracking_uri):
    """Predicting class 0 must return p(class 0), not p(class 1)."""
    model = FakePyFuncModel(["a"], prediction=0, probabilities=(0.85, 0.15))
    monkeypatch.setattr(predict_module, "load_model", lambda _name: model)

    prediction, probability = predict_module.predict("pk_model", pd.DataFrame([{"a": 1.0}]))

    assert prediction == 0.0
    assert probability == pytest.approx(0.85)


def test_predict_sends_identical_column_order_to_pyfunc_and_to_predict_proba(
    monkeypatch,
    tracking_uri,
):
    """
    The core of the 27.07. bug: pyfunc reorders by name, the raw estimator does not.
    Both must therefore see the signature's order.
    """
    model = FakePyFuncModel(["c", "a", "b"])
    monkeypatch.setattr(predict_module, "load_model", lambda _name: model)

    predict_module.predict("pk_model", pd.DataFrame([{"a": 1.0, "b": 2.0, "c": 3.0}]))

    assert model.seen_columns == ["c", "a", "b"]
    assert model.raw_model.seen_columns == ["c", "a", "b"]


def test_predict_casts_booleans_to_numeric(monkeypatch, tracking_uri):
    """
    The models are trained on 0/1 integers, but the request schemas declare the
    column_type_* features as bool, so every payload arrives with bool dtype.
    """
    model = FakePyFuncModel(["flag", "value"])
    monkeypatch.setattr(predict_module, "load_model", lambda _name: model)

    predict_module.predict("pk_model", pd.DataFrame([{"flag": True, "value": 2}]))

    assert all(dtype == np.float64 for dtype in model.seen_dtypes.values())
    assert model.seen_columns == ["flag", "value"]


def test_predict_does_not_mutate_the_caller_frame(monkeypatch, tracking_uri):
    """app.py builds the frame from the request; predict() must not alter it."""
    model = FakePyFuncModel(["flag"])
    monkeypatch.setattr(predict_module, "load_model", lambda _name: model)
    frame = pd.DataFrame([{"flag": True}])

    predict_module.predict("pk_model", frame)

    assert frame["flag"].dtype == bool


def test_predict_rejects_a_non_dataframe(tracking_uri):
    with pytest.raises(TypeError, match="Expected DataFrame"):
        predict_module.predict("pk_model", {"a": 1.0})


def test_predict_requires_a_tracking_uri(monkeypatch):
    """Without MLFLOW_TRACKING_URI the model can never load - fail loudly, not late."""
    monkeypatch.setattr(predict_module, "load_dotenv", lambda: None)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI is not set"):
        predict_module.predict("pk_model", pd.DataFrame([{"a": 1.0}]))


# --- predict_with_probabilities ------------------------------------------------


def test_predict_with_probabilities_returns_the_full_distribution(monkeypatch, tracking_uri):
    """The normalform table-level vote needs every class's probability, not just the max."""
    model = FakePyFuncModel(["a"], prediction=3, probabilities=(0.10, 0.20, 0.15, 0.55))
    monkeypatch.setattr(predict_module, "load_model", lambda _name: model)

    prediction, probability, distribution = predict_module.predict_with_probabilities(
        "denormalization_model",
        pd.DataFrame([{"a": 1.0}]),
    )

    assert prediction == 3.0
    assert probability == pytest.approx(0.55)
    assert distribution == {
        "0": pytest.approx(0.10),
        "1": pytest.approx(0.20),
        "2": pytest.approx(0.15),
        "3": pytest.approx(0.55),
    }


def test_predict_with_probabilities_keys_by_label_not_position(monkeypatch, tracking_uri):
    """
    `classes_` need not be sorted ascending in general - keying by position instead
    of by label would silently attach probabilities to the wrong class.
    """
    model = FakePyFuncModel(
        ["a"],
        prediction=2,
        probabilities=(0.60, 0.30, 0.10),
        classes=[2, 0, 1],
    )
    monkeypatch.setattr(predict_module, "load_model", lambda _name: model)

    _prediction, _probability, distribution = predict_module.predict_with_probabilities(
        "denormalization_model",
        pd.DataFrame([{"a": 1.0}]),
    )

    assert distribution == {
        "2": pytest.approx(0.60),
        "0": pytest.approx(0.30),
        "1": pytest.approx(0.10),
    }


# --- load_model ---------------------------------------------------------------


def test_load_model_requests_the_configured_alias_and_caches_per_model(monkeypatch):
    """
    Pins two things: the URI shape, and that the lru_cache actually prevents a second
    download for the same model.

    `MODEL_ALIAS` defaults to `"dev"` (item 3.2/5.3, "Model Alias/Version as Env
    Variable", closed via `os.getenv("MODEL_ALIAS", "dev")`). This test asserts that
    default, and the cache means a promotion that moves `@prod` to a new version has
    no effect on an already-running service until it restarts.
    """
    loaded_uris = []

    def fake_load_model(model_uri):
        loaded_uris.append(model_uri)
        return FakePyFuncModel(["a"])

    monkeypatch.setattr(predict_module.mlflow.pyfunc, "load_model", fake_load_model)

    predict_module.load_model("pk_model")
    predict_module.load_model("pk_model")
    predict_module.load_model("fk_model")

    assert loaded_uris == ["models:/pk_model@dev", "models:/fk_model@dev"]


def test_load_model_honors_a_non_default_model_alias(monkeypatch):
    """
    `MODEL_ALIAS` is read once at import time into a module constant, not re-read per
    call - so tests (and callers) override the constant directly, not the env var.
    """
    loaded_uris = []

    def fake_load_model(model_uri):
        loaded_uris.append(model_uri)
        return FakePyFuncModel(["a"])

    monkeypatch.setattr(predict_module.mlflow.pyfunc, "load_model", fake_load_model)
    monkeypatch.setattr(predict_module, "MODEL_ALIAS", "prod")

    predict_module.load_model("pk_model")

    assert loaded_uris == ["models:/pk_model@prod"]
