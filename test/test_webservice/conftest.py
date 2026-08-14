"""
Fixtures for the webservice unit tests.

Nothing in this package needs a running stack: no MLflow, no Trino, no Evidently.
The MLflow model is replaced by `FakePyFuncModel` below, which reproduces the three
attributes `webservice/predict.py` actually touches:

* `metadata.get_input_schema()`      - the logged input signature
* `predict(frame)`                   - the pyfunc call, which reorders by name
* `_model_impl.get_raw_model()`      - the raw sklearn estimator used for
                                       `predict_proba`, which compares positionally

That last split is the whole reason the 27.07. bug existed, so the fake keeps it
rather than pretending a model is a single callable.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel


CONFTEST_DIR = Path(__file__).parent
REPO_ROOT = (CONFTEST_DIR / "../..").resolve()
WEBSERVICE_DIR = REPO_ROOT / "webservice"

# The webservice is not an installable package (its Docker context is the directory
# itself, and app.py imports its siblings as top-level modules), so its modules are
# imported by putting that directory on the path - same approach as test_models/.
if str(WEBSERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(WEBSERVICE_DIR))


# --- Request payload helper ---------------------------------------------------


def _example_value(annotation: Any) -> Any:
    """One valid value for a request-model field, derived from its annotation."""
    # bool before int: bool is a subclass of int, so the order matters.
    if annotation is bool:
        return False
    if annotation is int:
        return 1
    if annotation is float:
        return 0.5
    if annotation is str:
        return "example"
    # Anything else here is an optional field (`str | None`), whose default is fine.
    return None


def payload_for(model_cls: type[BaseModel]) -> dict:
    """
    Build a valid request body for a Pydantic request model.

    Derived from `model_fields` rather than written out by hand: the five request
    models carry 32-52 features each, and a hand-maintained fixture would be one
    more place to forget when the feature set changes - the exact failure mode
    behind the 27.07. schema drift.
    """
    return {
        name: _example_value(field.annotation) for name, field in model_cls.model_fields.items()
    }


# --- MLflow model stand-in ----------------------------------------------------


class FakeInputSchema:
    """Mimics `mlflow.types.Schema` for the one method predict.py calls."""

    def __init__(self, names) -> None:
        self._names = list(names)

    def input_names(self) -> list[str]:
        return list(self._names)


class FakeMetadata:
    def __init__(self, names) -> None:
        self._names = names

    def get_input_schema(self):
        """Return None when the model was logged without a signature."""
        if self._names is None:
            return None
        return FakeInputSchema(self._names)


class FakeRawModel:
    """The raw sklearn estimator: `predict_proba` compares column order positionally."""

    def __init__(self, probabilities, classes=None) -> None:
        self._probabilities = probabilities
        self.seen_columns: list[str] | None = None
        # Real estimators always carry this after fitting; predict_with_probabilities
        # zips it with a predict_proba row to label each position, so the fake has to
        # have one too. Defaults to 0..n-1, matching sklearn's default for integer
        # class labels.
        self.classes_ = np.array(classes if classes is not None else range(len(probabilities)))

    def predict_proba(self, model_input):
        self.seen_columns = list(model_input.columns)
        return np.array([list(self._probabilities)])


class FakeModelImpl:
    def __init__(self, raw_model) -> None:
        self._raw_model = raw_model

    def get_raw_model(self):
        return self._raw_model


class FakePyFuncModel:
    """Stand-in for `mlflow.pyfunc.PyFuncModel`."""

    def __init__(
        self,
        feature_names,
        prediction: int = 1,
        probabilities=(0.25, 0.75),
        classes=None,
    ) -> None:
        self.metadata = FakeMetadata(feature_names)
        self.raw_model = FakeRawModel(probabilities, classes=classes)
        self._model_impl = FakeModelImpl(self.raw_model)
        self._prediction = prediction
        self.seen_columns: list[str] | None = None
        self.seen_dtypes: dict | None = None

    def predict(self, model_input):
        self.seen_columns = list(model_input.columns)
        self.seen_dtypes = dict(model_input.dtypes)
        return np.array([self._prediction])


@pytest.fixture
def fake_model():
    """Binary model over three features, predicting class 1 with p=0.75."""
    return FakePyFuncModel(["a", "b", "c"], prediction=1, probabilities=(0.25, 0.75))
