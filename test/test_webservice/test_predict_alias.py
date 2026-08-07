"""
Tests for webservice/predict.py's model alias resolution and caching behavior.

mlflow.pyfunc.load_model is mocked out - no real MLflow server needed.
"""

import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest


TEST_DIR = Path(__file__).parent
REPO_ROOT = (TEST_DIR / "../..").resolve()
WEBSERVICE_DIR = REPO_ROOT / "webservice"
if str(WEBSERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(WEBSERVICE_DIR))

import predict  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_model_cache() -> Iterator[None]:
    predict.load_model.cache_clear()
    yield
    predict.load_model.cache_clear()


def test_load_model_defaults_to_dev_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_ALIAS", raising=False)
    calls = []
    monkeypatch.setattr(
        predict.mlflow.pyfunc,
        "load_model",
        lambda uri: calls.append(uri) or MagicMock(),
    )

    predict.load_model("pk_model")

    assert calls == ["models:/pk_model@dev"]


def test_load_model_honors_model_alias_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_ALIAS", "prod")
    calls = []
    monkeypatch.setattr(
        predict.mlflow.pyfunc,
        "load_model",
        lambda uri: calls.append(uri) or MagicMock(),
    )

    predict.load_model("pk_model")

    assert calls == ["models:/pk_model@prod"]


def test_load_model_is_cached_per_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_ALIAS", "prod")
    calls = []
    monkeypatch.setattr(
        predict.mlflow.pyfunc,
        "load_model",
        lambda uri: calls.append(uri) or MagicMock(),
    )

    predict.load_model("pk_model")
    predict.load_model("pk_model")

    # A promotion that moves @prod to a new version has no effect on an already
    # running process until it restarts - this pins that behavior deliberately.
    assert calls == ["models:/pk_model@prod"]
