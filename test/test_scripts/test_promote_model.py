"""
Hermetic tests for scripts/promote_model.py against a local MLflow file store.

No Docker/Postgres/MinIO needed: the tracking URI is pointed at a throwaway
directory before the module is imported, so registry reads/writes never touch
the real stack.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier


TEST_DIR = Path(__file__).parent
REPO_ROOT = (TEST_DIR / "../..").resolve()
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# promote_model.py sets the MLflow tracking URI as a module-level side effect (same
# pattern as the task_*_train_and_register.py scripts), so it has to be pointed at a
# throwaway local store *before* import - never at whatever server .env names. MLflow
# 3.x put the plain filesystem backend into maintenance mode, so sqlite is used here
# instead of "file:...".
_STORE_DIR = tempfile.mkdtemp(prefix="promote_model_test_mlflow_")
_TRACKING_URI = f"sqlite:///{_STORE_DIR}/mlflow.db"
os.environ["MLFLOW_TRACKING_URI"] = _TRACKING_URI
mlflow.set_tracking_uri(_TRACKING_URI)

import promote_model  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_tracking_uri() -> None:
    # webservice/predict.py's own test suite calls mlflow.set_tracking_uri() to a fake
    # host mid-run; that mutates process-global state monkeypatch does not revert, so
    # a shared pytest session would otherwise leave this module's mlflow.* write calls
    # (log_model, register_model, ...) pointed at an unreachable host.
    mlflow.set_tracking_uri(_TRACKING_URI)


@pytest.fixture
def client() -> MlflowClient:
    return MlflowClient(tracking_uri=_TRACKING_URI)


def _register_version(
    client: MlflowClient,
    model_name: str,
    f1: float | None,
    alias: str | None = None,
) -> str:
    """Log a run (optionally with a test_f1_score metric) and register a version."""
    with mlflow.start_run() as run:
        if f1 is not None:
            mlflow.log_metric("test_f1_score", f1)
        mlflow.sklearn.log_model(
            DummyClassifier(strategy="constant", constant=0).fit([[0], [1]], [0, 1]),
            name="model",
        )
        run_id = run.info.run_id

    registration = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)

    deadline = time.time() + 30
    version = None
    while time.time() < deadline:
        version = client.get_model_version(model_name, registration.version)
        if version.status == "READY":
            break
        time.sleep(0.2)

    if alias:
        client.set_registered_model_alias(model_name, alias, registration.version)
    return registration.version


def test_bootstrap_promotes_unconditionally(client: MlflowClient) -> None:
    model_name = "test_bootstrap"
    _register_version(client, model_name, f1=0.5, alias="dev")

    outcome = promote_model.decide_and_promote(client, model_name)

    assert outcome.action == "bootstrap"
    dev_version = client.get_model_version_by_alias(model_name, "dev").version
    assert client.get_model_version_by_alias(model_name, "prod").version == dev_version


def test_no_dev_alias_is_a_noop(client: MlflowClient) -> None:
    outcome = promote_model.decide_and_promote(client, "test_no_dev")

    assert outcome.action == "no_dev"


def test_better_dev_gets_promoted(client: MlflowClient) -> None:
    model_name = "test_promote"
    _register_version(client, model_name, f1=0.7, alias="prod")
    _register_version(client, model_name, f1=0.9, alias="dev")

    outcome = promote_model.decide_and_promote(client, model_name)

    assert outcome.action == "promoted"
    dev_version = client.get_model_version_by_alias(model_name, "dev").version
    assert client.get_model_version_by_alias(model_name, "prod").version == dev_version


def test_worse_dev_is_rejected(client: MlflowClient) -> None:
    model_name = "test_reject"
    _register_version(client, model_name, f1=0.9, alias="prod")
    prod_before = client.get_model_version_by_alias(model_name, "prod").version
    _register_version(client, model_name, f1=0.5, alias="dev")

    outcome = promote_model.decide_and_promote(client, model_name)

    assert outcome.action == "rejected"
    assert client.get_model_version_by_alias(model_name, "prod").version == prod_before


def test_equal_metric_promotes(client: MlflowClient) -> None:
    model_name = "test_tie"
    _register_version(client, model_name, f1=0.8, alias="prod")
    _register_version(client, model_name, f1=0.8, alias="dev")

    outcome = promote_model.decide_and_promote(client, model_name)

    assert outcome.action == "promoted"


def test_same_version_is_a_noop(client: MlflowClient) -> None:
    model_name = "test_noop"
    version = _register_version(client, model_name, f1=0.8, alias="dev")
    client.set_registered_model_alias(model_name, "prod", version)

    outcome = promote_model.decide_and_promote(client, model_name)

    assert outcome.action == "noop_same_version"


def test_missing_metric_is_reported(client: MlflowClient) -> None:
    model_name = "test_missing_metric"
    _register_version(client, model_name, f1=None, alias="dev")

    outcome = promote_model.decide_and_promote(client, model_name)

    assert outcome.action == "missing_metric"


def test_dry_run_does_not_move_alias(client: MlflowClient) -> None:
    model_name = "test_dry_run"
    _register_version(client, model_name, f1=0.7, alias="prod")
    _register_version(client, model_name, f1=0.9, alias="dev")
    prod_before = client.get_model_version_by_alias(model_name, "prod").version

    outcome = promote_model.decide_and_promote(client, model_name, dry_run=True)

    assert outcome.action == "promoted"
    assert client.get_model_version_by_alias(model_name, "prod").version == prod_before
