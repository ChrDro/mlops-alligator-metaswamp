"""
Fixtures for the model/API schema contract tests.

Connecting to MLflow from the host is not quite as simple as reading `.env`: that
file carries the *compose-internal* hostnames (`mlflow:5000`, `minio:9000`), which
only resolve inside the Docker network. The helpers below fall back to `localhost`
for any hostname that does not resolve, so the same test runs unchanged on the host
and inside a container.

Reading a signature also touches the artifact store, so MinIO credentials have to be
present. They are derived from MINIO_ROOT_USER / MINIO_ROOT_PASSWORD unless AWS_*
variables are already set.

If MLflow cannot be reached at all - the normal case in CI, where no stack is
running - every test in this package skips rather than fails.
"""

import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import pytest
import requests
from dotenv import load_dotenv


CONFTEST_DIR = Path(__file__).parent
REPO_ROOT = (CONFTEST_DIR / "../..").resolve()
WEBSERVICE_DIR = REPO_ROOT / "webservice"

# The webservice is not an installable package (its Docker context is the directory
# itself), so its modules are imported by putting that directory on the path.
if str(WEBSERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(WEBSERVICE_DIR))


# (registered model name, module in webservice/, Pydantic class) for every model the
# API serves. A new model must be added here, otherwise its schema drifts unnoticed.
MODEL_CONTRACTS = [
    ("pk_model", "data_model_pk", "PrimaryKey"),
    ("composite_pk_model", "data_model_cpk", "CompositePrimaryKey"),
    ("fk_model", "data_model_fk", "ForeignKey"),
    ("composite_fk_model", "data_model_cfk", "CompositeForeignKey"),
    ("denormalization_model", "data_model_denormalization", "NormalForm"),
    ("subject_area_model", "data_model_subject_area", "SubjectArea"),
]

REACHABILITY_TIMEOUT_SECONDS = 3


def _reachable_url(url: str, fallback_host: str = "localhost") -> str:
    """Swap the hostname for `fallback_host` when it does not resolve."""
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return url

    try:
        socket.gethostbyname(hostname)
    except OSError:
        port = f":{parsed.port}" if parsed.port else ""
        return urlunparse(parsed._replace(netloc=f"{fallback_host}{port}"))
    return url


@pytest.fixture(scope="session")
def mlflow_client():
    """Configure MLflow against the running stack, or skip the whole package."""
    load_dotenv(REPO_ROOT / ".env")

    tracking_uri = _reachable_url(os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5000")
    s3_endpoint = _reachable_url(os.getenv("MLFLOW_S3_ENDPOINT_URL") or "http://localhost:9000")

    minio_user = os.getenv("MINIO_ROOT_USER")
    minio_password = os.getenv("MINIO_ROOT_PASSWORD")
    if not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
        if not (minio_user and minio_password):
            pytest.skip("No MinIO credentials available to read model artifacts.")
        os.environ["AWS_ACCESS_KEY_ID"] = minio_user
        os.environ["AWS_SECRET_ACCESS_KEY"] = minio_password
    os.environ.setdefault("AWS_DEFAULT_REGION", "eu-central-1")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint

    try:
        requests.get(f"{tracking_uri.rstrip('/')}/health", timeout=REACHABILITY_TIMEOUT_SECONDS)
    except requests.RequestException as error:
        pytest.skip(f"MLflow not reachable at {tracking_uri} ({error}); is the stack running?")

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    return mlflow


@pytest.fixture(scope="session")
def model_signatures(mlflow_client) -> dict:
    """
    Input signature per registered model, keyed by model name.

    A model that cannot be read stores the exception instead of a signature, so one
    missing model fails only its own test rather than erroring the whole session.
    """
    signatures: dict[str, object] = {}

    for model_name, _module, _cls in MODEL_CONTRACTS:
        try:
            info = mlflow_client.models.get_model_info(f"models:/{model_name}@dev")
            signatures[model_name] = [column.name for column in info.signature.inputs.inputs]
        except Exception as error:  # noqa: BLE001 - reported by the test that needs it
            signatures[model_name] = error

    return signatures


@pytest.fixture(scope="session")
def pydantic_fields() -> dict:
    """Declared field names per Pydantic request model, in declaration order."""
    import importlib

    fields = {}
    for model_name, module_name, class_name in MODEL_CONTRACTS:
        module = importlib.import_module(module_name)
        fields[model_name] = list(getattr(module, class_name).model_fields)
    return fields
