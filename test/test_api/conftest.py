"""
Fixtures for the API tests.

These run against the FastAPI app in-process, with no stack and no MLflow: the
registry lookup is patched per test. The webservice is not an installable package
(its Docker build context is the directory itself), so its modules are imported by
putting that directory on the path - same approach as test/test_models/conftest.py.
"""

import sys
from pathlib import Path

import pytest


CONFTEST_DIR = Path(__file__).parent
REPO_ROOT = (CONFTEST_DIR / "../..").resolve()
WEBSERVICE_DIR = REPO_ROOT / "webservice"

if str(WEBSERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(WEBSERVICE_DIR))


@pytest.fixture
def client():
    """TestClient for the app. Imported lazily so collection does not need FastAPI."""
    import app as app_module
    from fastapi.testclient import TestClient

    return TestClient(app_module.app)
