"""
Tests for the liveness and readiness probes.

The registry lookup is patched, so these never touch MLflow: readiness has to be
testable in exactly the states that are awkward to produce for real - a registry
that is down, or one where only some aliases are set.
"""

import pytest


ALL_RESOLVED = {
    "pk_model": "3",
    "composite_pk_model": "2",
    "fk_model": "5",
    "composite_fk_model": "1",
    "denormalization_model": "4",
}


def test_liveness_does_not_touch_the_registry(client, monkeypatch):
    """A liveness probe must answer even when the registry lookup would explode."""

    def explode() -> dict:
        msg = "the registry must not be consulted for liveness"
        raise AssertionError(msg)

    monkeypatch.setattr("app.resolve_model_versions", explode)

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_ok_when_every_model_resolves(client, monkeypatch):
    monkeypatch.setattr("app.resolve_model_versions", lambda: dict(ALL_RESOLVED))

    response = client.get("/health/ready")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["alias"] == "dev" or body["alias"] == "prod"
    assert body["models"] == ALL_RESOLVED


def test_readiness_degraded_still_serves(client, monkeypatch):
    """Some models missing is 200: the ones that resolve are still servable."""
    partial = dict(ALL_RESOLVED)
    partial["fk_model"] = None
    partial["denormalization_model"] = None
    monkeypatch.setattr("app.resolve_model_versions", lambda: partial)

    response = client.get("/health/ready")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "degraded"
    # The detail has to name the offenders - that is the whole point of reading it.
    assert "denormalization_model" in body["detail"]
    assert "fk_model" in body["detail"]
    assert "3 of 5" in body["detail"]


def test_readiness_503_when_nothing_resolves(client, monkeypatch):
    """Registry down or nothing registered: no point routing traffic here."""
    monkeypatch.setattr(
        "app.resolve_model_versions",
        lambda: dict.fromkeys(ALL_RESOLVED),
    )

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"


@pytest.mark.parametrize("route", ["/health/live", "/health/ready"])
def test_probes_are_documented_in_openapi(client, route):
    """Both probes belong in the schema, or /docs stops matching the README."""
    paths = client.get("/openapi.json").json()["paths"]

    assert route in paths


def test_per_model_registry_error_reports_unavailable_not_500(client, monkeypatch):
    """
    resolve_model_versions swallows registry errors per model.

    This pins that contract: a connection error has to arrive as an unresolved
    model, not as a 500 out of the probe.
    """
    import predict

    class DeadClient:
        def get_model_version_by_alias(self, name: str, alias: str) -> None:
            msg = f"connection refused for {name}@{alias}"
            raise OSError(msg)

    monkeypatch.setattr(predict, "_configure_tracking", lambda: "http://mlflow:5000")
    monkeypatch.setattr(predict, "_registry_reachable", lambda _uri: True)
    monkeypatch.setattr(predict, "MlflowClient", DeadClient)

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"


def test_unreachable_registry_short_circuits(client, monkeypatch):
    """
    A tracking server that does not answer must not be queried per model.

    The MLflow client retries connection errors with backoff, so five sequential
    lookups against a dead server outlast any probe timeout. The probe has to give
    up after the reachability check instead - hence a client that fails the test if
    it is constructed at all.
    """
    import predict

    def unexpected_client(*args: object, **kwargs: object) -> None:
        msg = "an unreachable registry must not be queried model by model"
        raise AssertionError(msg)

    monkeypatch.setattr(predict, "_configure_tracking", lambda: "http://mlflow:5000")
    monkeypatch.setattr(predict, "_registry_reachable", lambda _uri: False)
    monkeypatch.setattr(predict, "MlflowClient", unexpected_client)

    response = client.get("/health/ready")
    body = response.json()

    assert response.status_code == 503
    assert body["status"] == "unavailable"
    assert set(body["models"]) == set(ALL_RESOLVED)
    assert all(version is None for version in body["models"].values())
