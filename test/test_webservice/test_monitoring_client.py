"""
Tests for webservice/monitoring_client.py.

This module makes exactly one promise: **monitoring can never fail a prediction.**
It runs as a FastAPI background task, so an escaping exception would be logged as an
unhandled error for a side effect the caller never asked about - and with the monitor
down it would happen on every single request.

That promise was the one thing the module was not covering itself, so every failure
mode below is asserted separately rather than trusting the two `except` clauses by
inspection. `requests.post` is replaced throughout; no Evidently service is involved.
"""

import logging

import monitoring_client
import pytest
import requests


TRACK = "pk_columns"
PAYLOAD = {"number_unique_values": 5, "prediction": 1, "probability": 0.9}


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error


@pytest.fixture
def posts(monkeypatch):
    """Record every requests.post call instead of performing it."""
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(monitoring_client.requests, "post", fake_post)
    monkeypatch.setattr(monitoring_client, "MONITORING_ENABLED", True)
    return calls


def _post_raising(error: Exception):
    """A requests.post replacement that raises when called."""

    def fake_post(url, json=None, timeout=None):
        raise error

    return fake_post


# --- happy path ---------------------------------------------------------------


def test_forwards_the_payload_to_the_tracks_iterate_endpoint(posts):
    monitoring_client.forward_to_monitoring(PAYLOAD, TRACK)

    assert len(posts) == 1
    assert posts[0]["url"] == f"{monitoring_client.MONITORING_BASE_URL}/iterate/{TRACK}"
    assert posts[0]["json"] == PAYLOAD


def test_uses_the_configured_timeout(posts):
    """
    A hung monitor must not tie up a worker thread per prediction, so the timeout is
    short by design and non-optional.
    """
    monitoring_client.forward_to_monitoring(PAYLOAD, TRACK)

    assert posts[0]["timeout"] == monitoring_client.MONITORING_TIMEOUT_SECONDS
    assert posts[0]["timeout"] <= 5


def test_each_track_gets_its_own_url(posts):
    """
    Every model has its own feature schema, reference and DataDefinition in the
    Evidently service, so the track is part of the path rather than the payload.
    """
    for track in ("pk_columns", "cpk_columns", "fk_columns", "cfk_columns", "nf_columns"):
        monitoring_client.forward_to_monitoring(PAYLOAD, track)

    assert [call["url"].rsplit("/", 1)[-1] for call in posts] == [
        "pk_columns",
        "cpk_columns",
        "fk_columns",
        "cfk_columns",
        "nf_columns",
    ]


# --- the disable switch -------------------------------------------------------


def test_makes_no_request_when_monitoring_is_disabled(posts, monkeypatch):
    """MONITORING_ENABLED=false is for running the API with no monitor on the network."""
    monkeypatch.setattr(monitoring_client, "MONITORING_ENABLED", False)

    monitoring_client.forward_to_monitoring(PAYLOAD, TRACK)

    assert posts == []


# --- the failure contract -----------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        requests.exceptions.ConnectionError("monitor is down"),
        requests.exceptions.Timeout("monitor is slow"),
        requests.exceptions.RequestException("something requests-shaped"),
    ],
    ids=["connection-error", "timeout", "generic-request-exception"],
)
def test_swallows_request_failures(monkeypatch, error):
    monkeypatch.setattr(monitoring_client, "MONITORING_ENABLED", True)
    monkeypatch.setattr(monitoring_client.requests, "post", _post_raising(error))

    assert monitoring_client.forward_to_monitoring(PAYLOAD, TRACK) is None


def test_swallows_an_error_status(monkeypatch):
    """A 4xx/5xx from the monitor reaches this code through raise_for_status."""
    monkeypatch.setattr(monitoring_client, "MONITORING_ENABLED", True)
    monkeypatch.setattr(
        monitoring_client.requests,
        "post",
        lambda url, json=None, timeout=None: FakeResponse(
            requests.exceptions.HTTPError("500 Server Error"),
        ),
    )

    assert monitoring_client.forward_to_monitoring(PAYLOAD, TRACK) is None


def test_swallows_exceptions_that_are_not_request_errors(monkeypatch):
    """
    The bare `except Exception` is the actual safety net: a payload the monitor client
    cannot serialise, a DNS library raising something exotic, a typo in this module.
    None of it may reach the prediction path.
    """
    monkeypatch.setattr(monitoring_client, "MONITORING_ENABLED", True)
    monkeypatch.setattr(
        monitoring_client.requests,
        "post",
        _post_raising(TypeError("Object of type DataFrame is not JSON serializable")),
    )

    assert monitoring_client.forward_to_monitoring(PAYLOAD, TRACK) is None


def test_logs_a_transport_failure_at_debug_level(monkeypatch, caplog):
    """
    debug, not warning: with the monitor down this fires on every prediction, and at
    warning level it would bury real errors in the service log.
    """
    monkeypatch.setattr(monitoring_client, "MONITORING_ENABLED", True)
    monkeypatch.setattr(
        monitoring_client.requests,
        "post",
        _post_raising(requests.exceptions.ConnectionError("monitor is down")),
    )

    with caplog.at_level(logging.DEBUG, logger="monitoring_client"):
        monitoring_client.forward_to_monitoring(PAYLOAD, TRACK)

    records = [r for r in caplog.records if r.name == "monitoring_client"]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    assert "Could not forward prediction" in records[0].message


def test_logs_an_unexpected_failure_with_a_traceback(monkeypatch, caplog):
    """
    The opposite choice from above: a non-transport error is a real defect, so it is
    logged at error level with the traceback - swallowed, but not hidden.
    """
    monkeypatch.setattr(monitoring_client, "MONITORING_ENABLED", True)
    monkeypatch.setattr(monitoring_client.requests, "post", _post_raising(TypeError("boom")))

    with caplog.at_level(logging.DEBUG, logger="monitoring_client"):
        monitoring_client.forward_to_monitoring(PAYLOAD, TRACK)

    records = [r for r in caplog.records if r.name == "monitoring_client"]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert records[0].exc_info is not None
