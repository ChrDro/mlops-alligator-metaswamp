"""
Tests for webservice/event_publisher.py - the push half of the streaming trigger.

The failure contract here is the exact opposite of monitoring_client's: a lost event
means the change detector is not woken early, so this path *raises* and lets the
endpoint answer 503. (No data is lost either way - the detector also runs on cron -
but the caller is told it may retry.)

Two details are asserted because both were learned the hard way and are invisible
until the trigger silently stops working:

* `id` and `occurred` are sent explicitly. The Prefect *client* fills them in via
  default_factory, but the raw REST endpoint requires them and answers 422 without.
* the error message carries the response body. A bare "422 Unprocessable Entity" does
  not say which field was rejected.

`requests.post` is replaced throughout; no Prefect server is involved.
"""

import uuid
from datetime import datetime

import event_publisher
import pytest
import requests
from event_publisher import EventPublishError, publish_new_data_event


class FakeResponse:
    def __init__(self, error: Exception | None = None, text: str = "") -> None:
        self._error = error
        self.text = text

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error


@pytest.fixture
def posts(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(event_publisher.requests, "post", fake_post)
    return calls


# --- happy path ---------------------------------------------------------------


def test_posts_one_event_to_the_prefect_event_intake(posts):
    publish_new_data_event(schema="new_predict_data")

    assert len(posts) == 1
    assert posts[0]["url"] == f"{event_publisher.PREFECT_API_URL}/events"
    # The intake takes a list, even for a single event.
    assert isinstance(posts[0]["json"], list)
    assert len(posts[0]["json"]) == 1


def test_event_carries_the_names_the_automations_match_on(posts):
    """
    These two strings are duplicated in prefect/change_events.py (the webservice image
    cannot import that module). If they drift apart, the webhook publishes an event no
    deployment is listening for - and the only symptom is that push triggers stop
    working while the cron fallback quietly covers for them.
    """
    publish_new_data_event(schema="new_predict_data")

    event = posts[0]["json"][0]
    assert event["event"] == "alligator.new-data.arrived"
    assert event["resource"]["prefect.resource.id"] == "alligator.metaswamp/new-predict-data"


def test_event_includes_id_and_occurred(posts):
    """Required by the REST endpoint; the SDK would have defaulted them."""
    publish_new_data_event(schema="new_predict_data")

    event = posts[0]["json"][0]
    # Must be a parseable UUID and an ISO timestamp, not arbitrary strings.
    uuid.UUID(event["id"])
    parsed = datetime.fromisoformat(event["occurred"])
    assert parsed.tzinfo is not None, "occurred must be timezone-aware"


def test_event_payload_carries_the_schema_and_the_note(posts):
    publish_new_data_event(schema="other_schema", note="nightly load finished")

    event = posts[0]["json"][0]
    assert event["resource"]["alligator.schema"] == "other_schema"
    assert event["payload"] == {
        "schema": "other_schema",
        "note": "nightly load finished",
        "source": "webhook",
    }


def test_a_missing_note_is_sent_as_an_empty_string(posts):
    publish_new_data_event(schema="new_predict_data")

    assert posts[0]["json"][0]["payload"]["note"] == ""


def test_uses_the_configured_timeout(posts):
    publish_new_data_event(schema="new_predict_data")

    assert posts[0]["timeout"] == event_publisher.EVENT_TIMEOUT_SECONDS


# --- failure contract ---------------------------------------------------------


def _post_raising(error: Exception):
    """A requests.post replacement that raises a prepared exception."""

    def fake_post(url, json=None, timeout=None):
        raise error

    return fake_post


CONNECTION_ERROR = requests.exceptions.ConnectionError("prefect is down")


def test_a_transport_failure_raises_event_publish_error(monkeypatch):
    monkeypatch.setattr(event_publisher.requests, "post", _post_raising(CONNECTION_ERROR))

    with pytest.raises(EventPublishError, match="Could not publish event"):
        publish_new_data_event(schema="new_predict_data")


def test_the_error_names_the_endpoint_it_tried(monkeypatch):
    monkeypatch.setattr(event_publisher.requests, "post", _post_raising(CONNECTION_ERROR))

    with pytest.raises(EventPublishError) as error:
        publish_new_data_event(schema="new_predict_data")

    assert event_publisher.PREFECT_API_URL in str(error.value)


def test_the_error_includes_the_response_body(monkeypatch):
    """
    A 422 from the event intake means a field was rejected. Without the body the
    message says nothing about which one, which is how a one-line schema mismatch
    turns into a debugging session.
    """
    http_error = requests.exceptions.HTTPError("422 Client Error")
    http_error.response = FakeResponse(text='{"detail":"occurred: field required"}')

    monkeypatch.setattr(event_publisher.requests, "post", _post_raising(http_error))

    with pytest.raises(EventPublishError, match="occurred: field required"):
        publish_new_data_event(schema="new_predict_data")


def test_the_response_body_is_truncated(monkeypatch):
    """A giant HTML error page must not be pasted whole into the API's 503 detail."""
    http_error = requests.exceptions.HTTPError("500 Server Error")
    http_error.response = FakeResponse(text="x" * 5000)

    monkeypatch.setattr(event_publisher.requests, "post", _post_raising(http_error))

    with pytest.raises(EventPublishError) as error:
        publish_new_data_event(schema="new_predict_data")

    assert str(error.value).count("x") == 500
