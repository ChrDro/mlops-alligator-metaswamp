"""
Endpoint tests for webservice/app.py via FastAPI's TestClient.

`predict` and `forward_to_monitoring` are replaced in app.py's namespace, so no
MLflow and no Evidently service are involved - everything else (routing, request
validation, response models, error mapping, background tasks) is the real code.

The five predict handlers are near-identical copies that differ only in model name
and monitoring track (open item 3.1: "pull onto a shared helper"). Until that
refactor happens, a copy-paste slip is the most likely defect in this file, so the
tests below assert per route *which* model was called and *which* track received the
row - not merely that a 200 came back.
"""

import app as app_module
import numpy as np
import pytest
from data_model_cfk import CompositeForeignKey
from data_model_cpk import CompositePrimaryKey
from data_model_denormalization import NormalForm
from data_model_fk import ForeignKey
from data_model_pk import PrimaryKey
from event_publisher import EventPublishError
from fastapi.testclient import TestClient

from .conftest import payload_for


# (route, request model, registered model name, monitoring track)
ROUTES = [
    ("/predict_pk", PrimaryKey, "pk_model", "pk_columns"),
    ("/predict_cpk", CompositePrimaryKey, "composite_pk_model", "cpk_columns"),
    ("/predict_fk", ForeignKey, "fk_model", "fk_columns"),
    ("/predict_cfk", CompositeForeignKey, "composite_fk_model", "cfk_columns"),
    ("/predict_normalform", NormalForm, "denormalization_model", "nf_columns"),
]
ROUTE_IDS = [route[0] for route in ROUTES]

PREDICTED_CLASS = 1
PREDICTED_PROBABILITY = 0.87


class Recorder:
    """Collects what the handlers passed to their collaborators."""

    def __init__(self) -> None:
        self.predict_calls: list[tuple[str, list[str]]] = []
        self.monitoring_calls: list[tuple[str, dict]] = []


@pytest.fixture
def api(monkeypatch):
    """A TestClient whose model call and monitoring call are recorded, not performed."""
    recorder = Recorder()

    def fake_predict(model_name, frame):
        recorder.predict_calls.append((model_name, list(frame.columns)))
        return float(PREDICTED_CLASS), PREDICTED_PROBABILITY

    def fake_forward(payload, track):
        recorder.monitoring_calls.append((track, payload))

    monkeypatch.setattr(app_module, "predict", fake_predict)
    monkeypatch.setattr(app_module, "forward_to_monitoring", fake_forward)

    with TestClient(app_module.app) as client:
        yield client, recorder


# --- predict routes -----------------------------------------------------------


@pytest.mark.parametrize(("route", "model_cls", "_model_name", "_track"), ROUTES, ids=ROUTE_IDS)
def test_predict_route_returns_prediction_and_probability(
    api, route, model_cls, _model_name, _track
):
    client, _ = api

    response = client.post(route, json=payload_for(model_cls))

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == PREDICTED_CLASS
    assert body["probability"] == pytest.approx(PREDICTED_PROBABILITY)


@pytest.mark.parametrize(("route", "model_cls", "_model_name", "_track"), ROUTES, ids=ROUTE_IDS)
def test_predict_route_echoes_every_request_feature(api, route, model_cls, _model_name, _track):
    """The response model extends the request model, so nothing may be dropped."""
    client, _ = api
    payload = payload_for(model_cls)

    body = client.post(route, json=payload).json()

    for field, value in payload.items():
        assert field in body, f"{route} dropped the request feature {field!r}"
        assert body[field] == value


@pytest.mark.parametrize(("route", "model_cls", "model_name", "_track"), ROUTES, ids=ROUTE_IDS)
def test_predict_route_calls_its_own_registered_model(api, route, model_cls, model_name, _track):
    client, recorder = api

    client.post(route, json=payload_for(model_cls))

    assert [name for name, _columns in recorder.predict_calls] == [model_name]


@pytest.mark.parametrize(("route", "model_cls", "_model_name", "_track"), ROUTES, ids=ROUTE_IDS)
def test_predict_route_sends_all_declared_features_to_the_model(
    api,
    route,
    model_cls,
    _model_name,
    _track,
):
    client, recorder = api

    client.post(route, json=payload_for(model_cls))

    _name, columns = recorder.predict_calls[0]
    assert columns == list(model_cls.model_fields)


@pytest.mark.parametrize(("route", "model_cls", "_model_name", "track"), ROUTES, ids=ROUTE_IDS)
def test_predict_route_forwards_to_its_own_monitoring_track(
    api, route, model_cls, _model_name, track
):
    """
    Each model has its own feature schema and therefore its own Evidently track.
    Posting one model's payload to another's track fails inside the monitor with
    "partially present in data", which is only visible in the drift dashboards - so
    the wiring is asserted here instead.
    """
    client, recorder = api

    client.post(route, json=payload_for(model_cls))

    assert len(recorder.monitoring_calls) == 1
    forwarded_track, forwarded_payload = recorder.monitoring_calls[0]
    assert forwarded_track == track
    # The monitor receives the full response, features plus model output.
    assert forwarded_payload["prediction"] == PREDICTED_CLASS
    assert set(model_cls.model_fields) <= set(forwarded_payload)


@pytest.mark.parametrize(("route", "model_cls", "_model_name", "_track"), ROUTES, ids=ROUTE_IDS)
def test_predict_route_maps_a_model_failure_to_400(
    monkeypatch, route, model_cls, _model_name, _track
):
    """A schema-drift ValueError from predict() must surface as a 400, not a 500."""

    def failing_predict(_model_name_arg, _frame):
        message = "Feature mismatch for 'pk_model': Missing: ['x']"
        raise ValueError(message)

    monkeypatch.setattr(app_module, "predict", failing_predict)
    monkeypatch.setattr(app_module, "forward_to_monitoring", lambda *_args: None)

    with TestClient(app_module.app) as client:
        response = client.post(route, json=payload_for(model_cls))

    assert response.status_code == 400
    assert "Feature mismatch" in response.json()["detail"]


@pytest.mark.parametrize(
    ("raw_prediction", "expected"),
    [
        (np.int64(2), 2),
        (np.float64(3.0), 3),
        ([1], 1),
        ((0,), 0),
        (np.array([2]), 2),
        (1.0, 1),
    ],
    ids=["numpy-int", "numpy-float", "list", "tuple", "numpy-array", "python-float"],
)
@pytest.mark.parametrize(("route", "model_cls", "_model_name", "_track"), ROUTES, ids=ROUTE_IDS)
def test_predict_route_normalises_every_prediction_shape(
    monkeypatch,
    route,
    model_cls,
    _model_name,
    _track,
    raw_prediction,
    expected,
):
    """
    Each handler carries the same three-branch ladder (`.item()`, sequence, plain cast)
    to turn whatever the model returned into an int. `predict()` currently hands back a
    Python float, so the other branches are dormant - but they are duplicated five
    times, and a response_model needs an int either way.

    Parametrised over all five routes because the ladder is copy-pasted: this is what
    would catch one handler being edited and the other four left behind.
    """

    def fake_predict(_model_name_arg, _frame):
        return raw_prediction, PREDICTED_PROBABILITY

    monkeypatch.setattr(app_module, "predict", fake_predict)
    monkeypatch.setattr(app_module, "forward_to_monitoring", lambda *_args: None)

    with TestClient(app_module.app) as client:
        response = client.post(route, json=payload_for(model_cls))

    assert response.status_code == 200
    assert response.json()["prediction"] == expected


@pytest.mark.parametrize(("route", "model_cls", "_model_name", "_track"), ROUTES, ids=ROUTE_IDS)
def test_predict_route_rejects_an_incomplete_body(api, route, model_cls, _model_name, _track):
    client, recorder = api

    response = client.post(route, json={})

    assert response.status_code == 422
    # Validation must happen before the model is touched.
    assert recorder.predict_calls == []


@pytest.mark.parametrize(("route", "model_cls", "_model_name", "_track"), ROUTES, ids=ROUTE_IDS)
def test_predict_route_rejects_an_uncoercible_value(api, route, model_cls, _model_name, _track):
    client, recorder = api
    payload = payload_for(model_cls)
    payload[next(iter(payload))] = "not-a-number"

    response = client.post(route, json=payload)

    assert response.status_code == 422
    assert recorder.predict_calls == []


# --- service routes -----------------------------------------------------------


def test_root_route_reports_the_service_is_alive(api):
    client, _ = api

    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_metrics_route_exposes_the_families_grafana_queries(api):
    """
    The Golden Signals dashboard and prometheus/alert.yaml query these families by
    name. An instrumentator upgrade that renames them would leave every panel blank
    and every alert permanently inactive - silent failures both.
    """
    client, _ = api
    client.get("/")  # label-bearing families only appear after one observed request

    body = client.get("/metrics").text

    for family in (
        "http_requests_total",
        "http_request_duration_seconds_bucket",
        "http_request_duration_highr_seconds_bucket",
    ):
        assert family in body, f"{family} is queried by Grafana/alerts but not exposed"


# --- streaming trigger --------------------------------------------------------


def test_new_data_event_accepts_and_publishes(monkeypatch):
    published = []
    monkeypatch.setattr(
        app_module,
        "publish_new_data_event",
        lambda schema, note: published.append((schema, note)),
    )

    with TestClient(app_module.app) as client:
        response = client.post("/events/new-data", json={"schema": "new_predict_data"})

    assert response.status_code == 202
    assert published == [("new_predict_data", None)]
    body = response.json()
    assert body["status"] == "accepted"
    # 'schema' is reserved on BaseModel, so the field is schema_name internally while
    # the JSON key stays 'schema' via serialization_alias.
    assert body["schema"] == "new_predict_data"


def test_new_data_event_defaults_the_schema(monkeypatch):
    published = []
    monkeypatch.setattr(
        app_module,
        "publish_new_data_event",
        lambda schema, note: published.append((schema, note)),
    )

    with TestClient(app_module.app) as client:
        response = client.post("/events/new-data", json={"note": "nightly load finished"})

    assert response.status_code == 202
    assert published == [("new_predict_data", "nightly load finished")]


def test_new_data_event_returns_503_and_points_at_the_cron_fallback(monkeypatch):
    """
    A lost event is not lost data: the change detector also runs on a cron schedule.
    The 503 says "retry if you like", and the detail has to say why that is optional -
    otherwise a caller escalates a non-incident.
    """

    def failing_publish(schema, note):
        message = "Could not publish event to http://prefect:4200/api"
        raise EventPublishError(message)

    monkeypatch.setattr(app_module, "publish_new_data_event", failing_publish)

    with TestClient(app_module.app) as client:
        response = client.post("/events/new-data", json={"schema": "new_predict_data"})

    assert response.status_code == 503
    assert "scheduled" in response.json()["detail"]
