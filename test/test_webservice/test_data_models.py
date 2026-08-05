"""
Validation tests for the Pydantic request/response models.

These models are the API's contract in both directions: they reject malformed input
before it reaches a model, and they define the payload the Evidently drift tracks
receive. Whether the declared *features* match the trained models is a separate
question, answered against the live registry by
test/test_models/test_model_schema_contract.py - these tests only need the schemas
themselves and run with no stack.
"""

import pytest
from data_model_cfk import CompositeForeignKey, CompositeForeignKeyPrediction
from data_model_cpk import CompositePrimaryKey, CompositePrimaryKeyPrediction
from data_model_denormalization import NormalForm, NormalFormPrediction
from data_model_events import NewDataAccepted, NewDataNotification
from data_model_fk import ForeignKey, ForeignKeyPrediction
from data_model_pk import PrimaryKey, PrimaryKeyPrediction
from pydantic import ValidationError

from .conftest import payload_for


# (request model, matching response model)
MODEL_PAIRS = [
    (PrimaryKey, PrimaryKeyPrediction),
    (CompositePrimaryKey, CompositePrimaryKeyPrediction),
    (ForeignKey, ForeignKeyPrediction),
    (CompositeForeignKey, CompositeForeignKeyPrediction),
    (NormalForm, NormalFormPrediction),
]
PAIR_IDS = [request_cls.__name__ for request_cls, _ in MODEL_PAIRS]

REQUEST_MODELS = [pair[0] for pair in MODEL_PAIRS]
REQUEST_IDS = [cls.__name__ for cls in REQUEST_MODELS]

# Fields each response model appends after the request fields. The four binary models
# only ever need a class and its confidence; the normalform model is multiclass and
# its table-level aggregation needs the full per-class distribution (see
# aggregate_to_table's soft vote in prefect/normalform_pipeline.py), so its response
# carries one extra field.
DEFAULT_OUTPUT_FIELDS = ["prediction", "probability"]
OUTPUT_FIELDS = {
    NormalFormPrediction: [*DEFAULT_OUTPUT_FIELDS, "probabilities"],
}


@pytest.mark.parametrize("model_cls", REQUEST_MODELS, ids=REQUEST_IDS)
def test_request_model_accepts_a_complete_payload(model_cls):
    instance = model_cls(**payload_for(model_cls))

    assert set(instance.model_dump()) == set(model_cls.model_fields)


@pytest.mark.parametrize("model_cls", REQUEST_MODELS, ids=REQUEST_IDS)
def test_request_model_declares_every_field_as_required(model_cls):
    """
    No feature may carry a default. A silent default would let an incomplete request
    through and be scored as if it were a real measurement - worse than a 422,
    because the wrong answer looks like a valid one.
    """
    optional = [name for name, field in model_cls.model_fields.items() if not field.is_required()]

    assert optional == []


@pytest.mark.parametrize("model_cls", REQUEST_MODELS, ids=REQUEST_IDS)
def test_request_model_rejects_a_missing_feature(model_cls):
    payload = payload_for(model_cls)
    dropped = next(iter(payload))
    del payload[dropped]

    with pytest.raises(ValidationError, match=dropped):
        model_cls(**payload)


@pytest.mark.parametrize("model_cls", REQUEST_MODELS, ids=REQUEST_IDS)
def test_request_model_rejects_an_uncoercible_value(model_cls):
    payload = payload_for(model_cls)
    payload[next(iter(payload))] = "not-a-number"

    with pytest.raises(ValidationError, match="not-a-number"):
        model_cls(**payload)


@pytest.mark.parametrize("model_cls", REQUEST_MODELS, ids=REQUEST_IDS)
def test_request_model_ignores_unknown_fields(model_cls):
    """
    Documents current behaviour: extras are dropped, not rejected. That is what lets
    the prediction pipeline post a row that still carries metadata columns. Switching
    to `extra="forbid"` would break prefect/pk_fk_pipeline.py's predict_batch.
    """
    instance = model_cls(**payload_for(model_cls), surplus_column="ignored")

    assert not hasattr(instance, "surplus_column")


@pytest.mark.parametrize(("request_cls", "response_cls"), MODEL_PAIRS, ids=PAIR_IDS)
def test_response_model_extends_the_request_model(request_cls, response_cls):
    """The response echoes every request feature and appends the model output."""
    assert issubclass(response_cls, request_cls)

    response_fields = list(response_cls.model_fields)
    expected_output_fields = OUTPUT_FIELDS.get(response_cls, DEFAULT_OUTPUT_FIELDS)
    assert response_fields == [*request_cls.model_fields, *expected_output_fields]


@pytest.mark.parametrize(("request_cls", "response_cls"), MODEL_PAIRS, ids=PAIR_IDS)
def test_response_model_requires_the_model_output(request_cls, response_cls):
    with pytest.raises(ValidationError, match="prediction"):
        response_cls(**payload_for(request_cls))


# --- streaming trigger models -------------------------------------------------


def test_notification_defaults_to_the_streaming_source_schema():
    notification = NewDataNotification()

    assert notification.schema_name == "new_predict_data"
    assert notification.note is None


def test_notification_accepts_the_json_alias_and_the_field_name():
    """
    'schema' is reserved on BaseModel, so the field is schema_name internally while
    the JSON key stays 'schema'. populate_by_name keeps both working, which matters
    because curl_tests/ posts the alias and Python callers use the field name.
    """
    by_alias = NewDataNotification(schema="other_schema")
    by_name = NewDataNotification(schema_name="other_schema")

    assert by_alias.schema_name == by_name.schema_name == "other_schema"


def test_notification_serializes_back_to_the_json_alias():
    notification = NewDataNotification(schema_name="new_predict_data", note="load finished")

    assert notification.model_dump(by_alias=True)["schema"] == "new_predict_data"


def test_accepted_response_serializes_the_schema_alias():
    accepted = NewDataAccepted(
        status="accepted",
        schema_name="new_predict_data",
        detail="Predict Pipeline triggered..",
    )

    dumped = accepted.model_dump(by_alias=True)
    assert dumped == {
        "status": "accepted",
        "schema": "new_predict_data",
        "detail": "Predict Pipeline triggered..",
    }
