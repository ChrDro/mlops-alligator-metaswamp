"""
Publish "new data arrived" events to the Prefect API.

Why raw HTTP instead of ``prefect.events.emit_event``
-----------------------------------------------------
The model service is built from its own Docker context (``webservice/``) and
deliberately stays small: it loads a model and serves predictions. Pulling the
whole Prefect SDK into that image just to send one POST would roughly double it
and couple the serving path to the orchestrator's release cycle. Prefect's event
intake is a plain REST endpoint (``POST /api/events``, 204 on success), so
``requests`` - already a dependency - is enough.

The event carries no list of changed tables on purpose. It is only a hint meaning
"look now"; the change detector flow decides what actually changed by diffing
watermarks. That keeps the detection logic in exactly one place. See
``prefect/change_events.py`` for the other side of this contract.
"""

import os
import uuid
from datetime import UTC, datetime

import requests


# Inside the compose network the Prefect server is reachable by service name.
PREFECT_API_URL = os.getenv("PREFECT_API_URL", "http://prefect:4200/api")

# Must match NEW_DATA_EVENT / EVENT_RESOURCE_ID in prefect/change_events.py.
# Duplicated rather than imported: the webservice image cannot see that module,
# because it is built from the webservice/ directory alone.
NEW_DATA_EVENT = "alligator.new-data.arrived"
EVENT_RESOURCE_ID = "alligator.metaswamp/new-predict-data"

EVENT_TIMEOUT_SECONDS = 5


class EventPublishError(RuntimeError):
    """Raised when the Prefect API could not accept the event."""


def publish_new_data_event(schema: str, note: str | None = None) -> None:
    """
    Tell Prefect that ``schema`` may contain new data.

    Args:
        schema: The source schema that was loaded, e.g. ``new_predict_data``.
        note: Free-text hint from the caller, kept for traceability only.

    Raises:
        EventPublishError: The Prefect API was unreachable or rejected the event.
    """
    event = {
        "event": NEW_DATA_EVENT,
        "resource": {
            "prefect.resource.id": EVENT_RESOURCE_ID,
            "alligator.schema": schema,
        },
        "payload": {"schema": schema, "note": note or "", "source": "webhook"},
        # The Prefect *client* fills these in via default_factory, but the REST
        # endpoint does not: on the 3.6.x server both are required and their absence
        # is a 422. Since we post raw JSON without the SDK, we supply them ourselves.
        "id": str(uuid.uuid4()),
        "occurred": datetime.now(UTC).isoformat(),
    }

    try:
        response = requests.post(
            f"{PREFECT_API_URL}/events",
            json=[event],
            timeout=EVENT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as error:
        # Include the response body: a bare "422 Unprocessable Entity" says nothing
        # about which field the server rejected, which turns a one-line schema
        # mismatch into a debugging session.
        body = ""
        if error.response is not None:
            body = f" - response: {error.response.text[:500]}"
        msg = f"Could not publish event to {PREFECT_API_URL}: {error}{body}"
        raise EventPublishError(msg) from error
