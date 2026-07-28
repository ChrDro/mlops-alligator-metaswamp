"""Fire-and-forget forwarding of predictions to the Evidently drift monitor.

Separate from event_publisher because the failure contract is the opposite one.
A failed Prefect event means work could be lost, so that path raises and the
caller returns 503. A failed monitoring post only costs one row of a drift
window, so this path swallows everything: monitoring must never be able to fail a
prediction.
"""

import logging
import os

import requests


# Inside Docker Compose this points to the Evidently service name. It is
# configurable so you can run the API against another monitoring endpoint without
# changing application code.
#
# Base URL only: the track is appended per endpoint, because each model has its own
# feature schema and therefore its own reference and DataDefinition in the Evidently
# service. Posting one model's payload to another's track fails with a
# "partially present in data" error.
MONITORING_BASE_URL = os.getenv(
    "MONITORING_BASE_URL",
    "http://evidently_service:8085",
).rstrip("/")

# Set MONITORING_ENABLED=false to stop forwarding without editing code - useful
# when running the API standalone with no Evidently service on the network.
MONITORING_ENABLED = os.getenv("MONITORING_ENABLED", "true").lower() not in {
    "false",
    "0",
    "no",
}

# Short by design. This runs in a background task after the response is sent, but a
# hung connection would still tie up a worker thread per prediction.
MONITORING_TIMEOUT_SECONDS = float(os.getenv("MONITORING_TIMEOUT_SECONDS", "2.0"))

logger = logging.getLogger(__name__)


def forward_to_monitoring(payload: dict, track: str) -> None:
    """
    Post one prediction to the Evidently drift endpoint, ignoring all failures.

    Intended to be handed to FastAPI's BackgroundTasks so the prediction response
    is already on the wire before this runs.

    Args:
        payload: The endpoint's response object - request features plus prediction
            and probability.
        track: The monitoring track for this model, matching a key under `models`
            in evidently_service/config.yaml.

    Every exception is logged and dropped. The drift window is a rolling sample -
    losing rows degrades its resolution slightly and nothing else - whereas letting
    an exception escape a background task would spam the logs with tracebacks for
    an entirely optional side effect.
    """
    if not MONITORING_ENABLED:
        return

    url = f"{MONITORING_BASE_URL}/iterate/{track}"
    try:
        response = requests.post(url, json=payload, timeout=MONITORING_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.RequestException as error:
        # debug, not warning: with monitoring down this would otherwise fire on
        # every single prediction and bury real errors.
        logger.debug(f"Could not forward prediction to {url}: {error}")
    except Exception:
        logger.exception("Unexpected error while forwarding a prediction to monitoring")
