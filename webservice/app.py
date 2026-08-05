"""FastAPI entrypoint for the local prediction service.

This module owns the public API surface used in the monitoring tutorial:
- GET / for a simple health message
- GET /health/live and /health/ready as probes
- POST /predict for model inference plus monitoring side effects
- /metrics through prometheus-fastapi-instrumentator for service telemetry
"""

import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus

import pandas as pd
from data_model_cfk import CompositeForeignKey, CompositeForeignKeyPrediction
from data_model_cpk import CompositePrimaryKey, CompositePrimaryKeyPrediction
from data_model_denormalization import NormalForm, NormalFormPrediction
from data_model_events import NewDataAccepted, NewDataNotification
from data_model_fk import ForeignKey, ForeignKeyPrediction
from data_model_health import LivenessStatus, ReadinessStatus
from data_model_pk import PrimaryKey, PrimaryKeyPrediction
from data_model_subject_area import SubjectArea, SubjectAreaPrediction
from event_publisher import EventPublishError, publish_new_data_event
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from metrics import record_error, record_offline_quality
from monitoring_client import forward_to_monitoring
from predict import (
    MODEL_ALIAS,
    predict,
    predict_domain,
    read_offline_quality,
    resolve_model_versions,
)
from prometheus_fastapi_instrumentator import Instrumentator


def publish_offline_quality() -> None:
    """Export the offline F1 of each served model version as a Prometheus gauge.

    Startup is the right moment and the only one needed: the served model is pinned by
    lru_cache until the process restarts, so its offline score cannot change while the
    process lives. Restarting after a retrain - already the documented step, because the
    cache would otherwise keep serving the old model - is what refreshes this too.

    Failures are swallowed deliberately. This is provenance for a dashboard, not part of
    serving, and MLflow being briefly unreachable must not stop the container from coming
    up and answering with models it may already have cached.
    """
    try:
        for model_name, quality in read_offline_quality().items():
            record_offline_quality(
                model=model_name,
                f1=quality["f1"],
                estimator=quality["estimator"],
                version=quality["version"],
                f1_std=quality["f1_std"],
            )
    except Exception:  # noqa: BLE001 - a metrics export may never block startup
        traceback.print_exc()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hook. `on_event` is deprecated in this FastAPI version."""
    publish_offline_quality()
    yield


app = FastAPI(lifespan=lifespan)

# Expose default FastAPI request metrics on /metrics for Prometheus.
Instrumentator().instrument(app).expose(app)


@app.get("/")
def index() -> dict:
    # Keep the root route simple so users can tell the container is alive
    # before testing the full prediction path.
    return {"message": "PK, FK Candidate, Normalform and Subject Area Prediction"}


@app.get("/health/live", response_model=LivenessStatus)
def health_live() -> LivenessStatus:
    """
    Liveness: the process is up and answering HTTP.

    Touches nothing external on purpose. A liveness probe that failed while MLflow
    was down would restart a container that is perfectly capable of serving the
    models it has already cached, and throw that cache away for nothing.
    """
    return LivenessStatus()


@app.get("/health/ready", response_model=ReadinessStatus)
def health_ready(response: Response) -> ReadinessStatus:
    """
    Readiness: whether the registry currently resolves the models to be served.

    Returns 200 while at least one model resolves - with `status` distinguishing
    `ok` from `degraded`, because a missing alias on one of the five still leaves
    the other four servable. Only a registry that resolves nothing at all is a
    503, which is the point at which sending traffic here is pointless.
    """
    versions = resolve_model_versions()
    unresolved = sorted(name for name, version in versions.items() if version is None)
    resolved_count = len(versions) - len(unresolved)

    if not unresolved:
        readiness = "ok"
        detail = f"All {len(versions)} models resolve for alias '{MODEL_ALIAS}'."
    elif resolved_count:
        readiness = "degraded"
        detail = (
            f"{resolved_count} of {len(versions)} models resolve for alias "
            f"'{MODEL_ALIAS}'. Not resolving: {', '.join(unresolved)}."
        )
    else:
        readiness = "unavailable"
        detail = (
            f"No model resolves for alias '{MODEL_ALIAS}'. Either the registry is "
            f"unreachable or nothing has been registered yet - run the training "
            f"scripts, or ./scripts/setup_stack.sh."
        )
        response.status_code = HTTPStatus.SERVICE_UNAVAILABLE

    return ReadinessStatus(
        status=readiness,
        alias=MODEL_ALIAS,
        models=versions,
        detail=detail,
    )


@app.post("/predict_pk", response_model=PrimaryKeyPrediction)
def predict_primary_key(
    data: PrimaryKey,
    background_tasks: BackgroundTasks,
) -> PrimaryKeyPrediction:
    try:
        data_dict = data.model_dump()  # Nutze .dict() bei Pydantic v1

        input_df = pd.DataFrame([data_dict])

        print("Sending the following columns as features to the model:", input_df.columns.tolist())

        prediction, probability = predict("pk_model", input_df)

        if hasattr(prediction, "item"):
            prediction_value = int(prediction.item())
        elif isinstance(prediction, (list, tuple)) or hasattr(prediction, "__len__"):
            prediction_value = int(prediction[0])
        else:
            prediction_value = int(prediction)

        response = PrimaryKeyPrediction(
            **data_dict,
            prediction=prediction_value,
            probability=probability,
        )

        # Forward to the drift monitor after the response is sent. Each model has
        # its own track because each has its own feature schema; see `models` in
        # evidently_service/config.yaml.
        #
        # The drift track needs no ground truth, which is why it can run from the
        # serving path at all. Classification quality (F1/precision/recall) needs
        # the target column, which does not exist at prediction time - that track is
        # driven by prefect/model_quality_backtest.py instead.
        background_tasks.add_task(forward_to_monitoring, response.model_dump(), "pk_columns")

    except HTTPException:
        raise
    except Exception as error:
        record_error("pk_model", error)
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=f"Error in model input or predict logic: {error!s}",
        ) from error
    else:
        return response


@app.post("/predict_cpk", response_model=CompositePrimaryKeyPrediction)
def predict_composite_primary_key(
    data: CompositePrimaryKey,
    background_tasks: BackgroundTasks,
) -> CompositePrimaryKeyPrediction:
    try:
        data_dict = data.model_dump()  # Nutze .dict() bei Pydantic v1

        input_df = pd.DataFrame([data_dict])

        print("Sending the following columns as features to the model:", input_df.columns.tolist())

        prediction, probability = predict("composite_pk_model", input_df)

        if hasattr(prediction, "item"):
            prediction_value = int(prediction.item())
        elif isinstance(prediction, (list, tuple)) or hasattr(prediction, "__len__"):
            prediction_value = int(prediction[0])
        else:
            prediction_value = int(prediction)

        response = CompositePrimaryKeyPrediction(
            **data_dict,
            prediction=prediction_value,
            probability=probability,
        )

        background_tasks.add_task(forward_to_monitoring, response.model_dump(), "cpk_columns")

    except HTTPException:
        raise
    except Exception as error:
        record_error("composite_pk_model", error)
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=f"Error in model input or predict logic: {error!s}",
        ) from error
    else:
        return response


@app.post("/predict_fk", response_model=ForeignKeyPrediction)
def predict_foreign_key(
    data: ForeignKey,
    background_tasks: BackgroundTasks,
) -> ForeignKeyPrediction:
    try:
        data_dict = data.model_dump()  # Nutze .dict() bei Pydantic v1

        input_df = pd.DataFrame([data_dict])

        print("Sending the following columns as features to the model:", input_df.columns.tolist())

        prediction, probability = predict("fk_model", input_df)

        if hasattr(prediction, "item"):
            prediction_value = int(prediction.item())
        elif isinstance(prediction, (list, tuple)) or hasattr(prediction, "__len__"):
            prediction_value = int(prediction[0])
        else:
            prediction_value = int(prediction)

        response = ForeignKeyPrediction(
            **data_dict,
            prediction=prediction_value,
            probability=probability,
        )

        background_tasks.add_task(forward_to_monitoring, response.model_dump(), "fk_columns")

    except HTTPException:
        raise
    except Exception as error:
        record_error("fk_model", error)
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=f"Error in model input or predict logic: {error!s}",
        ) from error
    else:
        return response


@app.post("/predict_cfk", response_model=CompositeForeignKeyPrediction)
def predict_composite_foreign_key(
    data: CompositeForeignKey,
    background_tasks: BackgroundTasks,
) -> CompositeForeignKeyPrediction:
    try:
        data_dict = data.model_dump()  # Nutze .dict() bei Pydantic v1

        input_df = pd.DataFrame([data_dict])

        print("Sending the following columns as features to the model:", input_df.columns.tolist())

        prediction, probability = predict("composite_fk_model", input_df)

        if hasattr(prediction, "item"):
            prediction_value = int(prediction.item())
        elif isinstance(prediction, (list, tuple)) or hasattr(prediction, "__len__"):
            prediction_value = int(prediction[0])
        else:
            prediction_value = int(prediction)

        response = CompositeForeignKeyPrediction(
            **data_dict,
            prediction=prediction_value,
            probability=probability,
        )

        background_tasks.add_task(forward_to_monitoring, response.model_dump(), "cfk_columns")

    except HTTPException:
        raise
    except Exception as error:
        record_error("composite_fk_model", error)
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=f"Error in model input or predict logic: {error!s}",
        ) from error
    else:
        return response


@app.post("/predict_subject_area", response_model=SubjectAreaPrediction)
def predict_subject_area(data: SubjectArea) -> SubjectAreaPrediction:
    """
    Assign a table to a subject area (its business domain).

    The label set is not fixed in code: task_4 discovers it by clustering table names
    and column names, names each cluster with a local LLM, then distils the result
    into the TF-IDF classifier served here. Retraining can therefore both rename an
    area and introduce a new one.

    `prediction` is that name, so unlike the other endpoints it is a string. A table
    that belongs to no discovered area still gets the closest one - the signal for
    "do not trust this" is a low `probability`, not a special label.
    """
    try:
        data_dict = data.model_dump()

        input_df = pd.DataFrame([data_dict])

        print("Sending the following columns as features to the model:", input_df.columns.tolist())

        prediction_value, probability = predict_domain("subject_area_model", input_df)

        response = SubjectAreaPrediction(
            **data_dict,
            prediction=prediction_value,
            probability=probability,
        )

        # No monitoring track yet: the Evidently references are built from numeric
        # feature frames, and two free-text columns need a DataDefinition of their own
        # before a drift report on them means anything.

    except HTTPException:
        raise
    except Exception as error:
        record_error("subject_area_model", error)
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=f"Error in model input or predict logic: {error!s}",
        ) from error
    else:
        return response


@app.post("/events/new-data", response_model=NewDataAccepted, status_code=202)
def notify_new_data(notification: NewDataNotification) -> NewDataAccepted:
    """
    Push trigger for the streaming pipeline.

    A loader calls this right after writing to the source schema. The service only
    forwards a "look now" event to Prefect and returns immediately - it does not
    predict anything here. An automation starts the change detector, which diffs
    watermarks, records what changed and triggers both prediction pipelines.

    Missing this call is not fatal: the same detector also runs on a cron schedule
    and will pick the change up on its next pass. That is why a failure to publish
    returns 503 (retry if you like) rather than losing data.
    """
    try:
        publish_new_data_event(schema=notification.schema_name, note=notification.note)
    except EventPublishError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{error} - the change will still be picked up by the scheduled "
                f"change-detection run."
            ),
        ) from error

    return NewDataAccepted(
        status="accepted",
        schema_name=notification.schema_name,
        detail="Predict Pipeline triggered..",
    )


@app.post("/predict_normalform", response_model=NormalFormPrediction)
def predict_normalform_key_candidate(
    data: NormalForm,
    background_tasks: BackgroundTasks,
) -> NormalFormPrediction:
    try:
        data_dict = data.model_dump()  # Nutze .dict() bei Pydantic v1

        input_df = pd.DataFrame([data_dict])

        print("Sending the following columns as features to the model:", input_df.columns.tolist())

        prediction, probability = predict("denormalization_model", input_df)

        if hasattr(prediction, "item"):
            prediction_value = int(prediction.item())
        elif isinstance(prediction, (list, tuple)) or hasattr(prediction, "__len__"):
            prediction_value = int(prediction[0])
        else:
            prediction_value = int(prediction)

        response = NormalFormPrediction(
            **data_dict,
            prediction=prediction_value,
            probability=probability,
        )

        # Drift only. This model is multiclass, so its classification track cannot
        # use log loss - but drift needs no labels and no probabilities, so it works
        # here exactly as for the binary models.
        background_tasks.add_task(forward_to_monitoring, response.model_dump(), "nf_columns")

    except HTTPException:
        raise
    except Exception as error:
        record_error("denormalization_model", error)
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=f"Error in model input or predict logic: {error!s}",
        ) from error
    else:
        return response
