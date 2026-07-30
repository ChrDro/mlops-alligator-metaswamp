"""FastAPI entrypoint for the local prediction service.

This module owns the public API surface used in the monitoring tutorial:
- GET / for a simple health message
- POST /predict for model inference plus monitoring side effects
- /metrics through prometheus-fastapi-instrumentator for service telemetry
"""

import traceback

import pandas as pd
from data_model_cfk import CompositeForeignKey, CompositeForeignKeyPrediction
from data_model_cpk import CompositePrimaryKey, CompositePrimaryKeyPrediction
from data_model_denormalization import NormalForm, NormalFormPrediction
from data_model_events import NewDataAccepted, NewDataNotification
from data_model_fk import ForeignKey, ForeignKeyPrediction
from data_model_pk import PrimaryKey, PrimaryKeyPrediction
from event_publisher import EventPublishError, publish_new_data_event
from fastapi import BackgroundTasks, FastAPI, HTTPException
from metrics import record_error
from monitoring_client import forward_to_monitoring
from predict import predict
from prometheus_fastapi_instrumentator import Instrumentator


app = FastAPI()

# Expose default FastAPI request metrics on /metrics for Prometheus.
Instrumentator().instrument(app).expose(app)


@app.get("/")
def index() -> dict:
    # Keep the root route simple so users can tell the container is alive
    # before testing the full prediction path.
    return {"message": "PK, FK Candidate and Normalform Prediction"}


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
