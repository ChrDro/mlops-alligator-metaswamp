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
from fastapi import FastAPI, HTTPException
from predict import predict
from prometheus_fastapi_instrumentator import Instrumentator


# Inside Docker Compose this points to the Evidently service name.
# It is configurable so you can run the API against another monitoring
# endpoint without changing application code.
# MONITORING_URL = os.getenv(
#     "MONITORING_URL", "http://evidently_service:8085/iterate/green_taxi_data"
# )

app = FastAPI()

# Expose default FastAPI request metrics on /metrics for Prometheus.
Instrumentator().instrument(app).expose(app)


@app.get("/")
def index() -> dict:
    # Keep the root route simple so users can tell the container is alive
    # before testing the full prediction path.
    return {"message": "PK, FK Candidate and Normalform Prediction"}



@app.post("/predict_pk", response_model=PrimaryKeyPrediction)
def predict_primary_key(data: PrimaryKey) -> PrimaryKeyPrediction:
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

        return PrimaryKeyPrediction(
            **data_dict,
            prediction=prediction_value,
            probability=probability,
        )

    except HTTPException:
        raise
    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=f"Error in model input or predict logic: {error!s}",
        ) from error


@app.post("/predict_cpk", response_model=CompositePrimaryKeyPrediction)
def predict_composite_primary_key(data: CompositePrimaryKey) -> CompositePrimaryKeyPrediction:
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

        return CompositePrimaryKeyPrediction(
            **data_dict,
            prediction=prediction_value,
            probability=probability,
        )

    except HTTPException:
        raise
    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=f"Error in model input or predict logic: {error!s}",
        ) from error


@app.post("/predict_fk", response_model=ForeignKeyPrediction)
def predict_foreign_key(data: ForeignKey) -> ForeignKeyPrediction:
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

        return ForeignKeyPrediction(
            **data_dict,
            prediction=prediction_value,
            probability=probability,
        )

    except HTTPException:
        raise
    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=f"Error in model input or predict logic: {error!s}",
        ) from error


@app.post("/predict_cfk", response_model=CompositeForeignKeyPrediction)
def predict_composite_foreign_key(data: CompositeForeignKey) -> CompositeForeignKeyPrediction:
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

        return CompositeForeignKeyPrediction(
            **data_dict,
            prediction=prediction_value,
            probability=probability,
        )

    except HTTPException:
        raise
    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=f"Error in model input or predict logic: {error!s}",
        ) from error


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
        detail="Change detection triggered.",
    )


@app.post("/predict_normalform", response_model=NormalFormPrediction)
def predict_normalform_key_candidate(data: NormalForm) -> NormalFormPrediction:
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

        return NormalFormPrediction(
            **data_dict,
            prediction=prediction_value,
            probability=probability,
        )

    except HTTPException:
        raise
    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=f"Error in model input or predict logic: {error!s}",
        ) from error
