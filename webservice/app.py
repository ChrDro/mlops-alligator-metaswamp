"""FastAPI entrypoint for the local prediction service.

This module owns the public API surface used in the monitoring tutorial:
- GET / for a simple health message
- POST /predict for model inference plus monitoring side effects
- /metrics through prometheus-fastapi-instrumentator for service telemetry
"""


import traceback

import pandas as pd
from data_model_fk import ForeignKey, ForeignKeyPrediction
from data_model_pk import PrimaryKey, PrimaryKeyPrediction
from fastapi import FastAPI, HTTPException

#from prometheus_fastapi_instrumentator import Instrumentator
from predict import predict


# Inside Docker Compose this points to the Evidently service name.
# It is configurable so you can run the API against another monitoring
# endpoint without changing application code.
# MONITORING_URL = os.getenv(
#     "MONITORING_URL", "http://evidently_service:8085/iterate/green_taxi_data"
# )

app = FastAPI()

# Expose default FastAPI request metrics on /metrics for Prometheus.
#Instrumentator().instrument(app).expose(app)


@app.get("/")
def index() -> dict:
    # Keep the root route simple so users can tell the container is alive
    # before testing the full prediction path.
    return {"message": "PK Candidate Prediction"}


# @app.post("/predict_pk", response_model=PrimaryKeyPrediction)
# def predict_pk_key_candidate(data: PrimaryKey):
#     # First serve the model prediction. Monitoring should observe this request,
#     # but it should not change the prediction result returned to the client.
#     prediction = predict("pk_model", data)
#     #prediction = predict("pk-key-candidate-suggestion", data)
#     try:
#         print(f"Sending data to metrics application: {data}")

#         # Evidently needs both the input features and the model output so it can
#         # compare live predictions against the reference distribution.
#         # monitoring_payload = PrimaryKeyPrediction(
#         #     **data.model_dump(), prediction=prediction
#         # ).model_dump()

#         # This POST is intentionally fire-and-forget from the API's point of
#         # view. If monitoring is slow or unavailable, the client should still
#         # receive the prediction response from the model service.
#         # requests.post(
#         #     MONITORING_URL,
#         #     json=monitoring_payload,
#         #     timeout=5,
#         # )
#     except requests.exceptions.ConnectionError as error:
#         print(f"Cannot reach a metrics application, error: {error}, data: {data}")
#     except requests.exceptions.Timeout as error:
#         print(f"Metrics application timed out, error: {error}, data: {data}")

#     # Return the same payload shape used for monitoring so users can
#     # compare what the client sees with what Evidently receives.
#     return PrimaryKeyPrediction(**data.model_dump(), prediction=prediction)

@app.post("/predict_pk", response_model=PrimaryKeyPrediction)
def predict_pk_key_candidate(data: PrimaryKey) -> PrimaryKeyPrediction:
    try:
        data_dict = data.model_dump() # Nutze .dict() bei Pydantic v1

        input_df = pd.DataFrame([data_dict])

        print("Sende folgende Spalten an das Modell:", input_df.columns.tolist())

        prediction, probability = predict("pk_model", input_df)

        if hasattr(prediction, "item"):
            prediction_value = int(prediction.item())
        elif isinstance(prediction, (list, tuple)) or hasattr(prediction, "__len__"):
            prediction_value = int(prediction[0])
        else:
            prediction_value = int(prediction)

        return PrimaryKeyPrediction(**data_dict,
                                    prediction=prediction_value,
                                    probability=probability,
                                   )

    except HTTPException:
        raise
    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=f"Fehler im Modell-Input oder der Predict-Logik: {error!s}",
        ) from error

@app.post("/predict_fk", response_model=ForeignKeyPrediction)
def predict_fk_key_candidate(data: ForeignKey) -> ForeignKeyPrediction:
    try:
        data_dict = data.model_dump() # Nutze .dict() bei Pydantic v1

        input_df = pd.DataFrame([data_dict])

        print("Sende folgende Spalten an das Modell:", input_df.columns.tolist())

        prediction, probability = predict("fk_model", input_df)

        if hasattr(prediction, "item"):
            prediction_value = int(prediction.item())
        elif isinstance(prediction, (list, tuple)) or hasattr(prediction, "__len__"):
            prediction_value = int(prediction[0])
        else:
            prediction_value = int(prediction)

        return ForeignKeyPrediction(**data_dict,
                                    prediction=prediction_value,
                                    probability=probability,
                                   )

    except HTTPException:
        raise
    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=f"Fehler im Modell-Input oder der Predict-Logik: {error!s}",
        ) from error
