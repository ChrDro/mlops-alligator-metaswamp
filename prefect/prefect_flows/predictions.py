# prefect_flows/polling_predictions.py
import os
import time
from datetime import UTC, datetime

import pandas as pd
import requests
import urllib3
from dotenv import load_dotenv
from prefect.cache_policies import NONE as NO_CACHE
from sqlalchemy import Engine, create_engine
from sqlalchemy.sql.expression import text

from prefect import flow, task


# Suppress InsecureRequestWarning for unverified HTTPS requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

TRINO_IP_ADDRESS = str(os.environ.get("TRINO_IP_ADDRESS"))
TRINO_USERNAME = str(os.environ.get("TRINO_USERNAME"))
TRINO_PASSWORD = str(os.environ.get("TRINO_PASSWORD"))


def get_trino_engine() -> Engine:
    """
    Create and return a Trino engine instance.
    """
    return create_engine(
        f"trino://{TRINO_USERNAME}:{TRINO_PASSWORD}@{TRINO_IP_ADDRESS}:8443/duckdb",
        connect_args={
            "http_scheme": "https",
            "verify": False,
        },
    )


@task(cache_policy=NO_CACHE)
def fetch_new_rows(batch_size: int = 100) -> pd.DataFrame:
    """
    Fetch unprocessed rows from Trino.
    """
    trino_engine = get_trino_engine()

    query = text("""
        SELECT
            id,
            database,
            schema,
            table_name,
            column_name,
            number_unique_values,
            count,
            is_unique,
            ordinal_position,
            unique_ratio,
            is_first_column,
            relative_ordinal_position,
            is_first_unique_column,
            table_column_count,
            table_unique_column_count,
            table_row_count,
            table_has_unique_column,
            table_has_no_single_pk_candidate,
            table_near_unique_column_count,
            table_id_named_column_count,
            table_non_null_column_count,
            table_max_unique_ratio,
            unique_ratio_rank,
            null_ratio_rank,
            is_least_null_in_table,
            unique_ratio_relative_to_max,
            name_ends_with_id,
            name_contains_table_name,
            name_is_singular_table_id,
            name_length,
            column_type_boolean,
            column_type_date,
            column_type_decimal,
            column_type_double,
            column_type_integer,
            column_type_varchar
        FROM duckdb.predictions.queue
        WHERE processed = FALSE
        --ORDER BY created_at ASC
        LIMIT :batch_size
    """)

    with trino_engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"batch_size": batch_size})

    return df


@task(retries=2, retry_delay_seconds=10)
def predict_batch(rows: pd.DataFrame) -> list[dict]:
    """
    Call prediction endpoints for a batch of rows.
    """
    predictions = []

    for _idx, row in rows.iterrows():
        row_data = row.to_dict()
        row_predictions = {"queue_id": row["id"]}

        # Remove metadata columns that are selected but not needed for prediction
        metadata = {
            "id": row_data.pop("id", None),
            "database": row_data.pop("database", None),
            "schema": row_data.pop("schema", None),
            "table_name": row_data.pop("table_name", None),
            "column_name": row_data.pop("column_name", None),
        }

        # Track if all predictions succeed
        all_predictions_successful = True

        # Hint: Use localhost when running outside Docker, model-service when inside Docker
        for model_type in ["pk", "fk", "cpk", "cfk"]:
            try:
                url = f"http://localhost:8080/predict_{model_type}"

                response = requests.post(
                    url,
                    json=row_data,
                    timeout=30,
                )

                response.raise_for_status()
                result = response.json()

                # Debug: print the response
                # print(f"API Response for {model_type}: {result}")

                # Store prediction and probability (API returns 'probability' not 'confidence')
                prediction_value = result.get("prediction")
                row_predictions[f"{model_type}_prediction"] = prediction_value
                row_predictions[f"{model_type}_confidence"] = result.get("probability", 0.0)

                # Check if prediction is null or None
                if prediction_value is None:
                    all_predictions_successful = False
                    print(f"Warning: {model_type} prediction returned None for row {row['id']}")

            except (requests.RequestException, requests.Timeout, requests.HTTPError) as e:
                print(f"Error predicting {model_type} for row {row['id']}: {e}")
                print(f"Exception type: {type(e).__name__}")

                row_predictions[f"{model_type}_prediction"] = None
                row_predictions[f"{model_type}_confidence"] = 0.0
                all_predictions_successful = False

        # Add metadata in the desired order
        row_predictions.update(
            {
                "database": metadata["database"],
                "schema": metadata["schema"],
                "table_name": metadata["table_name"],
                "column_name": metadata["column_name"],
                "predicted_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
                "all_predictions_successful": all_predictions_successful,
            },
        )

        predictions.append(row_predictions)

    return predictions


@task(cache_policy=NO_CACHE)
def store_predictions_to_trino(predictions: list[dict]) -> None:
    """
    Store predictions and mark rows as processed ONLY if all predictions succeeded.
    """
    if not predictions:
        return

    trino_engine = get_trino_engine()

    # Log successful and failed predictions
    successful_predictions = []
    failed_predictions = []
    for p in predictions:
        if p.get("all_predictions_successful"):
            successful_predictions.append(p)

    for p in predictions:
        if "all_predictions_successful" and not p["all_predictions_successful"]:
            failed_predictions.append(p)

    # print failed ids for debugging
    if failed_predictions:
        print("Failed queue_ids:")
        for p in failed_predictions:
            print(f"  - {p['queue_id']}")

    # Remove all all_predictions_successful as it is not needed for storing the results
    df = pd.DataFrame(predictions)
    if "all_predictions_successful" in df.columns:
        df = df.drop(columns=["all_predictions_successful"])

    # Reorder columns for better visibility
    column_order = [
        "queue_id",
        "database",
        "schema",
        "table_name",
        "column_name",
        "pk_prediction",
        "pk_confidence",
        "fk_prediction",
        "fk_confidence",
        "cpk_prediction",
        "cpk_confidence",
        "cfk_prediction",
        "cfk_confidence",
        "predicted_at",
    ]

    valid_columns = []

    for col in column_order:
        if col in df.columns:
            valid_columns.append(col)

    df = df[valid_columns]

    # Convert queue_id to varchar to match table schema
    df["queue_id"] = df["queue_id"].astype(str)

    # Prediction columns should be varchar (convert None to empty string or 'NULL')
    prediction_cols = ["pk_prediction", "fk_prediction", "cpk_prediction", "cfk_prediction"]
    for col in prediction_cols:
        if col in df.columns:
            df[col] = df[col].fillna("NULL").astype(str)

    # Create Schema if it doesn't already exists
    with trino_engine.connect() as conn:
        create_schema = text("""
            CREATE SCHEMA IF NOT EXISTS duckdb.prediction_results
        """)

        conn.execute(create_schema)

        # inserting all predictions, even if they failed
        df.to_sql(
            "results",
            conn,
            schema="prediction_results",
            if_exists="append",
            index=False,
        )

        # Mark ONLY successful queue items as processed
        if successful_predictions:
            successful_queue_ids = []

            for p in successful_predictions:
                successful_queue_ids.append(p["queue_id"])

            # Build parameterized query with placeholders
            placeholder_list = []

            for i in range(len(successful_queue_ids)):
                placeholder_list.append(f":id_{i}")

            placeholders = ", ".join(placeholder_list)

            params = {}

            for i, qid in enumerate(successful_queue_ids):
                key = f"id_{i}"
                value = int(qid)
                params[key] = value

            params["current_timestamp"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

            # Using parameterized query to prevent SQL injection - placeholders are safe
            # ruff: noqa: S608
            query_str = f"""
                UPDATE duckdb.predictions.queue
                SET processed = TRUE,
                    processed_at = :current_timestamp
                WHERE id IN ({placeholders})
            """
            update_query = text(query_str)

            conn.execute(update_query, params)
            print(f"Marked {len(successful_predictions)} rows as processed in queue")
        else:
            print("No successful predictions to mark as processed")


@task
def log_metrics(num_processed: int, duration_seconds: float) -> None:
    """
    Log metrics to monitoring system (Prometheus/MLflow).
    """
    print(f"Processed {num_processed} predictions in {duration_seconds}s")


@flow(name="polling-predictions")
def prediction_polling_flow(batch_size: int = 100) -> None:
    """
    Main polling flow: fetch, predict, store.
    """
    start_time = time.time()

    # Test connection
    trino_engine = get_trino_engine()

    try:
        with trino_engine.connect() as connection:
            rows = connection.execute(text("Select 1"))
            print("Connection over HTTPS successful!")
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}")

    rows = fetch_new_rows(batch_size)

    if rows.empty:
        print("No new rows to process")
        return

    print(f"Fetched {len(rows)} rows to process")

    predictions = predict_batch(rows)

    store_predictions_to_trino(predictions)

    duration = time.time() - start_time
    log_metrics(len(predictions), duration)

    print(f"Successfully processed {len(predictions)} predictions")


if __name__ == "__main__":
    prediction_polling_flow()
