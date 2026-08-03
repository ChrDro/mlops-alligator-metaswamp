"""
Prefect flow for the subject-area (domain grouping) prediction track.

Same three steps as the pk/fk pipeline, at TABLE grain instead of column grain:
1. Extracts the two subject-area features from information_schema (plain
   Python/pandas) and writes them to iceberg.staging.stg_subject_area in Trino.
2. Populates iceberg.predictions.subject_area_queue from the extracted features.
3. Calls POST /predict_subject_area per queued table, stores the answer in
   iceberg.prediction_results.subject_area_results and marks the queue row processed.

Its own staging, queue and results tables on purpose: the feature set here is two
text columns (`table_name` plus the table's comma-separated column names), which
shares nothing with the 43-column pk/fk queue or the 50-feature normalform staging
table.

Feature extraction is much cheaper than the other two tracks. Subject area is a
property of what a table is *called* and what it holds, so no value profiling is
needed - no COUNT(DISTINCT), no per-column batching, just one information_schema
query. The flip side is that the features only move when the SCHEMA moves: a reload
that adds rows but no columns produces the identical prediction. Such a table is
still re-predicted when the detector flags it (see `requeue_tables`), which keeps
the behaviour consistent with the other tracks and leaves a dated history row.
"""

import os
import time
from datetime import UTC, datetime

import pandas as pd
import requests
import urllib3
from change_events import (
    TRACK_SUBJECT_AREA,
    TableRef,
    claim_pending_changes,
    complete_pending_changes,
    release_pending_changes,
    with_commit_retry,
)
from dotenv import load_dotenv
from prefect.cache_policies import NONE as NO_CACHE
from prefect.runtime import flow_run
from sqlalchemy import Connection, Engine, create_engine, text

from prefect import flow, task


# Suppress InsecureRequestWarning for unverified HTTPS requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

TRINO_IP_ADDRESS = str(os.environ.get("TRINO_IP_ADDRESS"))
TRINO_USERNAME = str(os.environ.get("TRINO_USERNAME"))
TRINO_PASSWORD = str(os.environ.get("TRINO_PASSWORD"))

# Model API base (localhost outside Docker, model-service:8080 inside the compose network).
MODEL_API_URL = os.environ.get("MODEL_API_URL", "http://localhost:8080")

# The model's second feature is called `columns` in the SubjectArea request schema,
# but the staging/queue tables store it as `column_names`. `columns` reads as a
# keyword in enough SQL dialects to be worth avoiding for an unquoted identifier, and
# it is confusing next to information_schema.columns. _row_to_payload does the
# renaming, so the request still sends `columns` as the model expects.
FEATURE_COLUMN = "column_names"


def get_trino_engine() -> Engine:
    """Create and return a Trino engine instance."""
    return create_engine(
        f"trino://{TRINO_USERNAME}:{TRINO_PASSWORD}@{TRINO_IP_ADDRESS}:8443/iceberg",
        connect_args={
            "http_scheme": "https",
            "verify": False,
            "session_properties": {"distinct_aggregations_strategy": "single_step"},
        },
    )


@task(
    name="extract-subject-area-features",
    retries=2,
    retry_delay_seconds=30,
    cache_policy=NO_CACHE,
)
def extract_subject_area_features(
    target_schemas: list[str],
    only_tables: list[TableRef] | None = None,
) -> int:
    """
    Build one feature row per table and write it to iceberg.staging.stg_subject_area.

    Args:
        target_schemas: Schemas to scan for tables/columns.
        only_tables: Restrict extraction to exactly these tables (streaming mode).

    Returns:
        Number of table rows written to the staging table.
    """
    engine = get_trino_engine()

    schema_filter = "', '".join(target_schemas)
    # ORDER BY ordinal_position is load-bearing, not cosmetic: training joined the
    # column names in ordinal order, so a different order here would hand the TF-IDF
    # vectoriser a different string for the same table.
    discovery_query = text(f"""
        SELECT c.table_catalog, c.table_schema, c.table_name, c.column_name
        FROM iceberg.information_schema.columns AS c
        WHERE c.table_schema IN ('{schema_filter}')
        ORDER BY c.table_catalog, c.table_schema, c.table_name, c.ordinal_position ASC
    """)  # noqa: S608 - schema names are operator-supplied config, not user input

    with engine.connect() as connection:
        found_columns = connection.execute(discovery_query).fetchall()

    print(f"Found {len(found_columns)} columns across all tables")

    grouped: dict = {}
    for database, schema, table, column in found_columns:
        grouped.setdefault((database, schema, table), []).append(column)

    if only_tables is not None:
        wanted = {tuple(ref) for ref in only_tables}
        grouped = {k: v for k, v in grouped.items() if k in wanted}
        print(f"Restricted to {len(grouped)} changed tables")

    if not grouped:
        print("No data available - check tables and connection.")
        return 0

    df_final = pd.DataFrame(
        [
            {
                "database": database,
                "schema": schema,
                "table_name": table,
                FEATURE_COLUMN: ", ".join(columns),
            }
            for (database, schema, table), columns in grouped.items()
        ],
    )

    # Rebuilt each run, mirroring stg_column_features and stg_normalform_features,
    # so the features can be inspected without running prediction.
    with engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS iceberg.staging"))
        df_final.to_sql(
            "stg_subject_area",
            connection,
            schema="staging",
            if_exists="replace",
            index=False,
        )

    print(
        f"Extracted features for {len(df_final)} tables "
        "(written to iceberg.staging.stg_subject_area)",
    )
    return len(df_final)


def _build_requeue_clause(
    requeue_tables: list[TableRef] | None,
) -> tuple[str, dict[str, str]]:
    """
    Build the SQL fragment that re-opens already-predicted tables.

    Without this a table is predicted exactly once, ever: the queue dedup skips
    anything with ``processed = TRUE``. A flagged table is re-predicted so the
    results table keeps a dated row per detection, the same contract the other two
    tracks offer.

    Returns:
        The ``OR (...)`` fragment (empty string if nothing is flagged) and the bound
        parameters it references.
    """
    if not requeue_tables:
        return "", {}

    predicates = []
    params: dict[str, str] = {}
    for i, ref in enumerate(requeue_tables):
        params[f"rq_db_{i}"] = ref.database
        params[f"rq_sc_{i}"] = ref.schema
        params[f"rq_tb_{i}"] = ref.table_name
        predicates.append(
            f"(f.database = :rq_db_{i} AND f.schema = :rq_sc_{i} AND f.table_name = :rq_tb_{i})",
        )

    return f"OR ({' OR '.join(predicates)})", params


@task(name="populate-subject-area-queue", cache_policy=NO_CACHE)
def populate_subject_area_queue(requeue_tables: list[TableRef] | None = None) -> int:
    """
    Populate the subject-area queue from the staging features table.

    A table is queued when it has no unprocessed row waiting AND either it was never
    predicted before, or it appears in ``requeue_tables``.

    Args:
        requeue_tables: Tables that must be predicted again because the detector
            flagged them. Supplied by the streaming trigger.

    Returns:
        Number of rows inserted into the queue.
    """
    requeue_clause, requeue_params = _build_requeue_clause(requeue_tables)
    trino_engine = get_trino_engine()

    with trino_engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS iceberg.predictions"))
        # id is INTEGER because store_predictions_to_trino marks rows processed via
        # int(id), matching how the pk/fk queue is keyed.
        conn.execute(
            text(f"""
                CREATE TABLE IF NOT EXISTS iceberg.predictions.subject_area_queue (
                    id INTEGER,
                    database VARCHAR,
                    schema VARCHAR,
                    table_name VARCHAR,
                    {FEATURE_COLUMN} VARCHAR,
                    processed BOOLEAN,
                    processed_at VARCHAR,
                    created_at VARCHAR
                )
            """),
        )
        print("Subject-area queue table verified/created")

        # Determine the id offset so new rows get unique, incrementing integer ids
        # across runs (Trino has no auto-increment / sequences).
        max_id = (
            conn.execute(
                text("SELECT COALESCE(MAX(id), 0) FROM iceberg.predictions.subject_area_queue"),
            ).scalar()
            or 0
        )

        # Insert new rows from staging that aren't already queued unprocessed.
        # Column list is explicit so the projection is matched by name, not position.
        insert_query = text(f"""
            INSERT INTO iceberg.predictions.subject_area_queue (
                id, database, schema, table_name, {FEATURE_COLUMN},
                processed, processed_at, created_at
            )
            SELECT
                CAST({max_id} + ROW_NUMBER() OVER (
                    ORDER BY database, schema, table_name
                ) AS INTEGER) AS id,
                database,
                schema,
                table_name,
                {FEATURE_COLUMN},
                FALSE AS processed,
                CAST(NULL AS VARCHAR) AS processed_at,
                CAST(CURRENT_TIMESTAMP AS VARCHAR) AS created_at
            FROM iceberg.staging.stg_subject_area AS f
            WHERE NOT EXISTS (
                SELECT 1
                FROM iceberg.predictions.subject_area_queue AS q
                WHERE q.database = f.database
                  AND q.schema = f.schema
                  AND q.table_name = f.table_name
                  AND q.processed = FALSE
            )
            AND (
                NOT EXISTS (
                    SELECT 1
                    FROM iceberg.predictions.subject_area_queue AS q
                    WHERE q.database = f.database
                      AND q.schema = f.schema
                      AND q.table_name = f.table_name
                      AND q.processed = TRUE
                )
                {requeue_clause}
            )
        """)  # noqa: S608 - max_id is an int and requeue_clause is built from bound parameters

        result = conn.execute(insert_query, requeue_params)
        rows_inserted = result.rowcount if hasattr(result, "rowcount") else 0

        print(f"Inserted {rows_inserted} new rows into the subject-area queue")

        status = conn.execute(
            text("""
                SELECT
                    COUNT(*) as total_rows,
                    SUM(CASE WHEN processed = FALSE THEN 1 ELSE 0 END) as unprocessed,
                    SUM(CASE WHEN processed = TRUE THEN 1 ELSE 0 END) as processed
                FROM iceberg.predictions.subject_area_queue
            """),
        ).fetchone()
        print(
            f"Queue status - Total: {status[0]}, Unprocessed: {status[1]}, Processed: {status[2]}",
        )

        return rows_inserted


@task(name="check-subject-area-queue-status", cache_policy=NO_CACHE)
def check_queue_status() -> dict:
    trino_engine = get_trino_engine()

    with trino_engine.connect() as conn:
        query = text("""
            SELECT
                COUNT(*) as total_rows,
                SUM(CASE WHEN processed = FALSE THEN 1 ELSE 0 END) as unprocessed,
                SUM(CASE WHEN processed = TRUE THEN 1 ELSE 0 END) as processed
            FROM iceberg.predictions.subject_area_queue
        """)

        result = conn.execute(query).fetchone()

        return {
            "total": result[0],
            "unprocessed": result[1],
            "processed": result[2],
        }


@task(name="fetch-new-subject-area-rows", cache_policy=NO_CACHE)
def fetch_new_rows(batch_size: int = 100) -> pd.DataFrame:
    """Fetch a batch of unprocessed rows from the subject-area queue."""
    trino_engine = get_trino_engine()

    query = text(f"""
        SELECT id, database, schema, table_name, {FEATURE_COLUMN}
        FROM iceberg.predictions.subject_area_queue
        WHERE processed = FALSE
        LIMIT :batch_size
    """)  # noqa: S608 - FEATURE_COLUMN is a module constant, not user input

    with trino_engine.connect() as conn:
        return pd.read_sql(query, conn, params={"batch_size": batch_size})


def _row_to_payload(row: pd.Series) -> dict:
    """Build the SubjectArea request payload from a queue row."""
    return {
        "table_name": str(row["table_name"]),
        "columns": str(row[FEATURE_COLUMN]),
    }


@task(name="predict-subject-area-batch", retries=2, retry_delay_seconds=10)
def predict_batch(rows: pd.DataFrame) -> list[dict]:
    """Call the subject-area endpoint once per queued table."""
    url = f"{MODEL_API_URL}/predict_subject_area"
    predictions = []

    for _idx, row in rows.iterrows():
        row_prediction = {"queue_id": row["id"]}
        prediction_successful = True

        try:
            response = requests.post(url, json=_row_to_payload(row), timeout=30)
            response.raise_for_status()
            result = response.json()

            # `prediction` is the subject-area name (a string), unlike the int class
            # the other five models return.
            prediction_value = result.get("prediction")
            row_prediction["predicted_subject_area"] = prediction_value
            row_prediction["confidence"] = result.get("probability", 0.0)

            if prediction_value is None:
                prediction_successful = False
                print(f"Warning: subject-area prediction returned None for row {row['id']}")
        except (requests.RequestException, requests.Timeout, requests.HTTPError) as e:
            loc = f"{row['schema']}.{row['table_name']}"
            print(f"Error predicting subject area for {loc}: {e}")
            row_prediction["predicted_subject_area"] = None
            row_prediction["confidence"] = 0.0
            prediction_successful = False

        row_prediction.update(
            {
                "database": row["database"],
                "schema": row["schema"],
                "table_name": row["table_name"],
                "predicted_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
                "prediction_successful": prediction_successful,
            },
        )
        predictions.append(row_prediction)

    return predictions


@task(name="store-subject-area-predictions", cache_policy=NO_CACHE)
def store_predictions_to_trino(predictions: list[dict]) -> None:
    """Store all predictions, and mark a queue row processed only if the model answered."""
    if not predictions:
        return

    trino_engine = get_trino_engine()

    successful_predictions = [p for p in predictions if p.get("prediction_successful")]
    failed_predictions = [p for p in predictions if not p.get("prediction_successful")]

    if failed_predictions:
        print("Failed queue_ids:")
        for p in failed_predictions:
            print(f"  - {p['queue_id']}")

    df = pd.DataFrame(predictions)
    if "prediction_successful" in df.columns:
        df = df.drop(columns=["prediction_successful"])

    # Reorder columns for readability, keeping only those present.
    column_order = [
        "queue_id",
        "database",
        "schema",
        "table_name",
        "predicted_subject_area",
        "confidence",
        "predicted_at",
    ]
    df = df[[c for c in column_order if c in df.columns]]

    df["queue_id"] = df["queue_id"].astype(str)
    df["predicted_subject_area"] = df["predicted_subject_area"].fillna("NULL").astype(str)

    with trino_engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS iceberg.prediction_results"))
        df.to_sql(
            "subject_area_results",
            conn,
            schema="prediction_results",
            if_exists="append",
            index=False,
        )

        if not successful_predictions:
            print("No successful predictions to mark as processed")
            return

        successful_queue_ids = [p["queue_id"] for p in successful_predictions]
        placeholders = ", ".join(f":id_{i}" for i in range(len(successful_queue_ids)))
        params = {f"id_{i}": int(qid) for i, qid in enumerate(successful_queue_ids)}
        params["current_timestamp"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        update_query = text(f"""
            UPDATE iceberg.predictions.subject_area_queue
            SET processed = TRUE,
                processed_at = :current_timestamp
            WHERE id IN ({placeholders})
        """)  # noqa: S608 - placeholders are bound parameters, not user input
        conn.execute(update_query, params)
        print(f"Marked {len(successful_predictions)} rows as processed in the queue")


@task(name="count-unprocessed-subject-area-for-tables", cache_policy=NO_CACHE)
def count_unprocessed_for_tables(tables: list[TableRef]) -> int:
    """
    How many queue rows for these tables are still waiting to be predicted.

    This is the success test for a streaming run: the pending change may only be
    marked complete once its tables actually have predictions.
    """
    if not tables:
        return 0

    predicates = []
    params: dict[str, str] = {}
    for i, ref in enumerate(tables):
        params[f"db_{i}"] = ref.database
        params[f"sc_{i}"] = ref.schema
        params[f"tb_{i}"] = ref.table_name
        predicates.append(
            f"(database = :db_{i} AND schema = :sc_{i} AND table_name = :tb_{i})",
        )

    query = text(f"""
        SELECT COUNT(*)
        FROM iceberg.predictions.subject_area_queue
        WHERE processed = FALSE
          AND ({" OR ".join(predicates)})
    """)  # noqa: S608 - predicates are built from bound parameters

    with get_trino_engine().connect() as conn:
        return conn.execute(query, params).scalar() or 0


@task(name="claim-subject-area-changes", cache_policy=NO_CACHE)
def claim_subject_area_changes(run_id: str) -> list[TableRef]:
    """
    Claim the tables the change detector flagged for the subject-area track.

    Retried on commit conflicts: all three tracks are woken by the same event and
    claim the same rows, so on iceberg two of the three lose the commit race.
    """
    return with_commit_retry(
        get_trino_engine(),
        lambda conn: claim_pending_changes(conn, TRACK_SUBJECT_AREA, run_id),
    )


@task(name="close-subject-area-changes", cache_policy=NO_CACHE)
def close_subject_area_changes(run_id: str, *, succeeded: bool) -> int:
    """Complete the claim on success, release it on failure so the next run retries."""

    def close(conn: Connection) -> int:
        if succeeded:
            return complete_pending_changes(conn, TRACK_SUBJECT_AREA, run_id)
        return release_pending_changes(conn, TRACK_SUBJECT_AREA, run_id)

    return with_commit_retry(get_trino_engine(), close)


def _drain_queue(prediction_batch_size: int, drain_queue: bool) -> int:
    """
    Predict unprocessed queue rows and return how many predictions were made.

    With ``drain_queue`` the loop keeps going until the backlog is empty. It stops
    early when a batch fails to reduce the unprocessed count - rows whose model call
    failed stay ``processed = FALSE``, so without that guard the loop would spin on
    the same failing batch forever.
    """
    predictions_made = 0
    previous_unprocessed = None

    while True:
        queue_stats = check_queue_status()
        unprocessed = queue_stats["unprocessed"] or 0

        if unprocessed == 0:
            print("All rows in queue already processed")
            break

        if previous_unprocessed is not None and unprocessed >= previous_unprocessed:
            print(
                f"Stopping: {unprocessed} rows still unprocessed after a full batch "
                "(their model calls are failing). Check the model service.",
            )
            break
        previous_unprocessed = unprocessed

        print(f"Predicting up to {prediction_batch_size} of {unprocessed} unprocessed rows...")
        start_time = time.time()
        rows = fetch_new_rows(prediction_batch_size)
        if rows.empty:
            break

        predictions = predict_batch(rows)
        store_predictions_to_trino(predictions)
        predictions_made += len(predictions)
        print(f"Processed {len(predictions)} predicts in {round(time.time() - start_time, 1)}s")

        if not drain_queue:
            break

    return predictions_made


@flow(name="subject-area-prediction-pipeline")
def subject_area_prediction_pipeline(
    target_schemas: list[str] | None = None,
    prediction_batch_size: int = 100,
    use_pending_changes: bool = False,
    drain_queue: bool = False,
) -> dict:
    """
    Complete subject-area track: Extract features → Queue → Predict

    Args:
        target_schemas: Schemas to scan for features (default: ['new_predict_data']).
        prediction_batch_size: Batch size for a single prediction round.
        use_pending_changes: Streaming mode. Only process the tables the change
            detector flagged, and re-predict the ones already processed.
            False (default) keeps the original full-schema scan.
        drain_queue: Keep predicting until the queue backlog is empty instead of
            stopping after one batch. Implied by ``use_pending_changes``, because a
            change is only marked complete once its tables are actually predicted.

    There is no ``batch_size``: unlike the other two tracks this one runs a single
    information_schema query and no value profiling, so there is nothing to chunk.
    """
    if target_schemas is None:
        target_schemas = ["new_predict_data"]

    only_tables: list[TableRef] | None = None
    run_id = str(flow_run.get_id())

    if use_pending_changes:
        drain_queue = True
        print("Step 0: Claiming pending changes for the subject-area track...")
        only_tables = claim_subject_area_changes(run_id)
        if not only_tables:
            # Normal case for a duplicate/late event - the work was already taken.
            print("No pending changes to process - nothing to do.")
            return {"rows_written": 0, "rows_queued": 0, "predictions_made": 0, "tables": 0}
        print(f"Claimed {len(only_tables)} changed tables")

    succeeded = False
    rows_written = 0
    rows_inserted = 0
    predictions_made = 0
    try:
        # Step 1: Extract features into iceberg.staging.stg_subject_area
        print("Step 1: Extracting subject-area features from information_schema...")
        rows_written = extract_subject_area_features(
            target_schemas=target_schemas,
            only_tables=only_tables,
        )

        if rows_written == 0:
            print("No features extracted - stopping pipeline.")
            # Nothing to predict, but the claim is genuinely handled: the flagged
            # tables are unreadable or gone, so retrying them would loop forever.
            succeeded = True
            return {"rows_written": 0, "rows_queued": 0, "predictions_made": 0}

        # Step 2: Populate the prediction queue
        print("Step 2: Populating the subject-area queue...")
        rows_inserted = populate_subject_area_queue(requeue_tables=only_tables)

        # Step 3: Predict unprocessed rows and store the results.
        predictions_made = _drain_queue(prediction_batch_size, drain_queue)

        # Step 4: A run only counts as done once the claimed tables actually have
        # predictions. Without this check a run whose every model call returned an
        # error would still mark its changes complete, and the retry path built into
        # the detector would never see them again - the work would vanish silently.
        if use_pending_changes:
            leftover = count_unprocessed_for_tables(only_tables)
            if leftover:
                msg = (
                    f"{leftover} queue rows for the claimed tables are still "
                    f"unpredicted (model calls failing). Releasing the changes so the "
                    f"next detection pass retries them."
                )
                print(msg)
                raise RuntimeError(msg)

        succeeded = True
    finally:
        if use_pending_changes:
            closed = close_subject_area_changes(run_id, succeeded=succeeded)
            verb = "completed" if succeeded else "released for retry"
            print(f"{closed} pending-change rows {verb}")

    print("Subject Area Pipeline Complete!")
    return {
        "rows_written": rows_written,
        "rows_queued": rows_inserted,
        "queue_stats": check_queue_status(),
        "predictions_made": predictions_made,
    }


if __name__ == "__main__":
    subject_area_prediction_pipeline(
        target_schemas=["new_predict_data"],
        prediction_batch_size=100,
    )
