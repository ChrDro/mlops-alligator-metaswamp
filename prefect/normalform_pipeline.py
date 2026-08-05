"""
Prefect flow for the normal-form (denormalization) prediction track.

Separate from the pk/fk/cpk/cfk pipeline on purpose (see
documentation/NORMALFORM_PREDICTION.md):
- the model (denormalization_model) is MULTICLASS 0-3 (0=violates 1NF, 1/2/3=pure NF),
- the label is a TABLE-level property inferred from per-column feature rows,
- it uses a different, larger feature set.

This flow:
1. Skips tables already present in nf_results (each table is processed only once).
2. Profiles the remaining tables with ``nf_features.build_features``, writing the result
   to iceberg.staging.stg_normalform_features for inspection.
3. Calls POST /predict_normalform per column via the model API.
4. Aggregates per table by CONFIDENCE-WEIGHTED VOTE -> one normal-form class per table.
5. Stores ONE row per table in iceberg.prediction_results.nf_results.

Where the features come from
----------------------------
``nf_features``, the same module the training set is built with - not a second
implementation of the same formulas. This flow used to carry its own ~250 lines of feature
SQL, ported "verbatim" from the training extractor, which is precisely the skew finding 1.6
describes: two copies that agree until one of them is edited, and the 1NF mean-length guard
had already drifted (120 characters per value here against 20 per token there). A change to
a feature now moves training and serving together or not at all.

What is deliberately unchanged: predicting per column and combining into one table-level
class. Whether the model should be trained at table level instead is an open question in
Phase 4 of TASK_3_PLAN.md, and answering it by accident here would confound the comparison.
"""

import os

import pandas as pd
import requests
import urllib3
from change_events import (
    TRACK_NF,
    TableRef,
    claim_pending_changes,
    complete_pending_changes,
    release_pending_changes,
    with_commit_retry,
)
from dotenv import load_dotenv
from model_health import diagnose, failure_detail
from nf_features import (
    COLUMN_TYPE_DUMMIES,
    FEATURE_COLUMNS,
    FLOAT_FEATURE_COLUMNS,
    build_features,
    encode_column_type,
)
from prefect.cache_policies import NONE as NO_CACHE
from prefect.runtime import flow_run
from sqlalchemy import Connection, Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from prefect import flow, task


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

TRINO_IP_ADDRESS = str(os.environ.get("TRINO_IP_ADDRESS"))
TRINO_USERNAME = str(os.environ.get("TRINO_USERNAME"))
TRINO_PASSWORD = str(os.environ.get("TRINO_PASSWORD"))

# Model API base (use localhost outside Docker, model-service name inside).
MODEL_API_URL = os.environ.get("MODEL_API_URL", "http://localhost:8080")

# The column_type dummies, from nf_features. The previous list here named seven types
# including char, decimal and timestamp - none of which occur in the data - while omitting
# `bigint`, which is 89% of it. The training script's `get_dummies` would have produced
# `column_type_bigint` and this pipeline would never have sent it: the same skew as the
# feature lists, one level down.
EXPECTED_TYPE_COLS = list(COLUMN_TYPE_DUMMIES)

# What the model sees, in order: every measured feature plus the one-hot column type.
#
# Derived from nf_features rather than listed here. The previous version was a hand-kept
# copy of the training script's selection, with a comment saying so - and a hand-kept copy
# of a formula is exactly the shape of skew finding 1.6 describes. There is now one list,
# in the module that produces the numbers, and a test that fails if it drifts from what
# `build_features` actually returns.
MODEL_FEATURES: list[str] = [*FEATURE_COLUMNS, *EXPECTED_TYPE_COLS]

# Features the API expects as float; everything else numeric is int, column_type_* are bool.
FLOAT_FEATURES = FLOAT_FEATURE_COLUMNS


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


@task(name="extract-normalform-features", retries=2, retry_delay_seconds=30, cache_policy=NO_CACHE)
def extract_normalform_features(
    target_schemas: list[str],
    batch_size: int = 30,
    skip_tables: set[tuple[str, str, str]] | None = None,
    only_tables: list[TableRef] | None = None,
) -> pd.DataFrame:
    """
    Profile every table in the target schema(s) into the model's feature set, one row per
    column, plus the id columns database/schema/table_name/column_name for aggregation.

    Args:
        target_schemas: Schemas to scan for tables.
        batch_size: Aggregates per profiling query. The profiler decides how many queries a
            table needs; this only caps how wide each SELECT list gets.
        skip_tables: (database, schema, table_name) triples to skip entirely (already
            processed in a previous run) so each table is only profiled/predicted once.
        only_tables: Restrict profiling to exactly these tables (streaming mode).
            Takes precedence over skip_tables - see the filter below for why.

    Returns a DataFrame with one row per column.
    """
    skip_tables = skip_tables or set()
    engine = get_trino_engine()

    schema_filter = "', '".join(target_schemas)
    discovery_query = text(f"""
        SELECT table_catalog, table_schema, table_name
        FROM iceberg.information_schema.tables
        WHERE table_schema IN ('{schema_filter}')
        ORDER BY table_schema, table_name
    """)  # noqa: S608 - schema names are operator-supplied config, not user input

    with engine.connect() as connection:
        found = [tuple(row) for row in connection.execute(discovery_query).fetchall()]

    print(f"Found {len(found)} tables")

    if only_tables is not None:
        # Streaming mode: the change detector already decided what needs work, so the
        # already-processed filter must NOT apply - a reloaded table is meant to be
        # profiled again even though nf_results holds an older row for it.
        wanted = {tuple(ref) for ref in only_tables}
        found = [ref for ref in found if ref in wanted]
        print(f"Restricted to {len(found)} changed tables")
    elif skip_tables:
        before = len(found)
        found = [ref for ref in found if ref not in skip_tables]
        print(f"Skipping {before - len(found)} already-processed tables")

    print(f"Processing {len(found)} tables")

    all_table_features = []
    with engine.connect() as connection:
        for database, schema, table in found:
            try:
                df_table = build_features(
                    connection,
                    database,
                    schema,
                    table,
                    batch_size=batch_size,
                )
            except SQLAlchemyError as error:
                # One unreadable table must not take the run down - the rest still predict.
                orig = getattr(error, "orig", None)
                message = str(orig) if orig else str(error).split("\n")[0]
                print(f"Error at {database}.{schema}.{table}: {message}")
                continue
            all_table_features.append(encode_column_type(df_table))

    if not all_table_features:
        print("No data available - check tables and connection.")
        return pd.DataFrame()

    df_final = pd.concat(all_table_features, ignore_index=True)

    # Keep the id columns (for aggregation) plus exactly the model features. `column_type`
    # drops out here: the model sees its one-hot dummies, not the raw string.
    id_cols = ["database", "schema", "table_name", "column_name"]
    missing = [f for f in MODEL_FEATURES if f not in df_final.columns]
    if missing:
        msg = f"Extractor is missing required model features: {missing}"
        raise RuntimeError(msg)

    df_final = df_final[id_cols + MODEL_FEATURES]
    n_tables = df_final["table_name"].nunique()

    # Persist the extracted features (rebuilt each run) so they can be inspected without
    # running prediction, mirroring pk_fk_pipeline's stg_column_features. This is its OWN
    # table: the normalform feature set differs from the pk/fk one, so the two must never
    # share a staging table.
    with engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS iceberg.staging"))
        df_final.to_sql(
            "stg_normalform_features",
            connection,
            schema="staging",
            if_exists="replace",
            index=False,
        )

    print(
        f"Extracted features for {len(df_final)} columns across {n_tables} tables "
        "(written to iceberg.staging.stg_normalform_features)",
    )
    return df_final


def _row_to_payload(row: pd.Series) -> dict:
    """Build the NormalForm request payload from a feature row, with correct types."""
    payload = {}
    for feat in MODEL_FEATURES:
        value = row[feat]
        if feat in EXPECTED_TYPE_COLS:
            payload[feat] = bool(value)
        elif feat in FLOAT_FEATURES:
            payload[feat] = float(value)
        else:
            payload[feat] = int(value)
    return payload


@task(name="predict-normalform", retries=2, retry_delay_seconds=10, cache_policy=NO_CACHE)
def predict_normalform(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Call POST /predict_normalform for each column row. Returns the input frame with
    per-column `prediction` (NF class), `confidence`, and `probabilities` (the full
    per-class distribution, keyed by class label) columns added, plus the distinct
    reasons any calls failed so a run that predicted nothing can say why.

    `probabilities` is what `aggregate_to_table` votes with - `confidence` alone is
    just its max, and a confidence-weighted vote needs the whole distribution.
    """
    url = f"{MODEL_API_URL}/predict_normalform"
    predictions = []
    confidences = []
    probabilities = []
    failure_reasons: set[str] = set()

    for _idx, row in features.iterrows():
        try:
            response = requests.post(url, json=_row_to_payload(row), timeout=30)
            response.raise_for_status()
            result = response.json()
            predictions.append(result.get("prediction"))
            confidences.append(result.get("probability", 0.0))
            probabilities.append(result.get("probabilities"))
        except (requests.RequestException, requests.Timeout, requests.HTTPError) as e:
            loc = f"{row['schema']}.{row['table_name']}.{row['column_name']}"
            # Keep the service's explanation, not just "400 Bad Request".
            detail = failure_detail(e)
            print(f"Error predicting normalform for {loc}: {detail}")
            failure_reasons.add(detail)
            predictions.append(None)
            confidences.append(0.0)
            probabilities.append(None)

    out = features.copy()
    out["prediction"] = predictions
    out["confidence"] = confidences
    out["probabilities"] = probabilities
    return out, sorted(failure_reasons)


@task(name="aggregate-normalform", cache_policy=NO_CACHE)
def aggregate_to_table(predicted: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-column predictions to one row per table via confidence-weighted
    (soft) voting: average every column's full probability distribution, then take
    the class with the highest averaged probability.

    Not a hard vote over `prediction` labels. Two columns that are barely 51% sure
    would otherwise outvote one column that is 99% sure, and an exact split would be
    broken arbitrarily by `value_counts` row order rather than by which columns were
    actually confident. Averaging probabilities fixes both: a confident column
    outweighs unsure ones, and a tie is only a tie when the averaged distribution
    itself is tied.
    """
    predicted = predicted.dropna(subset=["prediction", "probabilities"]).copy()
    if predicted.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["database", "schema", "table_name"]
    for (database, schema, table), grp in predicted.groupby(group_cols):
        # Each entry in `probabilities` is a {class_label: probability} dict from
        # one column; stacking them into a frame and averaging columnwise gives the
        # table's combined distribution, one column per NF class.
        mean_distribution = pd.DataFrame(list(grp["probabilities"])).astype(float).mean()
        winning_class = int(mean_distribution.idxmax())
        rows.append(
            {
                "database": database,
                "schema": schema,
                "table_name": table,
                "predicted_normal_form": winning_class,
                "confidence": round(float(mean_distribution.max()), 4),
                "n_columns": len(grp),
            },
        )

    result = pd.DataFrame(rows)
    print(f"Aggregated predictions for {len(result)} tables")
    return result


@task(name="store-normalform-results", cache_policy=NO_CACHE)
def store_results(results: pd.DataFrame) -> int:
    """Append one row per table to iceberg.prediction_results.nf_results."""
    if results.empty:
        print("No normalform results to store.")
        return 0

    results = results.copy()
    results["predicted_at"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    engine = get_trino_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS iceberg.prediction_results"))
        results.to_sql(
            "nf_results",
            conn,
            schema="prediction_results",
            if_exists="append",
            index=False,
        )
    print(f"Stored {len(results)} table-level normalform predictions")
    return len(results)


@task(name="get-processed-tables", cache_policy=NO_CACHE)
def get_processed_tables() -> set[tuple[str, str, str]]:
    """
    Return the (database, schema, table_name) triples already stored in nf_results, so a
    table is only ever profiled/predicted once across runs. This mirrors the pk/fk queue's
    processed-row dedup, at table grain (normalform stores one row per table).
    """
    engine = get_trino_engine()
    with engine.connect() as conn:
        table_exists = conn.execute(
            text(
                "SELECT COUNT(*) FROM iceberg.information_schema.tables "
                "WHERE table_schema = 'prediction_results' AND table_name = 'nf_results'",
            ),
        ).scalar()
        if not table_exists:
            return set()
        rows = conn.execute(
            text(
                "SELECT DISTINCT database, schema, table_name "
                "FROM iceberg.prediction_results.nf_results",
            ),
        ).fetchall()
    return {(r[0], r[1], r[2]) for r in rows}


@task(name="claim-nf-changes", cache_policy=NO_CACHE)
def claim_nf_changes(run_id: str) -> list[TableRef]:
    """
    Claim the tables the change detector flagged for the normalform track.

    Retried on commit conflicts: all three tracks are woken by the same event and
    claim the same rows, so on iceberg two of the three lose the commit race.
    """
    return with_commit_retry(
        get_trino_engine(),
        lambda conn: claim_pending_changes(conn, TRACK_NF, run_id),
    )


@task(name="close-nf-changes", cache_policy=NO_CACHE)
def close_nf_changes(run_id: str, *, succeeded: bool) -> int:
    """Complete the claim on success, release it on failure so the next run retries."""

    def close(conn: Connection) -> int:
        if succeeded:
            return complete_pending_changes(conn, TRACK_NF, run_id)
        return release_pending_changes(conn, TRACK_NF, run_id)

    return with_commit_retry(get_trino_engine(), close)


@flow(name="normalform-prediction-pipeline")
def normalform_prediction_pipeline(
    target_schemas: list[str] | None = None,
    batch_size: int = 30,
    use_pending_changes: bool = False,
) -> dict:
    """
    Complete normalform track: pick the tables to process -> extract 29 features ->
    predict per column -> majority-vote aggregate to one NF class per table -> store.

    Args:
        target_schemas: Schemas to scan (default: ['new_predict_data']).
        batch_size: Column batch size for the profiling queries.
        use_pending_changes: Streaming mode. Process exactly the tables the change
            detector flagged, including ones that already have a result - a reloaded
            table gets a fresh prediction. Results are appended, so nf_results keeps
            the full history per table rather than overwriting it.
            False (default) keeps the original "each table only once" behaviour.
    """
    if target_schemas is None:
        target_schemas = ["new_predict_data"]

    only_tables: list[TableRef] | None = None
    processed_tables: set[tuple[str, str, str]] | None = None
    run_id = str(flow_run.get_id())

    if use_pending_changes:
        print("Step 0: Claiming pending changes for the normalform track...")
        only_tables = claim_nf_changes(run_id)
        if not only_tables:
            print("No pending changes to process - nothing to do.")
            return {"tables": 0}
        print(f"Claimed {len(only_tables)} changed tables")
    else:
        print("Step 1: Checking which tables were already processed...")
        processed_tables = get_processed_tables()
        print(f"{len(processed_tables)} tables already have results")

    succeeded = False
    stored = 0
    try:
        print("Step 2: Extracting normalform features...")
        features = extract_normalform_features(
            target_schemas=target_schemas,
            batch_size=batch_size,
            skip_tables=processed_tables,
            only_tables=only_tables,
        )
        if features.empty:
            print("No tables to process - stopping.")
            # Claim is handled: the flagged tables are unreadable or gone, so
            # retrying them would loop forever.
            succeeded = True
            return {"tables": 0}

        print("Step 3: Predicting normal form per column...")
        predicted, failure_reasons = predict_normalform(features)

        print("Step 4: Aggregating to one row per table...")
        table_results = aggregate_to_table(predicted)

        print("Step 5: Storing table-level results...")
        stored = store_results(table_results)

        # A run only counts as done once every profiled table produced a result.
        # Otherwise a run whose model calls all failed would still mark its changes
        # complete, and the detector's retry path would never see them again.
        expected = len(features[["database", "schema", "table_name"]].drop_duplicates())
        if use_pending_changes and stored < expected:
            msg = (
                f"Only {stored} of {expected} profiled tables produced a normalform "
                f"result.\n"
                f"{diagnose(failure_reasons, MODEL_API_URL)}\n"
                f"  Note:  retriggering this flow changes nothing until the above is "
                f"fixed. The claimed changes are released, so the next detection pass "
                f"picks these tables up again automatically."
            )
            print(msg)
            raise RuntimeError(msg)

        succeeded = True
    finally:
        if use_pending_changes:
            closed = close_nf_changes(run_id, succeeded=succeeded)
            verb = "completed" if succeeded else "released for retry"
            print(f"{closed} pending-change rows {verb}")

    print("Normalform pipeline complete!")
    return {"tables": stored}


if __name__ == "__main__":
    normalform_prediction_pipeline(target_schemas=["new_predict_data"], batch_size=30)
