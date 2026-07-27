"""
Prefect flow for orchestrating the feature engineering and prediction pipeline.

This flow follows 3 steps:
1. Extracts column-level features from information_schema (plain Python/pandas)
   and writes them to duckdb.staging.stg_column_features in Trino.
2. Populates the prediction queue from the extracted features.
3. Triggers the prediction flow to call the ML model APIs and store results.

Note: feature extraction used to be a dbt Python model, but dbt-trino does not
support Python models (Trino has no Python runtime). The model also bypassed the
dbt session entirely and used its own SQLAlchemy connection, so it has been moved
here as a normal Prefect task with no loss of functionality.
"""

import os
import re
import time
from collections.abc import Iterator
from datetime import UTC, datetime

import pandas as pd
import requests
import urllib3
from change_events import (
    TRACK_KEYS,
    TableRef,
    claim_pending_changes,
    complete_pending_changes,
    release_pending_changes,
)
from dotenv import load_dotenv
from prefect.cache_policies import NONE as NO_CACHE
from prefect.runtime import flow_run
from sqlalchemy import Engine, create_engine, text

from prefect import flow, task


# Suppress InsecureRequestWarning for unverified HTTPS requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

TRINO_IP_ADDRESS = str(os.environ.get("TRINO_IP_ADDRESS"))
TRINO_USERNAME = str(os.environ.get("TRINO_USERNAME"))
TRINO_PASSWORD = str(os.environ.get("TRINO_PASSWORD"))

# Model API base (localhost outside Docker, model-service:8080 inside the compose network).
MODEL_API_URL = os.environ.get("MODEL_API_URL", "http://localhost:8080")

# The four column-level models served by the API, in stored-column order.
PREDICTION_MODELS = ["pk", "fk", "cpk", "cfk"]

# Column-type dummy columns for the feature table
EXPECTED_TYPE_COLS = [
    "column_type_boolean",
    "column_type_date",
    "column_type_decimal",
    "column_type_double",
    "column_type_integer",
    "column_type_varchar",
]


def get_trino_engine() -> Engine:
    """Create and return a Trino engine instance."""
    return create_engine(
        f"trino://{TRINO_USERNAME}:{TRINO_PASSWORD}@{TRINO_IP_ADDRESS}:8443/duckdb",
        connect_args={
            "http_scheme": "https",
            "verify": False,
            "session_properties": {"distinct_aggregations_strategy": "single_step"},
        },
    )


def chunks(lst: list, n: int) -> Iterator[list]:
    """Split list into chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def build_feature_query(union_sql: str) -> str:
    """Build the full feature extraction query with table-level stats and rankings."""
    return f"""
WITH base AS (
    {union_sql}
),
table_stats AS (
    SELECT
        MAX(CAST(ordinal_position AS INTEGER))   AS table_column_count,
        SUM(CASE
                WHEN number_unique_values = count AND null_count = 0 THEN 1
                ELSE 0
            END)                                 AS table_unique_column_count,
        MIN(CASE
                WHEN number_unique_values = count AND null_count = 0
                THEN CAST(ordinal_position AS INTEGER)
            END)                                 AS table_first_unique_ordinal,
        MAX(count)                               AS table_row_count,
        MAX(CASE
                WHEN number_unique_values = count AND null_count = 0 THEN 1
                ELSE 0
            END)                                 AS table_has_unique_column,
        SUM(CASE
                WHEN count > 0 AND CAST(number_unique_values AS DOUBLE) / count > 0.95 THEN 1
                ELSE 0
            END)                                 AS table_near_unique_column_count,
        SUM(CASE
                WHEN LOWER(column_name) LIKE '%\\_id' ESCAPE '\\' THEN 1
                ELSE 0
            END)                                 AS table_id_named_column_count,
        SUM(CASE
                WHEN null_count = 0 THEN 1
                ELSE 0
            END)                                 AS table_non_null_column_count,
        MAX(CASE
                WHEN count > 0 THEN CAST(number_unique_values AS DOUBLE) / count
                ELSE 0
            END)                                 AS table_max_unique_ratio,
        SUM(CASE
                WHEN column_type IN ('bigint', 'integer', 'int') THEN 1
                ELSE 0
            END)                                 AS table_integer_column_count
    FROM base
),
column_ranks AS (
    SELECT
        column_name,
        ordinal_position,
        ROW_NUMBER() OVER (ORDER BY
            CASE WHEN count > 0 THEN CAST(number_unique_values AS DOUBLE) / count ELSE 0 END DESC,
            CAST(ordinal_position AS INTEGER) ASC
        ) AS unique_ratio_rank,
        ROW_NUMBER() OVER (ORDER BY
            null_ratio ASC,
            CAST(ordinal_position AS INTEGER) ASC
        ) AS null_ratio_rank,
        CASE WHEN count > 0
             THEN CAST(number_unique_values AS DOUBLE) / count
             ELSE 0 END AS col_unique_ratio
    FROM base
)
SELECT
    b.database,
    b.schema,
    b.table_name,
    b.column_name,
    b.column_type,
    b.number_unique_values,
    b.count,
    b.null_count,
    b.null_ratio,
    b.is_unique,
    b.ordinal_position,
    CASE WHEN b.count > 0
         THEN CAST(b.number_unique_values AS DOUBLE) / b.count
         ELSE 0 END                                           AS unique_ratio,
    CASE WHEN b.null_count = 0 THEN 1 ELSE 0 END             AS is_non_null,
    CASE WHEN CAST(b.ordinal_position AS INTEGER) = 1
         THEN 1 ELSE 0 END                                    AS is_first_column,
    CAST(CAST(b.ordinal_position AS INTEGER) AS DOUBLE)
        / NULLIF(t.table_column_count, 0)                    AS relative_ordinal_position,
    CASE WHEN CAST(b.ordinal_position AS INTEGER)
              = t.table_first_unique_ordinal THEN 1
         ELSE 0 END                                           AS is_first_unique_column,
    t.table_column_count,
    t.table_unique_column_count,
    t.table_row_count,
    t.table_unique_column_count
        - CASE WHEN b.number_unique_values = b.count
                AND b.null_count = 0 THEN 1 ELSE 0 END       AS other_unique_columns_in_table,
    t.table_has_unique_column,
    CASE WHEN t.table_has_unique_column = 0 THEN 1 ELSE 0 END AS table_has_no_single_pk_candidate,
    t.table_near_unique_column_count,
    t.table_id_named_column_count,
    t.table_non_null_column_count,
    t.table_max_unique_ratio,
    t.table_integer_column_count,
    r.unique_ratio_rank,
    r.null_ratio_rank,
    CASE WHEN r.null_ratio_rank = 1 THEN 1 ELSE 0 END        AS is_least_null_in_table,
    CASE WHEN t.table_max_unique_ratio > 0
         THEN r.col_unique_ratio / t.table_max_unique_ratio
         ELSE 0 END                                           AS unique_ratio_relative_to_max,
    t.table_near_unique_column_count
        - CASE WHEN b.count > 0 AND CAST(b.number_unique_values AS DOUBLE) / b.count > 0.95
               THEN 1 ELSE 0 END                             AS other_near_unique_columns_in_table,
    CASE WHEN LOWER(b.column_name) LIKE '%\\_id' ESCAPE '\\'
         THEN 1 ELSE 0 END                                     AS name_ends_with_id,
    CASE WHEN LOWER(b.column_name) LIKE '%key%'
         THEN 1 ELSE 0 END                                     AS name_contains_key,
    CASE WHEN LOWER(b.column_name) LIKE '%' || LOWER(b.table_name) || '%'
         THEN 1 ELSE 0 END                                     AS name_contains_table_name,
    CASE WHEN LENGTH(b.table_name) > 1
          AND LOWER(b.column_name) =
              LOWER(
                  CASE WHEN SUBSTR(b.table_name, LENGTH(b.table_name), 1) = 's'
                       THEN SUBSTR(b.table_name, 1, LENGTH(b.table_name) - 1)
                       ELSE b.table_name
                  END
              ) || '_id'
         THEN 1 ELSE 0 END                                     AS name_is_singular_table_id,
    LENGTH(b.column_name)                                      AS name_length
FROM base AS b
CROSS JOIN table_stats AS t
LEFT JOIN column_ranks AS r
    ON b.column_name = r.column_name
    AND b.ordinal_position = r.ordinal_position
ORDER BY CAST(b.ordinal_position AS INTEGER)
"""  # noqa: S608 - union_sql is built from catalog metadata, not user input


def recompute_table_stats(df_table: pd.DataFrame) -> pd.DataFrame:
    """Recompute table-level statistics correctly across all batched columns."""
    table_col_count = int(df_table["ordinal_position"].max())
    table_unique_col_count = int(df_table["is_unique"].sum())
    table_row_count = int(df_table["count"].max())
    table_has_unique = int((df_table["is_unique"] == 1).any())
    table_near_unique = int((df_table["unique_ratio"] > 0.95).sum())
    table_id_named = int(df_table["name_ends_with_id"].sum())
    table_non_null = int((df_table.get("null_count", pd.Series([1])) == 0).sum())
    table_max_unique_ratio = float(df_table["unique_ratio"].max())
    table_int_col = int(df_table["column_type"].isin(["bigint", "integer", "int"]).sum())

    unique_rows = df_table[df_table["is_unique"] == 1]
    first_unique_ord = int(unique_rows["ordinal_position"].min()) if not unique_rows.empty else None

    df_table["table_column_count"] = table_col_count
    df_table["table_unique_column_count"] = table_unique_col_count
    df_table["table_row_count"] = table_row_count
    df_table["table_has_unique_column"] = table_has_unique
    df_table["table_has_no_single_pk_candidate"] = int(table_has_unique == 0)
    df_table["table_near_unique_column_count"] = table_near_unique
    df_table["table_id_named_column_count"] = table_id_named
    df_table["table_non_null_column_count"] = table_non_null
    df_table["table_max_unique_ratio"] = table_max_unique_ratio
    df_table["table_integer_column_count"] = table_int_col
    # Cross-column counts must exclude the current column; recomputed here (not from
    # the SQL) so they stay correct across batches, matching the training extractor.
    df_table["other_unique_columns_in_table"] = table_unique_col_count - df_table["is_unique"]
    df_table["other_near_unique_columns_in_table"] = table_near_unique - (
        df_table["unique_ratio"] > 0.95
    ).astype(int)
    df_table["relative_ordinal_position"] = (
        df_table["ordinal_position"] / table_col_count if table_col_count > 0 else 0.0
    )
    df_table["is_first_unique_column"] = (
        (df_table["ordinal_position"] == first_unique_ord).astype(int)
        if first_unique_ord is not None
        else 0
    )

    rank_idx = df_table.sort_values(
        ["unique_ratio", "ordinal_position"],
        ascending=[False, True],
    ).index
    df_table["unique_ratio_rank"] = pd.Series(range(1, len(rank_idx) + 1), index=rank_idx)

    rank_idx = df_table.sort_values(["ordinal_position"], ascending=[True]).index
    df_table["null_ratio_rank"] = pd.Series(range(1, len(rank_idx) + 1), index=rank_idx)

    df_table["is_least_null_in_table"] = (df_table["null_ratio_rank"] == 1).astype(int)
    df_table["unique_ratio_relative_to_max"] = (
        (df_table["unique_ratio"] / table_max_unique_ratio) if table_max_unique_ratio > 0 else 0.0
    )

    return df_table


@task(name="extract-features", retries=2, retry_delay_seconds=30, cache_policy=NO_CACHE)
def extract_features(
    target_schemas: list[str],
    batch_size: int = 30,
    only_tables: list[TableRef] | None = None,
) -> int:
    """
    Extract column-level features from information_schema and write them to
    duckdb.staging.stg_column_features (rebuilt each run).

    Args:
        target_schemas: Schemas to scan for tables/columns.
        batch_size: Number of columns to profile per Trino query.
        only_tables: Restrict profiling to these tables. Used by the streaming
            trigger so a run only pays for what actually changed instead of
            re-profiling the whole schema. None = profile everything (the
            original behaviour, kept for manual/full runs).

    Returns:
        Number of feature rows written.
    """
    engine = get_trino_engine()

    for schema in target_schemas:
        with engine.connect() as conn:
            create_schema = text(f"""
                CREATE SCHEMA IF NOT EXISTS duckdb.{schema}
            """)

            conn.execute(create_schema)

    # Step 1: Discover all tables and columns in the target schemas.
    schema_filter = "', '".join(target_schemas)
    discovery_query = text(f"""
        SELECT t.table_catalog, t.table_schema, t.table_name,
               c.column_name, c.data_type, c.ordinal_position
        FROM duckdb.information_schema.tables AS t
        INNER JOIN duckdb.information_schema.columns AS c
            ON t.table_catalog = c.table_catalog
            AND t.table_schema = c.table_schema
            AND t.table_name = c.table_name
        WHERE t.table_schema IN ('{schema_filter}')
        ORDER BY c.ordinal_position ASC
    """)  # noqa: S608 - schema names are operator-supplied config, not user input

    with engine.connect() as connection:
        found_tables = connection.execute(discovery_query).fetchall()

    print(f"Found {len(found_tables)} columns across all tables")

    # Step 2: Group columns by table.
    grouped: dict = {}
    for database, schema, table, column, data_type, ordinal_position in found_tables:
        key = (database, schema, table)
        grouped.setdefault(key, []).append((column, data_type, ordinal_position))

    # Step 2b: Narrow to the changed tables when the streaming trigger scoped the run.
    if only_tables is not None:
        wanted = {tuple(ref) for ref in only_tables}
        grouped = {key: cols for key, cols in grouped.items() if key in wanted}
        missing = wanted - set(grouped)
        if missing:
            # A table was flagged as changed but is gone from information_schema by
            # the time we got here (dropped between detection and processing).
            print(f"{len(missing)} flagged tables no longer exist, skipping them")

    print(f"Processing {len(grouped)} tables")

    # Step 3: Profile each table's columns in batches.
    all_table_features = []

    for (database, schema, table), columns in grouped.items():
        print(f"Processing {database}.{schema}.{table} ({len(columns)} columns)")

        table_dfs = []
        batch_error = False

        for col_batch in chunks(columns, batch_size):
            union_parts = []
            for column, data_type, ordinal_position in col_batch:
                display_type = re.sub(r"^decimal\(.*\)$", "decimal", data_type, flags=re.IGNORECASE)
                union_parts.append(f"""
                    SELECT
                        '{database}' AS database,
                        '{schema}' AS schema,
                        '{table}' AS table_name,
                        '{column}' AS column_name,
                        '{display_type}' AS column_type,
                        COUNT(DISTINCT "{column}") AS number_unique_values,
                        COUNT(*) AS count,
                        COUNT_IF("{column}" IS NULL) AS null_count,
                        CASE
                            WHEN COUNT(*) > 0
                                THEN CAST(COUNT_IF("{column}" IS NULL) AS DOUBLE) / COUNT(*)
                                ELSE 0
                            END AS null_ratio,
                        CASE
                            WHEN COUNT(DISTINCT "{column}") = COUNT(*)
                                AND COUNT_IF("{column}" IS NULL) = 0 THEN 1
                                ELSE 0
                            END AS is_unique,
                        '{ordinal_position}' AS ordinal_position
                    FROM {database}.{schema}.{table}
                """)  # noqa: S608 - identifiers come from catalog metadata, not user input

            full_query = build_feature_query(" UNION ALL ".join(union_parts))

            try:
                with engine.connect() as connection:
                    result = connection.execute(text(full_query))
                    table_dfs.append(pd.DataFrame(result.fetchall(), columns=result.keys()))
            except Exception as e:  # noqa: BLE001 - skip a table if a batch fails, keep the rest
                orig = getattr(e, "orig", None)
                msg = str(orig) if orig else str(e).split("\n")[0]
                print(f"Error at {database}.{schema}.{table}: {msg}")
                batch_error = True
                break

        if batch_error or not table_dfs:
            continue

        df_table = pd.concat(table_dfs, ignore_index=True)
        df_table["ordinal_position"] = df_table["ordinal_position"].astype(int)

        # Recompute cross-batch stats first: it reads the raw column_type (e.g. for
        # table_integer_column_count), so it must run before one-hot encoding drops it.
        df_table = recompute_table_stats(df_table)

        # One-hot encode column_type, then guarantee all expected dummy columns exist.
        df_table = pd.get_dummies(df_table, columns=["column_type"], prefix="column_type")
        for col in EXPECTED_TYPE_COLS:
            if col not in df_table.columns:
                df_table[col] = 0
        # Store dummies as ints so they match the queue table's INTEGER columns.
        df_table[EXPECTED_TYPE_COLS] = df_table[EXPECTED_TYPE_COLS].astype(int)

        all_table_features.append(df_table)

    if not all_table_features:
        print("No data available - check tables and connection.")
        return 0

    df_final = pd.concat(all_table_features, ignore_index=True)
    df_final = df_final.sort_values(
        by=["database", "schema", "table_name", "ordinal_position"],
    ).reset_index(drop=True)

    print(f"Final features shape: {df_final.shape}")

    # Step 4: Write the feature table (rebuild each run, like a dbt table model).
    with engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS duckdb.staging"))
        df_final.to_sql(
            "stg_column_features",
            connection,
            schema="staging",
            if_exists="replace",
            index=False,
        )

    print(f"Wrote {len(df_final)} rows to duckdb.staging.stg_column_features")
    return len(df_final)


def _build_requeue_clause(
    requeue_tables: list[TableRef] | None,
) -> tuple[str, dict[str, str]]:
    """
    Build the SQL fragment that re-opens already-predicted columns.

    Without this a column is predicted exactly once, ever: the queue dedup skips
    anything with ``processed = TRUE``. When a table is reloaded, its data-dependent
    features (unique_ratio, null_ratio, table_row_count, …) change, so the stale
    prediction has to be recomputed rather than kept.

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


@task(name="populate-prediction-queue", cache_policy=NO_CACHE)
def populate_prediction_queue(requeue_tables: list[TableRef] | None = None) -> int:
    """
    Populate the prediction queue from the staging features table.

    A column is queued when it has no unprocessed row waiting AND either it was
    never predicted before, or its table appears in ``requeue_tables``.

    Args:
        requeue_tables: Tables whose already-predicted columns must be predicted
            again because their data changed. Supplied by the streaming trigger.

    Returns:
        Number of rows inserted into queue
    """
    requeue_clause, requeue_params = _build_requeue_clause(requeue_tables)
    trino_engine = get_trino_engine()

    with trino_engine.connect() as conn:
        # Note: id is INTEGER (predictions.py marks rows processed via int(id)), and the
        # numeric columns are BIGINT/DOUBLE to match the types pandas.to_sql writes to the
        # staging table, avoiding INSERT type-coercion errors.
        create_queue_table = text("""
            CREATE TABLE IF NOT EXISTS duckdb.predictions.queue (
                id INTEGER,
                database VARCHAR,
                schema VARCHAR,
                table_name VARCHAR,
                column_name VARCHAR,
                number_unique_values BIGINT,
                count BIGINT,
                null_count BIGINT,
                null_ratio DOUBLE,
                is_unique BIGINT,
                ordinal_position BIGINT,
                unique_ratio DOUBLE,
                is_non_null BIGINT,
                is_first_column BIGINT,
                relative_ordinal_position DOUBLE,
                is_first_unique_column BIGINT,
                table_column_count BIGINT,
                table_unique_column_count BIGINT,
                table_row_count BIGINT,
                other_unique_columns_in_table BIGINT,
                table_has_unique_column BIGINT,
                table_has_no_single_pk_candidate BIGINT,
                table_near_unique_column_count BIGINT,
                table_id_named_column_count BIGINT,
                table_non_null_column_count BIGINT,
                table_max_unique_ratio DOUBLE,
                table_integer_column_count BIGINT,
                unique_ratio_rank BIGINT,
                null_ratio_rank BIGINT,
                is_least_null_in_table BIGINT,
                unique_ratio_relative_to_max DOUBLE,
                other_near_unique_columns_in_table BIGINT,
                name_ends_with_id BIGINT,
                name_contains_key BIGINT,
                name_contains_table_name BIGINT,
                name_is_singular_table_id BIGINT,
                name_length BIGINT,
                column_type_boolean BIGINT,
                column_type_date BIGINT,
                column_type_decimal BIGINT,
                column_type_double BIGINT,
                column_type_integer BIGINT,
                column_type_varchar BIGINT,
                processed BOOLEAN,
                processed_at VARCHAR,
                created_at VARCHAR
            )
        """)

        # Migration guard: an existing queue created before the CPK feature expansion
        # lacks the 7 added columns. CREATE TABLE IF NOT EXISTS won't alter it, and the
        # column-explicit INSERT below would then fail. Drop the stale table so it is
        # recreated with the full schema. Previously-processed rows get re-queued and
        # re-predicted, which is what we want after a feature-schema change.
        queue_exists = (
            conn.execute(
                text("""
                    SELECT COUNT(*) FROM duckdb.information_schema.tables
                    WHERE table_schema = 'predictions' AND table_name = 'queue'
                """),
            ).scalar()
            or 0
        )
        has_new_columns = (
            conn.execute(
                text("""
                    SELECT COUNT(*) FROM duckdb.information_schema.columns
                    WHERE table_schema = 'predictions' AND table_name = 'queue'
                      AND column_name = 'table_integer_column_count'
                """),
            ).scalar()
            or 0
        )
        if queue_exists and not has_new_columns:
            print(
                "Queue table predates the CPK feature expansion (missing 7 columns); "
                "dropping and recreating. Previously-processed rows will be re-predicted.",
            )
            conn.execute(text("DROP TABLE duckdb.predictions.queue"))

        conn.execute(create_queue_table)
        print("Queue table verified/created")

        # Determine the id offset so new rows get unique, incrementing integer ids
        # across runs (Trino has no auto-increment / sequences).
        max_id = (
            conn.execute(
                text("SELECT COALESCE(MAX(id), 0) FROM duckdb.predictions.queue"),
            ).scalar()
            or 0
        )

        # Insert new rows from staging that aren't already processed in the queue.
        # Column list is explicit so the projection is matched by name, not position.
        insert_query = text(f"""
            INSERT INTO duckdb.predictions.queue (
                id, database, schema, table_name, column_name,
                number_unique_values, count, null_count, null_ratio, is_unique,
                ordinal_position, unique_ratio, is_non_null,
                is_first_column, relative_ordinal_position, is_first_unique_column,
                table_column_count, table_unique_column_count, table_row_count,
                other_unique_columns_in_table, table_has_unique_column,
                table_has_no_single_pk_candidate,
                table_near_unique_column_count, table_id_named_column_count,
                table_non_null_column_count, table_max_unique_ratio,
                table_integer_column_count, unique_ratio_rank,
                null_ratio_rank, is_least_null_in_table, unique_ratio_relative_to_max,
                other_near_unique_columns_in_table,
                name_ends_with_id, name_contains_key, name_contains_table_name,
                name_is_singular_table_id,
                name_length, column_type_boolean, column_type_date, column_type_decimal,
                column_type_double, column_type_integer, column_type_varchar,
                processed, processed_at, created_at
            )
            SELECT
                CAST({max_id} + ROW_NUMBER() OVER (
                    ORDER BY table_name, ordinal_position
                ) AS INTEGER) AS id,
                database,
                schema,
                table_name,
                column_name,
                number_unique_values,
                count,
                null_count,
                null_ratio,
                is_unique,
                ordinal_position,
                unique_ratio,
                is_non_null,
                is_first_column,
                relative_ordinal_position,
                is_first_unique_column,
                table_column_count,
                table_unique_column_count,
                table_row_count,
                other_unique_columns_in_table,
                table_has_unique_column,
                table_has_no_single_pk_candidate,
                table_near_unique_column_count,
                table_id_named_column_count,
                table_non_null_column_count,
                table_max_unique_ratio,
                table_integer_column_count,
                unique_ratio_rank,
                null_ratio_rank,
                is_least_null_in_table,
                unique_ratio_relative_to_max,
                other_near_unique_columns_in_table,
                name_ends_with_id,
                name_contains_key,
                name_contains_table_name,
                name_is_singular_table_id,
                name_length,
                column_type_boolean,
                column_type_date,
                column_type_decimal,
                column_type_double,
                column_type_integer,
                column_type_varchar,
                FALSE AS processed,
                CAST(NULL AS VARCHAR) AS processed_at,
                CAST(CURRENT_TIMESTAMP AS VARCHAR) AS created_at
            FROM duckdb.staging.stg_column_features AS f
            WHERE NOT EXISTS (
                SELECT 1
                FROM duckdb.predictions.queue AS q
                WHERE q.database = f.database
                  AND q.schema = f.schema
                  AND q.table_name = f.table_name
                  AND q.column_name = f.column_name
                  AND q.processed = FALSE
            )
            AND (
                NOT EXISTS (
                    SELECT 1
                    FROM duckdb.predictions.queue AS q
                    WHERE q.database = f.database
                      AND q.schema = f.schema
                      AND q.table_name = f.table_name
                      AND q.column_name = f.column_name
                      AND q.processed = TRUE
                )
                {requeue_clause}
            )
        """)  # noqa: S608 - max_id is an int and requeue_clause is built from bound parameters

        result = conn.execute(insert_query, requeue_params)
        rows_inserted = result.rowcount if hasattr(result, "rowcount") else 0

        print(f"Inserted {rows_inserted} new rows into prediction queue")

        status_query = text("""
            SELECT
                COUNT(*) as total_rows,
                SUM(CASE WHEN processed = FALSE THEN 1 ELSE 0 END) as unprocessed,
                SUM(CASE WHEN processed = TRUE THEN 1 ELSE 0 END) as processed
            FROM duckdb.predictions.queue
        """)

        status = conn.execute(status_query).fetchone()
        print(
            f"Queue status - Total: {status[0]}, Unprocessed: {status[1]}, Processed: {status[2]}",
        )

        return rows_inserted


@task(name="check-queue-status", cache_policy=NO_CACHE)
def check_queue_status() -> dict:
    trino_engine = get_trino_engine()

    with trino_engine.connect() as conn:
        query = text("""
            SELECT
                COUNT(*) as total_rows,
                SUM(CASE WHEN processed = FALSE THEN 1 ELSE 0 END) as unprocessed,
                SUM(CASE WHEN processed = TRUE THEN 1 ELSE 0 END) as processed
            FROM duckdb.predictions.queue
        """)

        result = conn.execute(query).fetchone()

        stats = {
            "total": result[0],
            "unprocessed": result[1],
            "processed": result[2],
        }

        # print("\n📊 Queue Status:")
        # print(f"   Total rows: {stats['total']}")
        # print(f"   Unprocessed: {stats['unprocessed']}")
        # print(f"   Processed: {stats['processed']}")

        return stats


@task(name="fetch-new-rows", cache_policy=NO_CACHE)
def fetch_new_rows(batch_size: int = 100) -> pd.DataFrame:
    """Fetch a batch of unprocessed rows from the prediction queue."""
    trino_engine = get_trino_engine()

    query = text("""
        SELECT
            id, database, schema, table_name, column_name,
            number_unique_values, count, null_count, null_ratio, is_unique,
            ordinal_position, unique_ratio, is_non_null,
            is_first_column, relative_ordinal_position, is_first_unique_column,
            table_column_count, table_unique_column_count, table_row_count,
            other_unique_columns_in_table, table_has_unique_column,
            table_has_no_single_pk_candidate,
            table_near_unique_column_count, table_id_named_column_count,
            table_non_null_column_count, table_max_unique_ratio,
            table_integer_column_count, unique_ratio_rank,
            null_ratio_rank, is_least_null_in_table, unique_ratio_relative_to_max,
            other_near_unique_columns_in_table,
            name_ends_with_id, name_contains_key, name_contains_table_name,
            name_is_singular_table_id,
            name_length, column_type_boolean, column_type_date, column_type_decimal,
            column_type_double, column_type_integer, column_type_varchar
        FROM duckdb.predictions.queue
        WHERE processed = FALSE
        LIMIT :batch_size
    """)

    with trino_engine.connect() as conn:
        return pd.read_sql(query, conn, params={"batch_size": batch_size})


@task(name="predict-batch", retries=2, retry_delay_seconds=10)
def predict_batch(rows: pd.DataFrame) -> list[dict]:
    """Call the pk/fk/cpk/cfk endpoints for each queued row."""
    predictions = []

    for _idx, row in rows.iterrows():
        row_data = row.to_dict()
        row_predictions = {"queue_id": row["id"]}

        # Drop metadata columns that are not model features.
        metadata = {
            "database": row_data.pop("database", None),
            "schema": row_data.pop("schema", None),
            "table_name": row_data.pop("table_name", None),
            "column_name": row_data.pop("column_name", None),
        }
        row_data.pop("id", None)

        all_predictions_successful = True
        for model_type in PREDICTION_MODELS:
            try:
                response = requests.post(
                    f"{MODEL_API_URL}/predict_{model_type}",
                    json=row_data,
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()

                prediction_value = result.get("prediction")
                row_predictions[f"{model_type}_prediction"] = prediction_value
                row_predictions[f"{model_type}_confidence"] = result.get("probability", 0.0)

                if prediction_value is None:
                    all_predictions_successful = False
                    print(f"Warning: {model_type} prediction returned None for row {row['id']}")
            except (requests.RequestException, requests.Timeout, requests.HTTPError) as e:
                print(f"Error predicting {model_type} for row {row['id']}: {e}")
                row_predictions[f"{model_type}_prediction"] = None
                row_predictions[f"{model_type}_confidence"] = 0.0
                all_predictions_successful = False

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


@task(name="store-predictions", cache_policy=NO_CACHE)
def store_predictions_to_trino(predictions: list[dict]) -> None:
    """Store all predictions, and mark a queue row processed only if all 4 models succeeded."""
    if not predictions:
        return

    trino_engine = get_trino_engine()

    successful_predictions = [p for p in predictions if p.get("all_predictions_successful")]
    failed_predictions = [p for p in predictions if not p.get("all_predictions_successful")]

    if failed_predictions:
        print("Failed queue_ids:")
        for p in failed_predictions:
            print(f"  - {p['queue_id']}")

    df = pd.DataFrame(predictions)
    if "all_predictions_successful" in df.columns:
        df = df.drop(columns=["all_predictions_successful"])

    # Reorder columns for readability, keeping only those present.
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
    df = df[[c for c in column_order if c in df.columns]]

    df["queue_id"] = df["queue_id"].astype(str)
    for col in ["pk_prediction", "fk_prediction", "cpk_prediction", "cfk_prediction"]:
        if col in df.columns:
            df[col] = df[col].fillna("NULL").astype(str)

    with trino_engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS duckdb.prediction_results"))
        df.to_sql(
            "key_results",
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
            UPDATE duckdb.predictions.queue
            SET processed = TRUE,
                processed_at = :current_timestamp
            WHERE id IN ({placeholders})
        """)  # noqa: S608 - placeholders are bound parameters, not user input
        conn.execute(update_query, params)
        print(f"Marked {len(successful_predictions)} rows as processed in queue")


@task(name="count-unprocessed-for-tables", cache_policy=NO_CACHE)
def count_unprocessed_for_tables(tables: list[TableRef]) -> int:
    """
    How many queue rows for these tables are still waiting to be predicted.

    This is the success test for a streaming run: the pending change may only be
    marked complete once its columns actually have predictions. Counting leftovers
    is more honest than counting attempts, because store_predictions_to_trino marks
    a row processed only when all four models answered.
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
        FROM duckdb.predictions.queue
        WHERE processed = FALSE
          AND ({" OR ".join(predicates)})
    """)  # noqa: S608 - predicates are built from bound parameters

    with get_trino_engine().connect() as conn:
        return conn.execute(query, params).scalar() or 0


@task(name="claim-key-changes", cache_policy=NO_CACHE)
def claim_key_changes(run_id: str) -> list[TableRef]:
    """Claim the tables the change detector flagged for the pk/fk track."""
    with get_trino_engine().begin() as conn:
        return claim_pending_changes(conn, TRACK_KEYS, run_id)


@task(name="close-key-changes", cache_policy=NO_CACHE)
def close_key_changes(run_id: str, *, succeeded: bool) -> int:
    """Complete the claim on success, release it on failure so the next run retries."""
    with get_trino_engine().begin() as conn:
        if succeeded:
            return complete_pending_changes(conn, TRACK_KEYS, run_id)
        return release_pending_changes(conn, TRACK_KEYS, run_id)


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

        print(f"\n🤖 Predicting up to {prediction_batch_size} of {unprocessed} unprocessed rows...")
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


@flow(name="key-prediction-pipeline")
def feature_engineering_pipeline(
    target_schemas: list[str] | None = None,
    batch_size: int = 30,
    prediction_batch_size: int = 100,
    use_pending_changes: bool = False,
    drain_queue: bool = False,
) -> dict:
    """
    Complete pipeline: Extract features → Queue → Predict

    Args:
        target_schemas: Schemas to scan for features (default: ['new_predict_data']).
        batch_size: Column batch size for the feature extraction queries.
        prediction_batch_size: Batch size for a single prediction round.
        use_pending_changes: Streaming mode. Only process the tables the change
            detector flagged, and re-predict their already-processed columns.
            False (default) keeps the original full-schema scan.
        drain_queue: Keep predicting until the queue backlog is empty instead of
            stopping after one batch. Implied by ``use_pending_changes``, because a
            change is only marked complete once its columns are actually predicted.
    """
    if target_schemas is None:
        target_schemas = ["new_predict_data"]

    only_tables: list[TableRef] | None = None
    run_id = str(flow_run.get_id())

    if use_pending_changes:
        drain_queue = True
        print("\n📋 Step 0: Claiming pending changes for the pk/fk track...")
        only_tables = claim_key_changes(run_id)
        if not only_tables:
            # Normal case for a duplicate/late event - the work was already taken.
            print("No pending changes to process - nothing to do.")
            return {"rows_written": 0, "rows_queued": 0, "predictions_made": 0, "tables": 0}
        print(f"Claimed {len(only_tables)} changed tables")

    succeeded = False
    try:
        # Step 1: Extract features into duckdb.staging.stg_column_features
        print("\n📊 Step 1: Extracting features from information_schema...")
        rows_written = extract_features(
            target_schemas=target_schemas,
            batch_size=batch_size,
            only_tables=only_tables,
        )

        if rows_written == 0:
            print("\n⚠️  No features extracted - stopping pipeline.")
            # Nothing to predict, but the claim is genuinely handled: the flagged
            # tables are unreadable or gone, so retrying them would loop forever.
            succeeded = True
            return {"rows_written": 0, "rows_queued": 0, "predictions_made": 0}

        # Step 2: Populate the prediction queue
        print("\n📥 Step 2: Populating prediction queue...")
        rows_inserted = populate_prediction_queue(requeue_tables=only_tables)

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
                print(f"\n❌ {msg}")
                raise RuntimeError(msg)

        succeeded = True
    finally:
        if use_pending_changes:
            closed = close_key_changes(run_id, succeeded=succeeded)
            verb = "completed" if succeeded else "released for retry"
            print(f"{closed} pending-change rows {verb}")

    print("Feature Engineering Pipeline Complete!")

    return {
        "rows_written": rows_written,
        "rows_queued": rows_inserted,
        "queue_stats": check_queue_status(),
        "predictions_made": predictions_made,
    }


if __name__ == "__main__":
    feature_engineering_pipeline(
        target_schemas=["new_predict_data"],
        batch_size=30,
        prediction_batch_size=100,
    )
