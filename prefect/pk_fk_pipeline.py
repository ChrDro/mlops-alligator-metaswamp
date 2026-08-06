"""
Prefect flow for orchestrating the feature engineering and prediction pipeline.

This flow follows 3 steps:
1. Extracts column-level features from information_schema (plain Python/pandas)
   and writes them to iceberg.staging.stg_column_features in Trino.
2. Populates the prediction queue from the extracted features.
3. Triggers the prediction flow to call the ML model APIs and store results.

Note: feature extraction used to be a dbt Python model, but dbt-trino does not
support Python models (Trino has no Python runtime). The model also bypassed the
dbt session entirely and used its own SQLAlchemy connection, so it has been moved
here as a normal Prefect task with no loss of functionality.
"""

import importlib.util
import os
import re
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pandas as pd
import requests
import urllib3
from change_events import (
    TRACK_KEYS,
    TableRef,
    claim_pending_changes,
    complete_pending_changes,
    release_pending_changes,
    with_commit_retry,
)
from dotenv import load_dotenv
from model_health import diagnose, failure_detail
from prefect.cache_policies import NONE as NO_CACHE
from prefect.runtime import flow_run
from sqlalchemy import Connection, Engine, bindparam, create_engine, text

from prefect import flow, task


def _load_cross_table_features() -> ModuleType:
    """Load src/cross_table_features.py by path, without touching sys.path.

    The fk model's cross-table features have to be computed by the exact module the
    training script used - a second copy inside prefect/ would drift, and a feature
    computed differently at training and serving time is the silent failure
    test/test_models/test_model_schema_contract.py exists to catch.

    Importing it as `src.cross_table_features` would need the repo root on sys.path, and
    the repo root contains a directory called `prefect` that shadows the installed Prefect
    library - see the import-path note in test/test_pipelines/conftest.py. Loading the file
    directly avoids that trap.

    Two locations are tried: the repo layout (src/ next to prefect/) and /opt/src, where
    docker-compose.yaml bind-mounts it in the container.
    """
    for directory in (Path(__file__).resolve().parents[1] / "src", Path("/opt/src")):
        module_path = directory / "cross_table_features.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location("cross_table_features", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    msg_features_missing = (
        "cross_table_features.py not found next to prefect/ or at /opt/src. "
        "The fk model needs it; check the ./src bind mount in docker-compose.yaml."
    )
    raise ImportError(msg_features_missing)


_cross_table = _load_cross_table_features()
CROSS_TABLE_FEATURES: list[str] = _cross_table.CROSS_TABLE_FEATURES
add_cross_table_features = _cross_table.add_cross_table_features


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

# The reference scope for the fk model's cross-table features. `database` here is the
# Trino catalog, so the schema is what actually bounds a set of related tables. The model
# is trained on several scope widths (see src/cross_table_features.py) precisely because
# this scope cannot be assumed to hold one logical database.
SERVING_SCOPE_COLUMNS = ("database", "schema")

# Identifies one column across staging, the queue and information_schema.
UNIQUENESS_KEY = ["database", "schema", "table_name", "column_name"]

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
        f"trino://{TRINO_USERNAME}:{TRINO_PASSWORD}@{TRINO_IP_ADDRESS}:8443/iceberg",
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

    # Must match the ROW_NUMBER() ordering in the extraction SQL above
    # (null_ratio ASC, ordinal_position ASC), because this recompute overwrites the
    # value the SQL produced. Until 30.07. the null_ratio key was missing here, which
    # silently degraded null_ratio_rank to "position in the table" and
    # is_least_null_in_table to "is the first column".
    rank_idx = df_table.sort_values(
        ["null_ratio", "ordinal_position"],
        ascending=[True, True],
    ).index
    df_table["null_ratio_rank"] = pd.Series(range(1, len(rank_idx) + 1), index=rank_idx)

    df_table["is_least_null_in_table"] = (df_table["null_ratio_rank"] == 1).astype(int)
    df_table["unique_ratio_relative_to_max"] = (
        (df_table["unique_ratio"] / table_max_unique_ratio) if table_max_unique_ratio > 0 else 0.0
    )

    return df_table


def _known_uniqueness(engine: Engine, databases: set[str], schemas: set[str]) -> pd.DataFrame:
    """`is_unique` per column from the queue, for tables this run did not profile.

    The queue is the only durable record of a column's uniqueness: staging is rebuilt every
    run and a streaming run rebuilds it with just the changed tables. Without this lookup a
    streaming run would see a one-table reference scope and score 0 on every cross-table
    feature, which reads to the model as "this database has no relationships at all".

    A requeued column can appear more than once, so the newest row wins.
    """
    empty = pd.DataFrame(columns=[*UNIQUENESS_KEY, "is_unique"])
    if not databases or not schemas:
        return empty

    query = text("""
        SELECT database, schema, table_name, column_name, is_unique, created_at
        FROM iceberg.predictions.queue
        WHERE database IN :databases AND schema IN :schemas
    """).bindparams(
        bindparam("databases", value=tuple(databases), expanding=True),
        bindparam("schemas", value=tuple(schemas), expanding=True),
    )

    try:
        with engine.connect() as connection:
            known = pd.read_sql(query, connection)
    except Exception as e:  # noqa: BLE001 - a missing queue on a first run is not an error
        orig = getattr(e, "orig", None)
        print(f"No previous uniqueness available ({orig or str(e).split(chr(10))[0]})")
        return empty

    return (
        known.sort_values("created_at")
        .drop_duplicates(subset=UNIQUENESS_KEY, keep="last")
        .drop(columns=["created_at"])
    )


def add_serving_cross_table_features(
    df_final: pd.DataFrame,
    discovered_columns: list[tuple[str, str, str, str]],
    engine: Engine,
) -> pd.DataFrame:
    """Attach the fk model's cross-table features, scoped to the whole schema.

    The reference scope must be every table of the schema, not just the tables this run
    profiled: `n_other_tables_with_same_column_name` and friends are about the *other*
    tables, and a column whose parent table was not in this run has to find it anyway.

    Uniqueness comes from this run where available and from the queue otherwise. Columns
    that have never been profiled contribute their names but count as not unique, which is
    the safe default - it can only fail to find a parent, never invent one.

    Args:
        df_final: The rows profiled in this run, already carrying `is_unique`.
        discovered_columns: (database, schema, table_name, column_name) for every column of
            the target schemas, from information_schema and before any table filtering.
        engine: Trino connection, for the queue lookup.

    Returns:
        `df_final` with `CROSS_TABLE_FEATURES` appended, same rows in the same order.
    """
    reference = pd.DataFrame(discovered_columns, columns=UNIQUENESS_KEY)

    uniqueness = _known_uniqueness(
        engine,
        databases=set(reference["database"]),
        schemas=set(reference["schema"]),
    )
    reference = reference.merge(uniqueness, on=UNIQUENESS_KEY, how="left")
    # to_numeric before fillna: the merge can leave an object-dtype column when the queue
    # lookup came back empty, and .fillna on object dtype is deprecated in pandas 2.x.
    reference["is_unique"] = (
        pd.to_numeric(reference["is_unique"], errors="coerce").fillna(0).astype(int)
    )

    # Rows profiled in this run are authoritative, so their stale reference copies drop out.
    unprofiled = reference.merge(
        df_final[UNIQUENESS_KEY].assign(_in_run=1), on=UNIQUENESS_KEY, how="left"
    )
    unprofiled = unprofiled[unprofiled["_in_run"].isna()].drop(columns=["_in_run"])

    scope_sizes = reference.groupby(list(SERVING_SCOPE_COLUMNS))["table_name"].nunique()
    print(
        f"Cross-table reference scope: {len(reference)} columns / "
        f"{reference['table_name'].nunique()} tables across {len(scope_sizes)} scope(s); "
        f"largest scope has {int(scope_sizes.max())} tables. "
        f"{len(unprofiled)} columns contribute names only.",
    )
    unknown = int((unprofiled["is_unique"] == 0).sum()) if not unprofiled.empty else 0
    if unknown:
        print(f"  {unknown} reference columns have no recorded uniqueness (treated as 0)")

    # df_final first, so the profiled rows keep their positions after featurising.
    combined = pd.concat([df_final, unprofiled], ignore_index=True)
    featurised = add_cross_table_features(combined, scope_columns=SERVING_SCOPE_COLUMNS)

    result = featurised.iloc[: len(df_final)].copy()
    # The queue stores these as BIGINT; a float column would be rejected on insert.
    result[CROSS_TABLE_FEATURES] = result[CROSS_TABLE_FEATURES].astype(int)
    return result


@task(name="extract-features", retries=2, retry_delay_seconds=30, cache_policy=NO_CACHE)
def extract_features(
    target_schemas: list[str],
    batch_size: int = 30,
    only_tables: list[TableRef] | None = None,
    skip_tables: set[tuple[str, str, str]] | None = None,
) -> int:
    """
    Extract column-level features from information_schema and write them to
    iceberg.staging.stg_column_features (rebuilt each run).

    Args:
        target_schemas: Schemas to scan for tables/columns.
        batch_size: Number of columns to profile per Trino query.
        only_tables: Restrict profiling to these tables. Used by the streaming
            trigger so a run only pays for what actually changed instead of
            re-profiling the whole schema. None = profile everything (the
            original behaviour, kept for manual/full runs).
        skip_tables: (database, schema, table_name) triples whose columns are all
            queued already, so profiling them would produce nothing the queue
            dedup does not immediately discard. Mirrors the normalform track's
            skip_tables; see get_fully_queued_tables for how the set is built.
            Ignored when only_tables is given - see the filter below for why.

    Returns:
        Number of feature rows written.
    """
    skip_tables = skip_tables or set()
    engine = get_trino_engine()

    for schema in target_schemas:
        with engine.connect() as conn:
            create_schema = text(f"""
                CREATE SCHEMA IF NOT EXISTS iceberg.{schema}
            """)

            conn.execute(create_schema)

    # Step 1: Discover all tables and columns in the target schemas.
    schema_filter = "', '".join(target_schemas)
    discovery_query = text(f"""
        SELECT t.table_catalog, t.table_schema, t.table_name,
               c.column_name, c.data_type, c.ordinal_position
        FROM iceberg.information_schema.tables AS t
        INNER JOIN iceberg.information_schema.columns AS c
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
        # Streaming mode: the change detector already decided what needs work, so the
        # already-queued filter must NOT apply - a reloaded table is meant to be
        # profiled again even though the queue holds older rows for its columns.
        wanted = {tuple(ref) for ref in only_tables}
        grouped = {key: cols for key, cols in grouped.items() if key in wanted}
        missing = wanted - set(grouped)
        if missing:
            # A table was flagged as changed but is gone from information_schema by
            # the time we got here (dropped between detection and processing).
            print(f"{len(missing)} flagged tables no longer exist, skipping them")
    elif skip_tables:
        before = len(grouped)
        grouped = {key: cols for key, cols in grouped.items() if key not in skip_tables}
        print(f"Skipping {before - len(grouped)} already-queued tables")

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
        # Distinguish the healthy case (everything was filtered out, which is the
        # normal outcome of a full-scan run once every table is queued) from the
        # genuine fault, so the steady state does not print like a failure.
        if not grouped:
            print("No tables left to profile after filtering - nothing to extract.")
        else:
            print("No data available - check tables and connection.")
        return 0

    df_final = pd.concat(all_table_features, ignore_index=True)
    df_final = df_final.sort_values(
        by=["database", "schema", "table_name", "ordinal_position"],
    ).reset_index(drop=True)

    # Cross-table features come last because they are the only ones that look outside the
    # table being profiled, and they need the whole schema in view - `found_tables` is that
    # view, taken before only_tables/skip_tables narrowed this run down.
    df_final = add_serving_cross_table_features(
        df_final,
        discovered_columns=[
            (database, schema, table, column)
            for database, schema, table, column, _type, _pos in found_tables
        ],
        engine=engine,
    )

    print(f"Final features shape: {df_final.shape}")

    # Step 4: Write the feature table (rebuild each run, like a dbt table model).
    with engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS iceberg.staging"))
        df_final.to_sql(
            "stg_column_features",
            connection,
            schema="staging",
            if_exists="replace",
            index=False,
        )

    print(f"Wrote {len(df_final)} rows to iceberg.staging.stg_column_features")
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

        conn.execute(text("CREATE SCHEMA IF NOT EXISTS iceberg.predictions"))

        create_queue_table = text("""
            CREATE TABLE IF NOT EXISTS iceberg.predictions.queue (
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
                n_other_tables_with_same_column_name BIGINT,
                name_unique_in_other_table BIGINT,
                is_non_unique_and_name_unique_elsewhere BIGINT,
                name_references_other_table_exact BIGINT,
                name_references_other_table_fuzzy BIGINT,
                name_ends_with_id_no_underscore BIGINT,
                name_ends_with_code_or_num BIGINT,
                processed BOOLEAN,
                processed_at VARCHAR,
                created_at VARCHAR
            )
        """)

        # Migration guard: a queue created before a feature expansion lacks the added
        # columns. CREATE TABLE IF NOT EXISTS won't alter it, and the column-explicit
        # INSERT below would then fail. Drop the stale table so it is recreated with the
        # full schema. Previously-processed rows get re-queued and re-predicted, which is
        # what we want after a feature-schema change.
        #
        # The sentinel is the newest added column, so this guard covers every expansion:
        # first the CPK one (table_integer_column_count), now the fk cross-table features.
        queue_exists = (
            conn.execute(
                text("""
                    SELECT COUNT(*) FROM iceberg.information_schema.tables
                    WHERE table_schema = 'predictions' AND table_name = 'queue'
                """),
            ).scalar()
            or 0
        )
        has_new_columns = (
            conn.execute(
                text("""
                    SELECT COUNT(*) FROM iceberg.information_schema.columns
                    WHERE table_schema = 'predictions' AND table_name = 'queue'
                      AND column_name = 'name_ends_with_code_or_num'
                """),
            ).scalar()
            or 0
        )
        if queue_exists and not has_new_columns:
            print(
                "Queue table predates the fk cross-table feature expansion (missing 7 "
                "columns); dropping and recreating. Previously-processed rows will be "
                "re-predicted.",
            )
            conn.execute(text("DROP TABLE iceberg.predictions.queue"))

        conn.execute(create_queue_table)
        print("Queue table verified/created")

        # Determine the id offset so new rows get unique, incrementing integer ids
        # across runs (Trino has no auto-increment / sequences).
        max_id = (
            conn.execute(
                text("SELECT COALESCE(MAX(id), 0) FROM iceberg.predictions.queue"),
            ).scalar()
            or 0
        )

        # Insert new rows from staging that aren't already processed in the queue.
        # Column list is explicit so the projection is matched by name, not position.
        insert_query = text(f"""
            INSERT INTO iceberg.predictions.queue (
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
                n_other_tables_with_same_column_name, name_unique_in_other_table,
                is_non_unique_and_name_unique_elsewhere,
                name_references_other_table_exact, name_references_other_table_fuzzy,
                name_ends_with_id_no_underscore, name_ends_with_code_or_num,
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
                n_other_tables_with_same_column_name,
                name_unique_in_other_table,
                is_non_unique_and_name_unique_elsewhere,
                name_references_other_table_exact,
                name_references_other_table_fuzzy,
                name_ends_with_id_no_underscore,
                name_ends_with_code_or_num,
                FALSE AS processed,
                CAST(NULL AS VARCHAR) AS processed_at,
                CAST(CURRENT_TIMESTAMP AS VARCHAR) AS created_at
            FROM iceberg.staging.stg_column_features AS f
            WHERE NOT EXISTS (
                SELECT 1
                FROM iceberg.predictions.queue AS q
                WHERE q.database = f.database
                  AND q.schema = f.schema
                  AND q.table_name = f.table_name
                  AND q.column_name = f.column_name
                  AND q.processed = FALSE
            )
            AND (
                NOT EXISTS (
                    SELECT 1
                    FROM iceberg.predictions.queue AS q
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
            FROM iceberg.predictions.queue
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
            FROM iceberg.predictions.queue
        """)

        result = conn.execute(query).fetchone()

        stats = {
            "total": result[0],
            "unprocessed": result[1],
            "processed": result[2],
        }

        # print("Queue Status:")
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
            column_type_double, column_type_integer, column_type_varchar,
            n_other_tables_with_same_column_name, name_unique_in_other_table,
            is_non_unique_and_name_unique_elsewhere,
            name_references_other_table_exact, name_references_other_table_fuzzy,
            name_ends_with_id_no_underscore, name_ends_with_code_or_num
        FROM iceberg.predictions.queue
        WHERE processed = FALSE
        LIMIT :batch_size
    """)

    with trino_engine.connect() as conn:
        return pd.read_sql(query, conn, params={"batch_size": batch_size})


@task(name="predict-batch", retries=2, retry_delay_seconds=10)
def predict_batch(rows: pd.DataFrame) -> tuple[list[dict], list[str]]:
    """
    Call the pk/fk/cpk/cfk endpoints for each queued row.

    Returns the predictions plus the distinct reasons calls failed, so a run that
    predicted nothing can say why instead of only that it happened.
    """
    predictions = []
    failure_reasons: set[str] = set()

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
                    failure_reasons.add(f"{model_type}: the service answered with no prediction")
            except (requests.RequestException, requests.Timeout, requests.HTTPError) as e:
                # The status line alone ("400 Bad Request") is the one part of the
                # response that carries no information - keep what the service said.
                detail = failure_detail(e)
                print(f"Error predicting {model_type} for row {row['id']}: {detail}")
                failure_reasons.add(detail)
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

    return predictions, sorted(failure_reasons)


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
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS iceberg.prediction_results"))
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
            UPDATE iceberg.predictions.queue
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
        FROM iceberg.predictions.queue
        WHERE processed = FALSE
          AND ({" OR ".join(predicates)})
    """)  # noqa: S608 - predicates are built from bound parameters

    with get_trino_engine().connect() as conn:
        return conn.execute(query, params).scalar() or 0


@task(name="get-fully-queued-tables", cache_policy=NO_CACHE)
def get_fully_queued_tables(target_schemas: list[str]) -> set[tuple[str, str, str]]:
    """
    Return the tables whose every current column already has a queue row.

    These are exactly the tables that profiling would gain nothing from on a
    full-scan run: populate_prediction_queue's dedup discards any column that
    already has a row (unprocessed, or processed and not flagged for requeue), so
    the COUNT(DISTINCT) work would be paid and then thrown away. This mirrors the
    normalform track's get_processed_tables, which skips at table grain because it
    stores one row per table.

    Matching on *columns* rather than tables is what makes this safe: a table that
    gained a column has an unmatched column and is therefore not skipped, so the
    new column still gets profiled and queued.

    Returns an empty set when the queue does not exist yet (first ever run).
    """
    engine = get_trino_engine()
    schema_filter = "', '".join(target_schemas)

    with engine.connect() as conn:
        queue_exists = conn.execute(
            text("""
                SELECT COUNT(*) FROM iceberg.information_schema.tables
                WHERE table_schema = 'predictions' AND table_name = 'queue'
            """),
        ).scalar()
        if not queue_exists:
            print("No queue table yet - nothing to skip.")
            return set()

        # LEFT JOIN + "no column went unmatched": a column with several queue rows
        # (one per reload) multiplies the join, which the COUNT_IF is unaffected by.
        rows = conn.execute(
            text(f"""
                SELECT c.table_catalog, c.table_schema, c.table_name
                FROM iceberg.information_schema.columns AS c
                LEFT JOIN iceberg.predictions.queue AS q
                    ON q.database = c.table_catalog
                    AND q.schema = c.table_schema
                    AND q.table_name = c.table_name
                    AND q.column_name = c.column_name
                WHERE c.table_schema IN ('{schema_filter}')
                GROUP BY c.table_catalog, c.table_schema, c.table_name
                HAVING COUNT_IF(q.column_name IS NULL) = 0
            """),  # noqa: S608 - schema names are operator-supplied config, not user input
        ).fetchall()

    return {(r[0], r[1], r[2]) for r in rows}


@task(name="count-unprocessed-total", cache_policy=NO_CACHE)
def count_unprocessed_total() -> int:
    """
    How many queue rows are waiting to be predicted, across all tables.

    Used to decide whether "nothing new to profile" also means "nothing to do".
    It does not: an earlier run may have queued rows the model service was too
    broken to predict, and those must still be drained. Tolerates a missing queue
    table so a first run on an empty schema does not fail here.
    """
    engine = get_trino_engine()

    with engine.connect() as conn:
        queue_exists = conn.execute(
            text("""
                SELECT COUNT(*) FROM iceberg.information_schema.tables
                WHERE table_schema = 'predictions' AND table_name = 'queue'
            """),
        ).scalar()
        if not queue_exists:
            return 0

        return (
            conn.execute(
                text("SELECT COUNT(*) FROM iceberg.predictions.queue WHERE processed = FALSE"),
            ).scalar()
            or 0
        )


@task(name="claim-key-changes", cache_policy=NO_CACHE)
def claim_key_changes(run_id: str) -> list[TableRef]:
    """
    Claim the tables the change detector flagged for the pk/fk track.

    Retried on commit conflicts: all three tracks are woken by the same event and
    claim the same rows, so on iceberg two of the three lose the commit race.
    """
    return with_commit_retry(
        get_trino_engine(),
        lambda conn: claim_pending_changes(conn, TRACK_KEYS, run_id),
    )


@task(name="close-key-changes", cache_policy=NO_CACHE)
def close_key_changes(run_id: str, *, succeeded: bool) -> int:
    """Complete the claim on success, release it on failure so the next run retries."""

    def close(conn: Connection) -> int:
        if succeeded:
            return complete_pending_changes(conn, TRACK_KEYS, run_id)
        return release_pending_changes(conn, TRACK_KEYS, run_id)

    return with_commit_retry(get_trino_engine(), close)


def _drain_queue(prediction_batch_size: int, drain_queue: bool) -> tuple[int, list[str]]:
    """
    Predict unprocessed queue rows and return how many predictions were made,
    together with the distinct reasons any calls failed.

    With ``drain_queue`` the loop keeps going until the backlog is empty. It stops
    early when a batch fails to reduce the unprocessed count - rows whose model call
    failed stay ``processed = FALSE``, so without that guard the loop would spin on
    the same failing batch forever.
    """
    predictions_made = 0
    previous_unprocessed = None
    failure_reasons: set[str] = set()

    while True:
        queue_stats = check_queue_status()
        unprocessed = queue_stats["unprocessed"] or 0

        if unprocessed == 0:
            print("All rows in queue already processed")
            break

        if previous_unprocessed is not None and unprocessed >= previous_unprocessed:
            print(
                f"Stopping: {unprocessed} rows still unprocessed after a full batch, "
                f"because their model calls are failing.\n"
                f"{diagnose(failure_reasons, MODEL_API_URL)}",
            )
            break
        previous_unprocessed = unprocessed

        print(f"Predicting up to {prediction_batch_size} of {unprocessed} unprocessed rows...")
        start_time = time.time()
        rows = fetch_new_rows(prediction_batch_size)
        if rows.empty:
            break

        predictions, batch_failures = predict_batch(rows)
        failure_reasons.update(batch_failures)
        store_predictions_to_trino(predictions)
        predictions_made += len(predictions)
        print(f"Processed {len(predictions)} predicts in {round(time.time() - start_time, 1)}s")

        if not drain_queue:
            break

    return predictions_made, sorted(failure_reasons)


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
    skip_tables: set[tuple[str, str, str]] | None = None
    run_id = str(flow_run.get_id())

    if use_pending_changes:
        drain_queue = True
        print("Step 0: Claiming pending changes for the pk/fk track...")
        only_tables = claim_key_changes(run_id)
        if not only_tables:
            # Normal case for a duplicate/late event - the work was already taken.
            print("No pending changes to process - nothing to do.")
            return {"rows_written": 0, "rows_queued": 0, "predictions_made": 0, "tables": 0}
        print(f"Claimed {len(only_tables)} changed tables")
    else:
        # Full-scan mode: don't re-profile tables whose columns are all queued
        # already. The queue dedup would discard those rows anyway, so the
        # COUNT(DISTINCT) per column would be paid for nothing.
        print("Step 0: Checking which tables are already fully queued...")
        skip_tables = get_fully_queued_tables(target_schemas)
        print(f"{len(skip_tables)} tables are fully queued already")

    succeeded = False
    try:
        # Step 1: Extract features into iceberg.staging.stg_column_features
        print("Step 1: Extracting features from information_schema...")
        rows_written = extract_features(
            target_schemas=target_schemas,
            batch_size=batch_size,
            only_tables=only_tables,
            skip_tables=skip_tables,
        )

        rows_inserted = 0
        if rows_written:
            # Step 2: Populate the prediction queue
            print("Step 2: Populating prediction queue...")
            rows_inserted = populate_prediction_queue(requeue_tables=only_tables)
        elif use_pending_changes:
            print("No features extracted - stopping pipeline.")
            # Nothing to predict, but the claim is genuinely handled: the flagged
            # tables are unreadable or gone, so retrying them would loop forever.
            succeeded = True
            return {"rows_written": 0, "rows_queued": 0, "predictions_made": 0}
        else:
            # Everything was skipped, which is the healthy steady state of a full
            # run. It does NOT mean there is nothing to do: an earlier run may have
            # queued rows that the model service was too broken to predict, and
            # those still need draining. Returning here would strand them until a
            # table happened to change.
            backlog = count_unprocessed_total()
            if backlog == 0:
                print("Nothing new to profile and no backlog - nothing to do.")
                succeeded = True
                return {"rows_written": 0, "rows_queued": 0, "predictions_made": 0}
            print(f"Nothing new to profile; draining {backlog} unprocessed rows.")

        # Step 3: Predict unprocessed rows and store the results.
        predictions_made, failure_reasons = _drain_queue(prediction_batch_size, drain_queue)

        # Step 4: A run only counts as done once the claimed tables actually have
        # predictions. Without this check a run whose every model call returned an
        # error would still mark its changes complete, and the retry path built into
        # the detector would never see them again - the work would vanish silently.
        if use_pending_changes:
            leftover = count_unprocessed_for_tables(only_tables)
            if leftover:
                # Lead with the cause: retriggering is pointless until it is fixed,
                # and the row count alone has sent people looking at the queue.
                msg = (
                    f"Nothing could be predicted - {leftover} queue "
                    f"{'row' if leftover == 1 else 'rows'} for the claimed tables are "
                    f"still unpredicted.\n"
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
