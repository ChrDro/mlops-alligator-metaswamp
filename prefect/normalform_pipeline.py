"""
Prefect flow for the normal-form (denormalization) prediction track.

Separate from the pk/fk/cpk/cfk pipeline on purpose (see
documentation/NORMALFORM_PREDICTION.md):
- the model (denormalization_model) is MULTICLASS 0-3 (0=violates 1NF, 1/2/3=pure NF),
- the label is a TABLE-level property inferred from per-column feature rows,
- it uses a different, larger (50) feature set.

This flow:
1. Discovers + profiles columns in the target schema(s) and builds the exact 50
   features the registered model expects (feature logic ported verbatim from
   get_trino_summaries_task_3_training.py so inference matches training).
2. Calls POST /predict_normalform per column via the model API.
3. Aggregates per table by MAJORITY VOTE -> one normal-form class per table.
4. Stores ONE row per table in duckdb.normalform_predictions.results.

Retraining is intentionally out of scope: the training data is frozen and was
mostly hand-labeled (no label generator exists in the repo).
"""

import json
import os
import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import requests
import urllib3
from dotenv import load_dotenv
from prefect.cache_policies import NONE as NO_CACHE
from sqlalchemy import Engine, create_engine, text

from prefect import flow, task


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

TRINO_IP_ADDRESS = str(os.environ.get("TRINO_IP_ADDRESS"))
TRINO_USERNAME = str(os.environ.get("TRINO_USERNAME"))
TRINO_PASSWORD = str(os.environ.get("TRINO_PASSWORD"))

# Model API base (use localhost outside Docker, model-service name inside).
MODEL_API_URL = os.environ.get("MODEL_API_URL", "http://localhost:8080")

# The column_type dummies the normalform model was trained with (drop_first=True
# reference category is anything NOT in this set -> all-zero dummies).
EXPECTED_TYPE_COLS = [
    "column_type_char",
    "column_type_date",
    "column_type_decimal",
    "column_type_double",
    "column_type_integer",
    "column_type_timestamp",
    "column_type_varchar",
]

# The 50 features the registered model / NormalForm pydantic schema expect.
_FEATURES_JSON = (
    Path(__file__).resolve().parent / ".." / ".." / "features_normalform.json"
).resolve()
with _FEATURES_JSON.open() as f:
    MODEL_FEATURES: list[str] = json.load(f)["features"]

# Features the API expects as float (NormalForm pydantic); everything else numeric
# is int, and column_type_* are bool.
FLOAT_FEATURES = {
    "null_ratio",
    "unique_ratio",
    "relative_ordinal_position",
    "table_max_unique_ratio",
    "unique_ratio_relative_to_max",
    "table_avg_unique_ratio",
    "table_avg_null_ratio",
    "table_ratio_of_pk_candidates",
    "table_ratio_composite_key_cols",
    "table_ratio_1nf_violations",
    "table_std_unique_ratio",
}


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
    """
    Build the normalform feature query (table stats + ranks + per-column features).
    Ported from get_trino_summaries_task_3_training.py, minus the training-only
    columns (target_normal_form, is_unique_non_null).
    """
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
            END)                                 AS table_integer_column_count,
        AVG(
            CASE WHEN count > 0
                THEN CAST(number_unique_values AS DOUBLE) / count
                ELSE 0
            END
        )                                        AS table_avg_unique_ratio,
        AVG(null_ratio)                          AS table_avg_null_ratio,
        CASE
            WHEN MAX(CASE WHEN number_unique_values = count AND null_count = 0
                          THEN 1 ELSE 0 END) = 0
            AND SUM(CASE WHEN LOWER(column_name) LIKE '%\\_id' ESCAPE '\\'
                         THEN 1 ELSE 0 END) >= 2
            THEN 1 ELSE 0
        END                                      AS table_has_composite_pk
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
    LENGTH(b.column_name)                                      AS name_length,
    CASE
        WHEN LOWER(b.column_name) LIKE '%id%'
        AND CASE WHEN t.table_has_unique_column = 0 THEN 1 ELSE 0 END = 1
        THEN 1 ELSE 0
    END                                                      AS is_composite_key_part,
    CAST(t.table_unique_column_count AS DOUBLE)
        / NULLIF(t.table_column_count, 0)                   AS table_ratio_of_pk_candidates,
    t.table_avg_unique_ratio,
    t.table_avg_null_ratio,
    b.is_this_col_violating_1nf,
    t.table_has_composite_pk
FROM base AS b
CROSS JOIN table_stats AS t
LEFT JOIN column_ranks AS r
    ON b.column_name = r.column_name
    AND b.ordinal_position = r.ordinal_position
ORDER BY CAST(b.ordinal_position AS INTEGER)
"""  # noqa: S608 - union_sql is built from catalog metadata, not user input


def recompute_table_features(df_table: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute table-level stats/rankings across all batched columns and derive the
    FD-style features. Ported verbatim from the training script so inference matches
    how the model was trained.
    """
    table_col_count = int(df_table["ordinal_position"].max())
    table_unique_col_count = int(df_table["is_unique"].sum())
    table_row_count = int(df_table["count"].max())
    table_has_unique = int((df_table["is_unique"] == 1).any())
    table_near_unique = int((df_table["unique_ratio"] > 0.95).sum())
    table_id_named = int(df_table["name_ends_with_id"].sum())
    table_non_null = int((df_table["null_count"] == 0).sum())
    table_max_unique_ratio = float(df_table["unique_ratio"].max())
    table_int_col = int(df_table["column_type"].isin(["bigint", "integer", "int"]).sum())
    table_has_composite_pk = int(table_has_unique == 0 and table_id_named >= 2)
    table_avg_unique_ratio = float(df_table["unique_ratio"].mean())
    table_avg_null_ratio = float(df_table["null_ratio"].mean())
    table_ratio_pk = table_unique_col_count / table_col_count if table_col_count > 0 else 0.0

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
    df_table["table_has_composite_pk"] = table_has_composite_pk
    df_table["table_avg_unique_ratio"] = round(table_avg_unique_ratio, 4)
    df_table["table_avg_null_ratio"] = round(table_avg_null_ratio, 4)
    df_table["table_ratio_of_pk_candidates"] = round(table_ratio_pk, 4)
    df_table["is_composite_key_part"] = (
        df_table["column_name"].str.lower().str.contains("id")
        & (df_table["table_has_no_single_pk_candidate"] == 1)
    ).astype(int)

    rank_idx = df_table.sort_values(
        ["unique_ratio", "ordinal_position"],
        ascending=[False, True],
    ).index
    df_table["unique_ratio_rank"] = pd.Series(range(1, len(rank_idx) + 1), index=rank_idx)

    rank_idx = df_table.sort_values(
        ["null_ratio", "ordinal_position"],
        ascending=[True, True],
    ).index
    df_table["null_ratio_rank"] = pd.Series(range(1, len(rank_idx) + 1), index=rank_idx)

    df_table["is_least_null_in_table"] = (df_table["null_ratio_rank"] == 1).astype(int)
    df_table["unique_ratio_relative_to_max"] = (
        (df_table["unique_ratio"] / table_max_unique_ratio) if table_max_unique_ratio > 0 else 0.0
    )

    # Partial-dependency heuristic (metadata-only approximation).
    cpk_unique_counts = set(
        df_table[df_table["is_composite_key_part"] == 1]["number_unique_values"].tolist(),
    )
    df_table["is_this_col_partial_dependency"] = (
        (df_table["is_composite_key_part"] == 0)
        & (df_table["is_unique"] == 0)
        & (table_has_composite_pk == 1)
        & (df_table["number_unique_values"].isin(cpk_unique_counts))
    ).astype(int)

    df_table["table_ratio_composite_key_cols"] = round(df_table["is_composite_key_part"].mean(), 4)
    df_table["table_ratio_1nf_violations"] = round(df_table["is_this_col_violating_1nf"].mean(), 4)
    std_val = df_table["unique_ratio"].std()
    df_table["table_std_unique_ratio"] = round(0.0 if pd.isna(std_val) else float(std_val), 4)
    df_table["table_has_partial_dependency"] = int(df_table["is_this_col_partial_dependency"].max())

    return df_table


def _normalize_type(raw: str) -> str:
    """Normalize a Trino/DuckDB data type to one of the model's column_type categories."""
    t = re.sub(r"\(.*\)", "", str(raw)).strip().lower()
    if t == "int":
        t = "integer"
    return t


def _encode_type_dummies(df_table: pd.DataFrame) -> pd.DataFrame:
    """Produce the 7 expected column_type_* dummy columns; unknown types -> all zero."""
    normalized = df_table["column_type"].map(_normalize_type)
    for col in EXPECTED_TYPE_COLS:
        type_name = col[len("column_type_") :]
        df_table[col] = (normalized == type_name).astype(int)
    return df_table


@task(name="extract-normalform-features", retries=2, retry_delay_seconds=30, cache_policy=NO_CACHE)
def extract_normalform_features(
    target_schemas: list[str],
    batch_size: int = 30,
) -> pd.DataFrame:
    """
    Profile all columns in the target schema(s) and produce the 50 model features
    (plus id columns database/schema/table_name/column_name for aggregation).

    Returns a DataFrame with one row per column.
    """
    engine = get_trino_engine()

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

    grouped: dict = {}
    for database, schema, table, column, data_type, ordinal_position in found_tables:
        grouped.setdefault((database, schema, table), []).append(
            (column, data_type, ordinal_position),
        )

    print(f"Processing {len(grouped)} tables")

    all_table_features = []
    for (database, schema, table), columns in grouped.items():
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
                        '{ordinal_position}' AS ordinal_position,
                        CASE
                            WHEN COUNT_IF(CAST("{column}" AS VARCHAR) LIKE '%,%') > 0 THEN 1
                            ELSE 0
                        END AS is_this_col_violating_1nf
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
        df_table = recompute_table_features(df_table)
        df_table = _encode_type_dummies(df_table)
        all_table_features.append(df_table)

    if not all_table_features:
        print("No data available - check tables and connection.")
        return pd.DataFrame()

    df_final = pd.concat(all_table_features, ignore_index=True)

    # Keep id columns (for aggregation) + exactly the 50 model features.
    id_cols = ["database", "schema", "table_name", "column_name"]
    missing = [f for f in MODEL_FEATURES if f not in df_final.columns]
    if missing:
        msg = f"Extractor is missing required model features: {missing}"
        raise RuntimeError(msg)

    df_final = df_final[id_cols + MODEL_FEATURES]
    n_tables = df_final["table_name"].nunique()
    print(f"Extracted features for {len(df_final)} columns across {n_tables} tables")
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
def predict_normalform(features: pd.DataFrame) -> pd.DataFrame:
    """
    Call POST /predict_normalform for each column row. Returns the input frame with
    per-column `prediction` (NF class) and `confidence` columns added.
    """
    url = f"{MODEL_API_URL}/predict_normalform"
    predictions = []
    confidences = []

    for _idx, row in features.iterrows():
        try:
            response = requests.post(url, json=_row_to_payload(row), timeout=30)
            response.raise_for_status()
            result = response.json()
            predictions.append(result.get("prediction"))
            confidences.append(result.get("probability", 0.0))
        except (requests.RequestException, requests.Timeout, requests.HTTPError) as e:
            loc = f"{row['schema']}.{row['table_name']}.{row['column_name']}"
            print(f"Error predicting normalform for {loc}: {e}")
            predictions.append(None)
            confidences.append(0.0)

    out = features.copy()
    out["prediction"] = predictions
    out["confidence"] = confidences
    return out


@task(name="aggregate-normalform", cache_policy=NO_CACHE)
def aggregate_to_table(predicted: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-column predictions to one row per table via majority vote.
    Confidence = mean probability among the columns that voted for the winning class.
    """
    predicted = predicted.dropna(subset=["prediction"]).copy()
    if predicted.empty:
        return pd.DataFrame()

    predicted["prediction"] = predicted["prediction"].astype(int)

    rows = []
    group_cols = ["database", "schema", "table_name"]
    for (database, schema, table), grp in predicted.groupby(group_cols):
        winning_class = int(grp["prediction"].value_counts().idxmax())
        winners = grp[grp["prediction"] == winning_class]
        rows.append(
            {
                "database": database,
                "schema": schema,
                "table_name": table,
                "predicted_normal_form": winning_class,
                "confidence": round(float(winners["confidence"].mean()), 4),
                "n_columns": len(grp),
            },
        )

    result = pd.DataFrame(rows)
    print(f"Aggregated predictions for {len(result)} tables")
    return result


@task(name="store-normalform-results", cache_policy=NO_CACHE)
def store_results(results: pd.DataFrame) -> int:
    """Append one row per table to duckdb.normalform_predictions.results."""
    if results.empty:
        print("No normalform results to store.")
        return 0

    results = results.copy()
    results["predicted_at"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    engine = get_trino_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS duckdb.normalform_predictions"))
        results.to_sql(
            "results",
            conn,
            schema="normalform_predictions",
            if_exists="append",
            index=False,
        )
    print(f"Stored {len(results)} table-level normalform predictions")
    return len(results)


@flow(name="normalform-prediction-pipeline")
def normalform_prediction_pipeline(
    target_schemas: list[str] | None = None,
    batch_size: int = 30,
) -> dict:
    """
    Complete normalform track: extract 50 features -> predict per column ->
    majority-vote aggregate to one NF class per table -> store.
    """
    if target_schemas is None:
        target_schemas = ["new_predict_data"]

    print("\n📊 Step 1: Extracting normalform features...")
    features = extract_normalform_features(target_schemas=target_schemas, batch_size=batch_size)
    if features.empty:
        print("⚠️  No features extracted - stopping.")
        return {"tables": 0}

    print("\n🤖 Step 2: Predicting normal form per column...")
    predicted = predict_normalform(features)

    print("\n📊 Step 3: Aggregating to one row per table...")
    table_results = aggregate_to_table(predicted)

    print("\n📥 Step 4: Storing table-level results...")
    stored = store_results(table_results)

    print("\n✅ Normalform pipeline complete!")
    return {"tables": stored}


if __name__ == "__main__":
    normalform_prediction_pipeline(target_schemas=["new_predict_data"], batch_size=30)
