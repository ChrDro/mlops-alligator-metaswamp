"""
Change detector for the streaming trigger: what is new in ``new_predict_data``?

This is the single source of truth for "what changed". Both trigger paths end up
here - the cron schedule (safety net) and the push webhook (which only wakes this
flow up early). See ``change_events.py`` for the full contract.

Why a watermark instead of real CDC
-----------------------------------
The source is DuckDB accessed through Trino. There is no write-ahead log to tail,
no triggers, and no ``LISTEN/NOTIFY``, so Debezium-style CDC is not available. The
cheapest reliable substitute is a per-table version token:

* ``row_count``    - catches appended/deleted rows and full reloads
* ``column_count`` - catches structural changes (added/dropped columns)

Both are far cheaper than the profiling the prediction pipelines do
(``COUNT(DISTINCT …)`` per column, which scans every value). Counting rows is a
scan, but a single aggregate per table rather than one per column - exactly the
cost split described in ``documentation/INCREMENTAL_FEATURE_PROCESSING.md``.

Known blind spot: an in-place ``UPDATE`` that changes values without changing the
row or column count is invisible to this detector. The real-world load pattern for
this project is "reload the whole table", which always moves the row count unless
the new data has byte-identical cardinality. If that ever matters, add a checksum
column to the snapshot query.
"""

import os

import pandas as pd
import urllib3
from change_events import (
    TRACKS,
    TableRef,
    count_unclaimed_changes,
    emit_changes_recorded,
    record_pending_changes,
    release_stale_claims,
)
from dotenv import load_dotenv
from prefect.cache_policies import NONE as NO_CACHE
from sqlalchemy import Engine, create_engine, text

from prefect import flow, task


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

TRINO_IP_ADDRESS = str(os.environ.get("TRINO_IP_ADDRESS"))
TRINO_USERNAME = str(os.environ.get("TRINO_USERNAME"))
TRINO_PASSWORD = str(os.environ.get("TRINO_PASSWORD"))

WATERMARK_TABLE = "duckdb.staging.source_watermarks"

# How many per-table COUNT(*) aggregates to pack into one UNION ALL query.
SNAPSHOT_BATCH_SIZE = 25


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


def chunks(lst: list, n: int) -> list[list]:
    """Split a list into chunks of size n."""
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def _row_count_select(database: str, schema: str, table: str) -> str:
    """One branch of the UNION ALL row-count query."""
    return f"""
        SELECT '{database}' AS database,
               '{schema}' AS schema,
               '{table}' AS table_name,
               COUNT(*) AS row_count
        FROM {database}.{schema}.{table}
    """  # noqa: S608 - identifiers come from catalog metadata, not user input


@task(name="snapshot-source-tables", retries=2, retry_delay_seconds=30, cache_policy=NO_CACHE)
def snapshot_source_tables(target_schemas: list[str]) -> pd.DataFrame:
    """
    Build the current watermark for every table in the target schemas.

    Returns:
        DataFrame with database, schema, table_name, row_count, column_count.
        Empty DataFrame (with the right columns) if the schemas hold no tables.
    """
    engine = get_trino_engine()
    columns = ["database", "schema", "table_name", "row_count", "column_count"]

    schema_filter = "', '".join(target_schemas)
    discovery_query = text(f"""
        SELECT t.table_catalog, t.table_schema, t.table_name, COUNT(c.column_name) AS column_count
        FROM duckdb.information_schema.tables AS t
        INNER JOIN duckdb.information_schema.columns AS c
            ON t.table_catalog = c.table_catalog
            AND t.table_schema = c.table_schema
            AND t.table_name = c.table_name
        WHERE t.table_schema IN ('{schema_filter}')
        GROUP BY t.table_catalog, t.table_schema, t.table_name
    """)  # noqa: S608 - schema names are operator-supplied config, not user input

    with engine.connect() as conn:
        discovered = conn.execute(discovery_query).fetchall()

    if not discovered:
        print(f"No tables found in {target_schemas}")
        return pd.DataFrame(columns=columns)

    print(f"Found {len(discovered)} tables, counting rows...")

    column_counts = {(d[0], d[1], d[2]): d[3] for d in discovered}
    row_counts: dict[tuple[str, str, str], int] = {}

    # One UNION ALL per batch instead of one query per table: the row count itself is
    # cheap, the per-query round trip to Trino is what would dominate otherwise.
    for batch in chunks(list(column_counts), SNAPSHOT_BATCH_SIZE):
        union_parts = [_row_count_select(*ref) for ref in batch]
        query = " UNION ALL ".join(union_parts)

        try:
            with engine.connect() as conn:
                for row in conn.execute(text(query)).fetchall():
                    row_counts[(row[0], row[1], row[2])] = int(row[3])
        except Exception as e:  # noqa: BLE001 - one unreadable table must not blind the whole detector
            orig = getattr(e, "orig", None)
            msg = str(orig) if orig else str(e).split("\n")[0]
            print(f"Row count batch failed, skipping {len(batch)} tables: {msg}")

    records = [
        {
            "database": database,
            "schema": schema,
            "table_name": table,
            "row_count": row_counts[(database, schema, table)],
            "column_count": int(column_count),
        }
        for (database, schema, table), column_count in column_counts.items()
        if (database, schema, table) in row_counts
    ]

    return pd.DataFrame(records, columns=columns)


@task(name="load-previous-watermarks", cache_policy=NO_CACHE)
def load_previous_watermarks() -> dict[tuple[str, str, str], tuple[int, int]]:
    """Read the previous snapshot. Returns an empty mapping on the very first run."""
    engine = get_trino_engine()

    with engine.connect() as conn:
        exists = conn.execute(
            text(
                "SELECT COUNT(*) FROM duckdb.information_schema.tables "
                "WHERE table_schema = 'staging' AND table_name = 'source_watermarks'",
            ),
        ).scalar()
        if not exists:
            print("No previous watermarks - treating every table as new.")
            return {}

        select_watermarks = (
            f"SELECT database, schema, table_name, row_count, column_count FROM {WATERMARK_TABLE}"  # noqa: S608 - constant table name
        )
        rows = conn.execute(text(select_watermarks)).fetchall()

    return {(r[0], r[1], r[2]): (int(r[3]), int(r[4])) for r in rows}


def diff_snapshots(
    current: pd.DataFrame,
    previous: dict[tuple[str, str, str], tuple[int, int]],
) -> list[tuple[TableRef, str]]:
    """
    Compare the current snapshot against the previous one.

    Tables that disappeared are not reported: there is nothing left to predict on,
    and their existing predictions stay as a historical record.

    Returns:
        ``(table, change_type)`` pairs for every table that needs reprocessing.
    """
    changes: list[tuple[TableRef, str]] = []

    for row in current.itertuples(index=False):
        key = (row.database, row.schema, row.table_name)
        ref = TableRef(*key)

        if key not in previous:
            changes.append((ref, "new_table"))
            continue

        previous_rows, previous_columns = previous[key]
        if previous_columns != row.column_count:
            changes.append((ref, "schema_changed"))
        elif previous_rows != row.row_count:
            changes.append((ref, "row_count_changed"))

    return changes


@task(name="persist-watermarks", cache_policy=NO_CACHE)
def persist_watermarks(current: pd.DataFrame) -> int:
    """
    Overwrite the watermark table with the current snapshot.

    Full replace is intentional: the snapshot *is* the state, and dropped tables
    should vanish from it so a later table of the same name counts as new again.
    """
    engine = get_trino_engine()

    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS duckdb.staging"))
        current.to_sql(
            "source_watermarks",
            conn,
            schema="staging",
            if_exists="replace",
            index=False,
        )

    return len(current)


@task(name="record-changes", cache_policy=NO_CACHE)
def record_changes(changes: list[tuple[TableRef, str]], source: str) -> int:
    """Append the detected changes to the durable pending-changes work list."""
    engine = get_trino_engine()

    with engine.begin() as conn:
        return record_pending_changes(conn, changes, source=source)


@task(name="count-open-work", cache_policy=NO_CACHE)
def count_open_work() -> int:
    """How many pending-change rows are waiting for a pipeline to pick them up."""
    with get_trino_engine().connect() as conn:
        return count_unclaimed_changes(conn)


@task(name="recover-stale-claims", cache_policy=NO_CACHE)
def recover_stale_claims(stale_claim_minutes: int) -> int:
    """
    Hand back work claimed by runs that died without releasing it.

    The detector is the natural place for this: it runs on a schedule anyway, and it
    runs before the pipelines, so recovered work is picked up in the same cycle.
    """
    engine = get_trino_engine()
    recovered = 0

    with engine.begin() as conn:
        for track in TRACKS:
            recovered += release_stale_claims(conn, track, older_than_minutes=stale_claim_minutes)

    if recovered:
        print(f"Recovered {recovered} stale claims from runs that never finished")
    return recovered


@flow(name="change-detection-poller")
def change_detection_poller(
    target_schemas: list[str] | None = None,
    source: str = "poller",
    emit: bool = True,
    stale_claim_minutes: int = 120,
) -> dict:
    """
    Detect new/changed tables, record them, and signal the prediction pipelines.

    Runs both on a cron schedule (safety net) and on demand when the push webhook
    fires - the behaviour is identical, only ``source`` differs for traceability.

    Args:
        target_schemas: Schemas to watch (default ``['new_predict_data']``).
        source: Who triggered this run; stored on each pending-change row.
        emit: Emit the follow-up event. Set False to inspect the diff without
            triggering any predictions.
        stale_claim_minutes: Reclaim work that has been claimed for longer than this
            without completing. Must exceed the longest realistic pipeline runtime.

    Returns:
        Counts for tables scanned, changes recorded and claims recovered.
    """
    if target_schemas is None:
        target_schemas = ["new_predict_data"]

    recovered = recover_stale_claims(stale_claim_minutes)

    print(f"Snapshotting {target_schemas} (source: {source})...")
    current = snapshot_source_tables(target_schemas)
    if current.empty:
        print("No readable tables - nothing to do.")
        return {"tables_scanned": 0, "changes_recorded": 0, "claims_recovered": recovered}

    previous = load_previous_watermarks()
    changes = diff_snapshots(current, previous)

    recorded = 0
    if changes:
        for ref, change_type in changes:
            print(f"  • {change_type}: {ref.database}.{ref.schema}.{ref.table_name}")
        recorded = record_changes(changes, source=source)
    else:
        print(f"{len(current)} tables scanned, nothing changed.")

    # Persist the new watermark only AFTER the work is durably recorded. If this flow
    # dies in between, the next run re-detects the same change and records it twice -
    # harmless, because consumers dedup - whereas the other order would lose it.
    persist_watermarks(current)

    # Signal on the state of the work list, not on "did I just find something". Work
    # can be waiting without this run having detected it: a failed pipeline released
    # its claim, an earlier run recorded changes but died before emitting, or a run
    # with emit=False recorded them deliberately. Keying off newly-found changes only
    # would strand all three cases forever, because the watermarks are already
    # persisted and no future run would rediscover those tables.
    open_work = count_open_work()

    if emit and open_work > 0:
        emit_changes_recorded(table_count=open_work, source=source)
        print(
            f"Emitted change signal for {open_work} waiting rows "
            f"({len(changes)} newly detected, {recovered} recovered claims).",
        )
    elif not emit:
        print(f"(emit disabled - {open_work} rows left waiting in pending_changes)")
    else:
        print("Nothing waiting - no signal sent.")

    return {
        "tables_scanned": len(current),
        "changes_recorded": recorded,
        "claims_recovered": recovered,
        "open_work": open_work,
    }


if __name__ == "__main__":
    change_detection_poller(target_schemas=["new_predict_data"], source="manual")
