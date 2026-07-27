"""
Shared contract for the change-driven ("streaming") prediction trigger.

Two detectors feed one mechanism
--------------------------------
1. **Push** - the process that loads data into ``new_predict_data`` calls
   ``POST /events/new-data`` on the model service, which emits
   :data:`NEW_DATA_EVENT`. Lowest latency, but only as reliable as the caller.
2. **Poll** - ``change_detector.py`` runs on a cron schedule and compares a cheap
   watermark per table against ``duckdb.staging.source_watermarks``. It catches
   everything the push path missed (loader crashed before the call, event lost
   during a server restart, someone loaded data by hand).

Both paths converge on the *same* code: the push event does not carry a payload of
changed tables, it merely wakes the poller up early. That keeps "what actually
changed" in exactly one place instead of duplicating the diff logic in the API.

Durable work list vs. transient signal
--------------------------------------
``duckdb.staging.pending_changes`` is the real work list; a Prefect event is only a
wake-up signal. If an event is lost, the next scheduled run still finds the work in
the table. If a duplicate event arrives, the run claims an empty list and exits
cheaply. The trigger is therefore **at-least-once**, which is the property we can
actually guarantee - so every consumer must be idempotent.

Per-track claiming
------------------
Two independent consumers process each change:

* ``keys`` - the pk/fk/cpk/cfk pipeline (``pk_fk_pipeline.py``)
* ``nf``   - the normal-form pipeline (``normalform_pipeline.py``)

A row therefore carries one claim/completion column pair per track, so one track
finishing does not hide the change from the other.

Concurrency: claiming is a plain ``UPDATE`` followed by a ``SELECT``. Trino/DuckDB
offers no ``SELECT ... FOR UPDATE``, so two concurrent runs of the *same* track
could claim overlapping rows. The streaming deployments are therefore created with
a concurrency limit of 1 per track in ``streaming_setup.py``.
"""

from datetime import UTC, datetime, timedelta
from typing import NamedTuple

from prefect.events import emit_event
from sqlalchemy import Connection, text


# --- Event names -------------------------------------------------------------

# Emitted by the model service webhook (push path). Only a hint: "look now".
NEW_DATA_EVENT = "alligator.new-data.arrived"

# Emitted by the change detector once it has written rows to pending_changes.
# This is what actually triggers the two prediction pipelines.
CHANGES_RECORDED_EVENT = "alligator.changes.recorded"

# Resource id used for both events. Prefect requires every event to name a
# resource; we use one logical resource so a single automation can match all of it.
EVENT_RESOURCE_ID = "alligator.metaswamp/new-predict-data"


# --- Tracks ------------------------------------------------------------------

TRACK_KEYS = "keys"
TRACK_NF = "nf"
TRACKS = (TRACK_KEYS, TRACK_NF)

PENDING_TABLE = "duckdb.staging.pending_changes"


class TableRef(NamedTuple):
    """A fully qualified source table."""

    database: str
    schema: str
    table_name: str


def _now() -> str:
    """Timestamp as VARCHAR, matching how the existing pipelines store timestamps."""
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _check_track(track: str) -> None:
    if track not in TRACKS:
        msg = f"Unknown track {track!r}, expected one of {TRACKS}"
        raise ValueError(msg)


def ensure_pending_table(conn: Connection) -> None:
    """Create the pending-changes work list if it does not exist yet."""
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS duckdb.staging"))
    conn.execute(
        text(f"""
            CREATE TABLE IF NOT EXISTS {PENDING_TABLE} (
                database VARCHAR,
                schema VARCHAR,
                table_name VARCHAR,
                change_type VARCHAR,
                source VARCHAR,
                detected_at VARCHAR,
                keys_claimed_by VARCHAR,
                keys_claimed_at VARCHAR,
                keys_completed_at VARCHAR,
                nf_claimed_by VARCHAR,
                nf_claimed_at VARCHAR,
                nf_completed_at VARCHAR
            )
        """),
    )


def record_pending_changes(
    conn: Connection,
    changes: list[tuple[TableRef, str]],
    source: str,
) -> int:
    """
    Append detected changes to the durable work list.

    Duplicates are deliberately not filtered out: an extra row is a harmless audit
    entry, because consumers claim with ``SELECT DISTINCT``. Filtering would need a
    read-modify-write that is not atomic here anyway.

    Args:
        conn: Open SQLAlchemy connection to Trino.
        changes: ``(table, change_type)`` pairs, e.g. ``(ref, "new_table")``.
        source: Which detector produced this ("poller", "manual", ...).

    Returns:
        Number of rows appended.
    """
    if not changes:
        return 0

    ensure_pending_table(conn)
    detected_at = _now()

    # One multi-row INSERT keeps this to a single round trip to Trino.
    value_rows = []
    params: dict[str, str] = {"detected_at": detected_at, "source": source}
    for i, (ref, change_type) in enumerate(changes):
        params[f"db_{i}"] = ref.database
        params[f"sc_{i}"] = ref.schema
        params[f"tb_{i}"] = ref.table_name
        params[f"ct_{i}"] = change_type
        value_rows.append(
            f"(:db_{i}, :sc_{i}, :tb_{i}, :ct_{i}, :source, :detected_at, "
            f"NULL, NULL, NULL, NULL, NULL, NULL)",
        )

    conn.execute(
        text(f"INSERT INTO {PENDING_TABLE} VALUES {', '.join(value_rows)}"),  # noqa: S608 - values are bound parameters
        params,
    )
    return len(changes)


def claim_pending_changes(conn: Connection, track: str, run_id: str) -> list[TableRef]:
    """
    Claim every open change for ``track`` and return the distinct tables to process.

    Claiming marks the rows with ``run_id`` so a later ``complete_pending_changes``
    only closes what this run actually handled - changes detected *while* the run
    was in flight stay open and are picked up by the next run.

    Args:
        conn: Open SQLAlchemy connection to Trino.
        track: ``TRACK_KEYS`` or ``TRACK_NF``.
        run_id: Identifier of the claiming flow run.

    Returns:
        Distinct tables with open changes for this track.
    """
    _check_track(track)
    ensure_pending_table(conn)

    conn.execute(
        text(f"""
            UPDATE {PENDING_TABLE}
            SET {track}_claimed_by = :run_id,
                {track}_claimed_at = :claimed_at
            WHERE {track}_claimed_by IS NULL
              AND {track}_completed_at IS NULL
        """),  # noqa: S608 - track is validated against a fixed allowlist
        {"run_id": run_id, "claimed_at": _now()},
    )

    rows = conn.execute(
        text(f"""
            SELECT DISTINCT database, schema, table_name
            FROM {PENDING_TABLE}
            WHERE {track}_claimed_by = :run_id
              AND {track}_completed_at IS NULL
        """),  # noqa: S608 - track is validated against a fixed allowlist
        {"run_id": run_id},
    ).fetchall()

    return [TableRef(r[0], r[1], r[2]) for r in rows]


def complete_pending_changes(conn: Connection, track: str, run_id: str) -> int:
    """Close all rows this run claimed for ``track``. Returns the number closed."""
    _check_track(track)

    result = conn.execute(
        text(f"""
            UPDATE {PENDING_TABLE}
            SET {track}_completed_at = :completed_at
            WHERE {track}_claimed_by = :run_id
              AND {track}_completed_at IS NULL
        """),  # noqa: S608 - track is validated against a fixed allowlist
        {"run_id": run_id, "completed_at": _now()},
    )
    return result.rowcount if hasattr(result, "rowcount") else 0


def release_pending_changes(conn: Connection, track: str, run_id: str) -> int:
    """
    Undo a claim without completing it, so the next run retries the same tables.

    Used when a run claims work but then fails before storing predictions - without
    this the changes would stay claimed by a dead run and never be processed again.
    """
    _check_track(track)

    result = conn.execute(
        text(f"""
            UPDATE {PENDING_TABLE}
            SET {track}_claimed_by = NULL,
                {track}_claimed_at = NULL
            WHERE {track}_claimed_by = :run_id
              AND {track}_completed_at IS NULL
        """),  # noqa: S608 - track is validated against a fixed allowlist
        {"run_id": run_id},
    )
    return result.rowcount if hasattr(result, "rowcount") else 0


def release_stale_claims(conn: Connection, track: str, older_than_minutes: int = 120) -> int:
    """
    Free work claimed by a run that never finished.

    ``complete``/``release`` run in the flow's ``finally`` block, which does not
    execute if the process is killed outright (OOM, container restart, ``kill -9``).
    Those rows would stay claimed by a dead run and never be processed again. This
    sweep hands them back so a later run picks them up.

    ``older_than_minutes`` must exceed the longest realistic pipeline runtime -
    otherwise a slow but healthy run gets its work stolen and both runs process the
    same tables (wasteful, though not incorrect: predictions are appended).

    Timestamps are stored as ``YYYY-MM-DD HH:MM:SS`` strings, whose lexicographic
    order matches chronological order, so a plain string comparison is safe here.
    """
    _check_track(track)
    ensure_pending_table(conn)

    cutoff = datetime.now(UTC) - timedelta(minutes=older_than_minutes)
    result = conn.execute(
        text(f"""
            UPDATE {PENDING_TABLE}
            SET {track}_claimed_by = NULL,
                {track}_claimed_at = NULL
            WHERE {track}_claimed_by IS NOT NULL
              AND {track}_completed_at IS NULL
              AND {track}_claimed_at < :cutoff
        """),  # noqa: S608 - track is validated against a fixed allowlist
        {"cutoff": cutoff.strftime("%Y-%m-%d %H:%M:%S")},
    )
    return result.rowcount if hasattr(result, "rowcount") else 0


def count_unclaimed_changes(conn: Connection) -> int:
    """
    Count open work that no run currently holds, across all tracks.

    "Unclaimed and not completed" is exactly what a fresh pipeline run would pick up.
    Rows a live run is holding are excluded, so a healthy in-flight run does not keep
    retriggering itself.

    The detector emits its signal based on this rather than on "did I just find a
    change". Otherwise work could strand permanently: a failed pipeline releases its
    claim, but the watermarks are already persisted, so no later run would ever
    detect a change for those tables again and nothing would wake the pipeline up.
    """
    ensure_pending_table(conn)

    conditions = " OR ".join(
        f"({track}_claimed_by IS NULL AND {track}_completed_at IS NULL)" for track in TRACKS
    )
    return (
        conn.execute(
            text(f"SELECT COUNT(*) FROM {PENDING_TABLE} WHERE {conditions}"),  # noqa: S608 - tracks come from a fixed allowlist
        ).scalar()
        or 0
    )


def emit_changes_recorded(table_count: int, source: str) -> None:
    """Signal that new work is waiting in the pending-changes table."""
    emit_event(
        event=CHANGES_RECORDED_EVENT,
        resource={"prefect.resource.id": EVENT_RESOURCE_ID},
        payload={"table_count": table_count, "source": source},
    )
