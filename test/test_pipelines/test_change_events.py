"""
Tests for the per-track claim protocol in prefect/change_events.py.

This is the concurrency contract of the streaming pipeline. Three consumers (`keys`,
`nf` and `subject_area`) work the same rows independently, and the trigger is
at-least-once, so the properties that matter are:

* a run only ever completes or releases **its own** claim - otherwise one pipeline
  closes work the other is still doing, and that change is never predicted again
* the track name is validated against a fixed allowlist before it reaches an f-string
  in a SQL statement
* table names and change types are **bound parameters**, never interpolated
* a work list created before a track existed gains that track's columns rather than
  being dropped - dropping it strands work no detector would ever re-flag

All of it is asserted against a fake connection, because the interesting failures are
in the SQL that gets built, not in what Trino does with it.
"""

import re
from datetime import UTC, datetime

import pytest
from change_events import (
    PENDING_TABLE,
    TRACK_KEYS,
    TRACK_NF,
    TRACKS,
    TableRef,
    claim_columns,
    claim_pending_changes,
    complete_pending_changes,
    count_unclaimed_changes,
    ensure_pending_table,
    record_pending_changes,
    release_pending_changes,
    release_stale_claims,
)

from .conftest import FakeConnection


RUN_ID = "run-abc-123"
OTHER_TABLE = TableRef("duckdb", "new_predict_data", "customer")

# The claim columns a work list had before the subject-area track was added. Written
# out rather than derived from TRACKS: the point is to pin the *old* shape, so adding
# a track must not silently update this fixture too.
LEGACY_CLAIM_COLUMNS = [
    "keys_claimed_by",
    "keys_claimed_at",
    "keys_completed_at",
    "nf_claimed_by",
    "nf_claimed_at",
    "nf_completed_at",
]

# A table name that would break out of a string literal if it were interpolated
# instead of bound. Not a realistic table name - that is the point.
HOSTILE_TABLE = TableRef("duckdb", "new_predict_data", "orders'); DROP TABLE staging--")


# --- track validation ---------------------------------------------------------


@pytest.mark.parametrize(
    "unknown_track",
    ["", "KEYS", "keys ", "keys; DROP TABLE x--", "pk"],
    ids=["empty", "wrong-case", "trailing-space", "injection", "plausible-but-wrong"],
)
@pytest.mark.parametrize(
    "function",
    [
        claim_pending_changes,
        complete_pending_changes,
        release_pending_changes,
        release_stale_claims,
    ],
    ids=["claim", "complete", "release", "release-stale"],
)
def test_unknown_track_is_rejected_before_any_sql_runs(function, unknown_track):
    """
    The track is interpolated into the statement (column names cannot be bound), so
    the allowlist is what makes that safe. It has to reject *before* executing.
    """
    connection = FakeConnection()

    with pytest.raises(ValueError, match="Unknown track"):
        function(connection, unknown_track, RUN_ID)

    assert connection.executed == []


@pytest.mark.parametrize("track", TRACKS)
def test_known_tracks_are_accepted(track):
    connection = FakeConnection(rows=[("duckdb", "new_predict_data", "customer")])

    assert claim_pending_changes(connection, track, RUN_ID) == [OTHER_TABLE]


# --- recording ----------------------------------------------------------------


def test_recording_nothing_touches_the_database():
    """The detector calls this on every pass, including the quiet ones."""
    connection = FakeConnection()

    assert record_pending_changes(connection, [], source="poller") == 0
    assert connection.executed == []


def test_recording_appends_one_insert_for_all_changes():
    """
    One multi-row INSERT rather than one per change: the round trip to Trino dominates
    the cost, and a partial write would leave the work list inconsistent.
    """
    connection = FakeConnection()
    changes = [
        (TableRef("duckdb", "new_predict_data", "customer"), "new_table"),
        (TableRef("duckdb", "new_predict_data", "orders"), "row_count_changed"),
    ]

    recorded = record_pending_changes(connection, changes, source="poller")

    assert recorded == 2
    inserts = connection.statements("INSERT INTO")
    assert len(inserts) == 1
    # Two value tuples, one per change.
    assert inserts[0].count("(:db_") == 2


def test_recording_creates_the_work_list_if_missing():
    connection = FakeConnection()

    record_pending_changes(connection, [(OTHER_TABLE, "new_table")], source="poller")

    assert connection.statements("CREATE SCHEMA IF NOT EXISTS duckdb.staging")
    assert connection.statements("CREATE TABLE IF NOT EXISTS")


# --- work-list schema ---------------------------------------------------------
#
# ensure_pending_table probes information_schema for the existing columns, so the
# canned rows below are (column_name,) tuples rather than table references.


def test_work_list_ddl_declares_a_claim_trio_for_every_track():
    """
    The DDL is generated from TRACKS. If the two ever drift, a claim UPDATE hits a
    column that does not exist and the whole track stops silently.
    """
    connection = FakeConnection(rows=[(column,) for column in claim_columns()])

    ensure_pending_table(connection)

    ddl = connection.statements("CREATE TABLE IF NOT EXISTS")[0]
    missing = [column for column in claim_columns() if f"{column} VARCHAR" not in ddl]
    assert missing == []
    assert len(claim_columns()) == 3 * len(TRACKS)


def test_a_current_work_list_is_left_alone():
    connection = FakeConnection(rows=[(column,) for column in claim_columns()])

    ensure_pending_table(connection)

    assert connection.statements("ALTER TABLE") == []


def test_a_work_list_predating_a_track_gains_its_columns():
    """
    CREATE TABLE IF NOT EXISTS does not alter an existing table, so the columns of a
    newly added track have to be added explicitly.
    """
    connection = FakeConnection(rows=[(column,) for column in LEGACY_CLAIM_COLUMNS])

    ensure_pending_table(connection)

    added = connection.statements("ALTER TABLE")
    expected = [column for column in claim_columns() if column not in LEGACY_CLAIM_COLUMNS]
    assert len(added) == len(expected)
    for column, statement in zip(expected, added, strict=True):
        assert statement == f"ALTER TABLE {PENDING_TABLE} ADD COLUMN {column} VARCHAR"


def test_a_work_list_predating_a_track_is_never_dropped():
    """
    pk_fk_pipeline drops and recreates its queue when the feature schema changes,
    because queue rows are rebuilt from staging. This table is different: the
    watermarks are already persisted, so a change dropped here is never re-detected.
    """
    connection = FakeConnection(rows=[(column,) for column in LEGACY_CLAIM_COLUMNS])

    ensure_pending_table(connection)

    assert connection.statements("DROP TABLE") == []


def test_a_claim_column_that_cannot_be_added_fails_loudly():
    """
    Better to stop with an actionable message than to let every later claim fail on a
    missing column, which reads like a broken pipeline rather than a stale schema.
    """

    class RefusesAlter(FakeConnection):
        def execute(self, statement, params=None):
            if "ALTER TABLE" in str(statement):
                message = "This connector does not support adding columns"
                raise RuntimeError(message)
            return super().execute(statement, params)

    connection = RefusesAlter(rows=[(column,) for column in LEGACY_CLAIM_COLUMNS])

    with pytest.raises(RuntimeError, match="DROP TABLE"):
        ensure_pending_table(connection)


def test_recording_binds_table_identifiers_instead_of_interpolating_them():
    """
    Identifiers come from catalog metadata, but the loader controls the names in that
    catalog - so a table name is untrusted input here. It must never reach the SQL
    text.
    """
    connection = FakeConnection()

    record_pending_changes(connection, [(HOSTILE_TABLE, "new_table")], source="poller")

    insert_sql = connection.statements("INSERT INTO")[0]
    assert HOSTILE_TABLE.table_name not in insert_sql
    assert "DROP TABLE" not in insert_sql
    assert connection.params_of("INSERT INTO")["tb_0"] == HOSTILE_TABLE.table_name


def test_recording_stores_the_source_and_a_detection_timestamp():
    connection = FakeConnection()

    record_pending_changes(connection, [(OTHER_TABLE, "new_table")], source="manual")

    params = connection.params_of("INSERT INTO")
    assert params["source"] == "manual"
    assert params["ct_0"] == "new_table"
    _assert_is_timestamp(params["detected_at"])


# --- claiming -----------------------------------------------------------------


def test_claiming_marks_open_rows_then_reads_back_its_own():
    connection = FakeConnection(rows=[("duckdb", "new_predict_data", "customer")])

    claimed = claim_pending_changes(connection, TRACK_KEYS, RUN_ID)

    update_sql = connection.statements("UPDATE")[0]
    assert f"{TRACK_KEYS}_claimed_by = :run_id" in update_sql
    # Only unclaimed, uncompleted rows may be taken.
    assert f"{TRACK_KEYS}_claimed_by IS NULL" in update_sql
    assert f"{TRACK_KEYS}_completed_at IS NULL" in update_sql
    assert connection.params_of("UPDATE")["run_id"] == RUN_ID

    select_sql = connection.statements("SELECT DISTINCT")[0]
    assert f"{TRACK_KEYS}_claimed_by = :run_id" in select_sql
    assert claimed == [OTHER_TABLE]


def test_claiming_deduplicates_tables_via_distinct():
    """
    Duplicate change rows for one table are deliberately not filtered on insert, so
    the reader is what collapses them - otherwise a table reloaded five times would be
    profiled five times in one run.
    """
    connection = FakeConnection(rows=[("duckdb", "new_predict_data", "customer")])

    claim_pending_changes(connection, TRACK_KEYS, RUN_ID)

    assert connection.statements("SELECT DISTINCT")


def test_claiming_uses_the_track_specific_columns():
    """
    One column pair per track, so `keys` finishing cannot hide the row from `nf`.

    Asserted on the claim statements only - the CREATE TABLE necessarily names both
    tracks' columns, since it defines them.
    """
    connection = FakeConnection()

    claim_pending_changes(connection, TRACK_NF, RUN_ID)

    claim_sql = "\n".join(
        connection.statements("UPDATE") + connection.statements("SELECT DISTINCT")
    )
    assert "nf_claimed_by" in claim_sql
    assert "keys_claimed_by" not in claim_sql


# --- completing and releasing -------------------------------------------------


def test_completing_only_closes_rows_this_run_claimed():
    """
    Changes detected *while* the run was in flight must stay open. Closing everything
    would silently drop them, because the watermark has already moved on.
    """
    connection = FakeConnection(rowcount=3)

    closed = complete_pending_changes(connection, TRACK_KEYS, RUN_ID)

    assert closed == 3
    update_sql = connection.statements("UPDATE")[0]
    assert f"{TRACK_KEYS}_completed_at = :completed_at" in update_sql
    assert f"{TRACK_KEYS}_claimed_by = :run_id" in update_sql
    _assert_is_timestamp(connection.params_of("UPDATE")["completed_at"])


def test_releasing_hands_the_work_back_without_completing_it():
    """Used when a run claims work and then fails - the next run must retry it."""
    connection = FakeConnection(rowcount=2)

    released = release_pending_changes(connection, TRACK_KEYS, RUN_ID)

    assert released == 2
    update_sql = connection.statements("UPDATE")[0]
    assert f"{TRACK_KEYS}_claimed_by = NULL" in update_sql
    assert f"{TRACK_KEYS}_claimed_at = NULL" in update_sql
    # Crucially not marked complete: the change still needs processing.
    assert "completed_at = :" not in update_sql


def test_stale_claim_recovery_uses_a_cutoff_in_the_past():
    """
    Recovers work claimed by a process that was killed outright, where the flow's
    `finally` block never ran. The cutoff must exceed the longest realistic pipeline
    runtime, or a slow but healthy run gets its work stolen.
    """
    connection = FakeConnection(rowcount=1)
    older_than_minutes = 120

    recovered = release_stale_claims(connection, TRACK_KEYS, older_than_minutes=older_than_minutes)

    assert recovered == 1
    cutoff = _parse_timestamp(connection.params_of("UPDATE")["cutoff"])
    age_minutes = (datetime.now(UTC) - cutoff).total_seconds() / 60
    assert older_than_minutes - 1 <= age_minutes <= older_than_minutes + 1


def test_stale_claim_recovery_ignores_completed_and_unclaimed_rows():
    connection = FakeConnection(rowcount=0)

    release_stale_claims(connection, TRACK_KEYS, older_than_minutes=120)

    update_sql = connection.statements("UPDATE")[0]
    assert f"{TRACK_KEYS}_claimed_by IS NOT NULL" in update_sql
    assert f"{TRACK_KEYS}_completed_at IS NULL" in update_sql


def test_stale_claim_cutoff_is_string_comparable():
    """
    Timestamps are stored as `YYYY-MM-DD HH:MM:SS` strings, whose lexicographic order
    matches chronological order - which is what makes the plain `<` comparison valid.
    """
    connection = FakeConnection()

    release_stale_claims(connection, TRACK_KEYS, older_than_minutes=1)

    cutoff = connection.params_of("UPDATE")["cutoff"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", cutoff)


# --- open-work counting -------------------------------------------------------


def test_counting_open_work_covers_every_track():
    """
    The detector emits its wake-up signal based on this count rather than on "did I
    just find something". Missing a track here would strand that track's work: the
    watermark is already persisted, so no later run would rediscover it.
    """
    connection = FakeConnection(scalar_value=7)

    assert count_unclaimed_changes(connection) == 7

    count_sql = connection.statements("SELECT COUNT(*)")[0]
    for track in TRACKS:
        assert f"{track}_claimed_by IS NULL AND {track}_completed_at IS NULL" in count_sql
    assert " OR " in count_sql


def test_counting_open_work_returns_zero_when_the_table_is_empty():
    """A scalar() of None must not propagate as None into arithmetic upstream."""
    connection = FakeConnection(scalar_value=None)

    assert count_unclaimed_changes(connection) == 0


def test_counting_open_work_creates_the_table_on_a_fresh_install():
    connection = FakeConnection(scalar_value=None)

    count_unclaimed_changes(connection)

    assert connection.statements("CREATE TABLE IF NOT EXISTS")


# --- schema -------------------------------------------------------------------


def test_work_list_carries_one_claim_column_pair_per_track():
    connection = FakeConnection()

    ensure_pending_table(connection)

    create_sql = connection.statements("CREATE TABLE IF NOT EXISTS")[0]
    assert PENDING_TABLE in create_sql
    for track in TRACKS:
        for column in ("claimed_by", "claimed_at", "completed_at"):
            assert f"{track}_{column}" in create_sql


# --- helpers ------------------------------------------------------------------

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def _parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, TIMESTAMP_FORMAT).replace(tzinfo=UTC)


def _assert_is_timestamp(value: str) -> None:
    parsed = _parse_timestamp(value)
    assert abs((datetime.now(UTC) - parsed).total_seconds()) < 60
