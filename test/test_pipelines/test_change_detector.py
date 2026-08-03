"""
Tests for the watermark diff in prefect/change_detector.py.

`diff_snapshots` decides what the streaming pipeline reprocesses, so a wrong answer
here is either a missed reload (stale predictions kept forever) or a permanent
re-prediction loop. It is a pure function over a snapshot frame and the previous
watermarks, which makes it the highest-value thing in this module to test - no Trino
needed.

The detector's known blind spot is asserted too: an in-place UPDATE that leaves the
row and column count unchanged is invisible. That is a deliberate trade-off (iceberg
over Trino has no WAL to tail), and a test is the right place to record it - if
someone later adds a checksum column, this test is what tells them the contract
changed on purpose.
"""

import pandas as pd
import pytest
from change_detector import chunks, diff_snapshots
from change_events import TableRef


SNAPSHOT_COLUMNS = ["database", "schema", "table_name", "row_count", "column_count"]

CUSTOMER = TableRef("iceberg", "new_predict_data", "customer")
ORDERS = TableRef("iceberg", "new_predict_data", "orders")


def _snapshot(*rows) -> pd.DataFrame:
    """Build a snapshot frame from (ref, row_count, column_count) tuples."""
    records = [
        {
            "database": ref.database,
            "schema": ref.schema,
            "table_name": ref.table_name,
            "row_count": row_count,
            "column_count": column_count,
        }
        for ref, row_count, column_count in rows
    ]
    return pd.DataFrame(records, columns=SNAPSHOT_COLUMNS)


def _watermarks(*rows) -> dict:
    """Build a previous-watermark mapping from (ref, row_count, column_count) tuples."""
    return {tuple(ref): (row_count, column_count) for ref, row_count, column_count in rows}


# --- diff_snapshots -----------------------------------------------------------


def test_first_run_reports_every_table_as_new():
    """No previous watermarks means everything needs a first prediction."""
    current = _snapshot((CUSTOMER, 100, 8), (ORDERS, 50, 5))

    changes = diff_snapshots(current, {})

    assert changes == [(CUSTOMER, "new_table"), (ORDERS, "new_table")]


def test_unchanged_tables_are_not_reported():
    current = _snapshot((CUSTOMER, 100, 8))
    previous = _watermarks((CUSTOMER, 100, 8))

    assert diff_snapshots(current, previous) == []


def test_a_new_row_count_is_reported():
    current = _snapshot((CUSTOMER, 150, 8))
    previous = _watermarks((CUSTOMER, 100, 8))

    assert diff_snapshots(current, previous) == [(CUSTOMER, "row_count_changed")]


def test_deleted_rows_are_reported_too():
    """A shrinking table is as much a reload as a growing one."""
    current = _snapshot((CUSTOMER, 40, 8))
    previous = _watermarks((CUSTOMER, 100, 8))

    assert diff_snapshots(current, previous) == [(CUSTOMER, "row_count_changed")]


def test_a_column_change_outranks_a_row_change():
    """
    Both changed, but the schema change is the more informative label: it means the
    feature set itself may be different, not just the statistics.
    """
    current = _snapshot((CUSTOMER, 150, 9))
    previous = _watermarks((CUSTOMER, 100, 8))

    assert diff_snapshots(current, previous) == [(CUSTOMER, "schema_changed")]


def test_a_dropped_column_is_a_schema_change():
    current = _snapshot((CUSTOMER, 100, 7))
    previous = _watermarks((CUSTOMER, 100, 8))

    assert diff_snapshots(current, previous) == [(CUSTOMER, "schema_changed")]


def test_disappeared_tables_are_not_reported():
    """
    Nothing left to predict on, and the existing predictions stay as a historical
    record. Reporting them would queue work against a table that no longer exists.
    """
    current = _snapshot((CUSTOMER, 100, 8))
    previous = _watermarks((CUSTOMER, 100, 8), (ORDERS, 50, 5))

    assert diff_snapshots(current, previous) == []


def test_an_in_place_update_is_invisible():
    """
    Documented blind spot, not an oversight: iceberg through Trino offers no WAL, no
    triggers and no LISTEN/NOTIFY, so the watermark is row_count + column_count. An
    UPDATE that changes values but neither count cannot be detected without adding a
    checksum to the snapshot query.

    The real load pattern here is "reload the whole table", which moves the row count
    unless the new data has byte-identical cardinality.
    """
    current = _snapshot((CUSTOMER, 100, 8))
    previous = _watermarks((CUSTOMER, 100, 8))

    assert diff_snapshots(current, previous) == []


def test_tables_are_matched_on_the_full_three_part_name():
    """
    Same table name in two schemas must not be conflated - otherwise loading
    `other_schema.customer` would mark `new_predict_data.customer` as changed.
    """
    other_schema_customer = TableRef("iceberg", "other_schema", "customer")
    current = _snapshot((CUSTOMER, 100, 8), (other_schema_customer, 999, 3))
    previous = _watermarks((CUSTOMER, 100, 8))

    assert diff_snapshots(current, previous) == [(other_schema_customer, "new_table")]


def test_an_empty_snapshot_yields_no_changes():
    assert diff_snapshots(_snapshot(), _watermarks((CUSTOMER, 100, 8))) == []


# --- chunks -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (1, [[1], [2], [3], [4], [5]]),
        (2, [[1, 2], [3, 4], [5]]),
        (5, [[1, 2, 3, 4, 5]]),
        (10, [[1, 2, 3, 4, 5]]),
    ],
    ids=["one-each", "uneven-tail", "exact-fit", "larger-than-input"],
)
def test_chunks_splits_without_losing_items(size, expected):
    """
    Row counts are batched into UNION ALL queries; a lost chunk means a table is never
    counted and therefore never detected as changed.
    """
    assert chunks([1, 2, 3, 4, 5], size) == expected


def test_chunks_of_an_empty_list_is_empty():
    assert chunks([], 25) == []
