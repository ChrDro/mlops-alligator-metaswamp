"""
Tests for the training manifest (Phase 2a of TASK_3_PLAN.md).

The manifest is the **y** side of the new training data. Two properties matter more
than the SQL: the declared FDs must round-trip exactly (2c re-derives the label from
them and compares against FDs discovered on the raw table), and the label must be
*computed* here rather than accepted from a recipe.
"""

import json

import pytest
from nf_labeling import normal_form
from nf_manifest import (
    MANIFEST_COLUMNS,
    MANIFEST_TABLE,
    build_manifest_row,
    decode_attribute_sets,
    decode_fds,
    encode_attribute_sets,
    encode_fds,
    ensure_manifest_table,
    insert_manifest_rows,
    load_manifest,
)

from .conftest import FakeConnection


ENROLMENT_ATTRIBUTES = ["student_id", "course_id", "grade", "student_name"]
ENROLMENT_FDS = [
    ({"student_id", "course_id"}, {"grade"}),
    ({"student_id"}, {"student_name"}),
]


# --------------------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------------------


def test_encoding_is_canonical():
    """Same dependencies declared in a different order must encode identically."""
    forwards = encode_fds(ENROLMENT_FDS)
    backwards = encode_fds(list(reversed(ENROLMENT_FDS)))

    assert forwards == backwards


def test_fds_round_trip_to_the_same_label():
    """
    The property 2c stands on.

    If encoding lost or reordered anything, the harness would re-derive a different
    label from the manifest than the generator wrote - and would blame the data.
    """
    decoded = decode_fds(encode_fds(ENROLMENT_FDS))

    assert normal_form(ENROLMENT_ATTRIBUTES, decoded, violates_1nf=False) == normal_form(
        ENROLMENT_ATTRIBUTES, ENROLMENT_FDS, violates_1nf=False
    )


def test_attribute_sets_round_trip():
    keys = [frozenset({"street", "zip"}), frozenset({"city", "street"})]

    assert sorted(decode_attribute_sets(encode_attribute_sets(keys)), key=sorted) == sorted(
        keys, key=sorted
    )


# --------------------------------------------------------------------------------------
# build_manifest_row
# --------------------------------------------------------------------------------------


def test_label_is_derived_not_accepted():
    """
    There is no ``target_normal_form`` parameter, on purpose.

    A recipe that believes it built a 2NF table but declared a partial dependency gets
    the label its dependencies support, not the one it intended.
    """
    row = build_manifest_row(
        table_name="tbl_enrolment",
        attributes=ENROLMENT_ATTRIBUTES,
        fds=ENROLMENT_FDS,
        violates_1nf=False,
        generation_params={"rows": 5000},
    )

    assert row.target_normal_form == 1


def test_row_carries_keys_and_the_reason_behind_the_label():
    row = build_manifest_row(
        table_name="tbl_enrolment",
        attributes=ENROLMENT_ATTRIBUTES,
        fds=ENROLMENT_FDS,
        violates_1nf=False,
        generation_params={"rows": 5000},
    )

    assert decode_attribute_sets(row.candidate_keys) == [frozenset({"course_id", "student_id"})]
    params = json.loads(row.generation_params)
    assert params["rows"] == 5000
    assert "partial dependency" in params["label_reason"]


def test_review_reason_defaults_to_none():
    """Empty review queue by default - **E1** fills it in when no UCC is found."""
    row = build_manifest_row(
        table_name="tbl_enrolment",
        attributes=ENROLMENT_ATTRIBUTES,
        fds=ENROLMENT_FDS,
        violates_1nf=False,
        generation_params={},
    )

    assert row.review_reason is None


def test_typo_in_a_declared_fd_is_rejected():
    """A non-firing FD makes a table look better normalized - it must not pass quietly."""
    with pytest.raises(ValueError, match="unknown attributes"):
        build_manifest_row(
            table_name="tbl_enrolment",
            attributes=ENROLMENT_ATTRIBUTES,
            fds=[({"stundent_id"}, {"student_name"})],
            violates_1nf=False,
            generation_params={},
        )


# --------------------------------------------------------------------------------------
# SQL
# --------------------------------------------------------------------------------------


def test_ddl_covers_every_manifest_column():
    conn = FakeConnection()

    ensure_manifest_table(conn)

    assert conn.statements("CREATE SCHEMA IF NOT EXISTS")
    ddl = conn.statements("CREATE TABLE IF NOT EXISTS")[0]
    for column in MANIFEST_COLUMNS:
        assert column in ddl


def test_rows_are_inserted_with_bound_parameters():
    conn = FakeConnection()
    row = build_manifest_row(
        table_name="tbl_enrolment",
        attributes=ENROLMENT_ATTRIBUTES,
        fds=ENROLMENT_FDS,
        violates_1nf=False,
        generation_params={},
    )

    written = insert_manifest_rows(conn, [row])

    assert written == 1
    insert_sql, params = conn.executed[-1]
    assert insert_sql.startswith(f"INSERT INTO {MANIFEST_TABLE}")
    # The FD JSON contains quotes and braces; it must never be interpolated.
    assert row.declared_fds not in insert_sql
    assert params["declared_fds_0"] == row.declared_fds
    assert params["table_name_0"] == "tbl_enrolment"


def test_insert_of_nothing_touches_no_table():
    """Not even the DDL - an empty generation run must be a complete no-op."""
    conn = FakeConnection()

    assert insert_manifest_rows(conn, []) == 0
    assert conn.executed == []


def test_review_queue_filters_on_review_reason():
    conn = FakeConnection(rows=[("iceberg", "nf_training", "t", 3, False, "[]", "[]", "{}", "x")])

    rows = load_manifest(conn, only_review=True)

    assert "review_reason IS NOT NULL" in conn.statements("SELECT")[0]
    assert rows[0].review_reason == "x"


def test_full_manifest_read_has_no_filter():
    conn = FakeConnection()

    load_manifest(conn)

    assert "WHERE" not in conn.statements("SELECT")[0]
