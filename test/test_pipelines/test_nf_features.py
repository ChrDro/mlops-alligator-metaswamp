"""
Tests for the shared normal-form feature extractor (Phase 2b of TASK_3_PLAN.md).

``nf_features`` is the module that has to be right in two places at once - the training
set is built with it and the Prefect pipeline serves with it, so a change here moves
training and serving together instead of pulling them apart (finding 1.6).

Everything below is tested without a database. The measurements are handed in as a
``TableProfile``; what is under test is the reasoning on top of them.
"""

import pytest
from nf_features import (
    LIST_TOKEN_MAX_MEAN_LENGTH,
    LIST_VALUE_MIN_SHARE,
    ColumnInfo,
    TableProfile,
    classify_dependencies,
    find_dependencies,
    find_uccs,
    is_freetext_name,
    repeating_group_columns,
)

from .conftest import FakeConnection


def _profile(row_count, distinct, pair_distinct=None, **kwargs):  # noqa: ANN003
    """A TableProfile with only the fields a given test cares about filled in."""
    names = list(distinct)
    zeros = dict.fromkeys(names, 0.0)
    return TableProfile(
        row_count=row_count,
        columns=tuple(ColumnInfo(name=name, data_type="varchar") for name in names),
        distinct=distinct,
        non_null=kwargs.get("non_null", dict.fromkeys(names, row_count)),
        list_like_ratio=kwargs.get("list_like_ratio", zeros),
        mean_length=kwargs.get("mean_length", zeros),
        mean_separators=kwargs.get("mean_separators", zeros),
        pair_distinct=pair_distinct or {},
    )


# --------------------------------------------------------------------------------------
# Atomicity by name: repeating groups
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (["id", "phone1", "phone2", "phone3"], {"phone1", "phone2", "phone3"}),
        (["id", "order_1", "order_2"], {"order_1", "order_2"}),
        (["id", "phone1"], set()),  # a single numbered column is not a group
        (["id", "name", "city"], set()),
    ],
)
def test_repeating_groups_are_found_by_name(columns, expected):
    """The 1NF violation no value statistic can see - the values are perfectly atomic."""
    assert repeating_group_columns(columns) == expected


def test_a_table_wide_naming_scheme_is_not_a_repeating_group():
    """
    ``col_01`` .. ``col_12`` is obfuscated naming, not a flattened 1:n relationship.

    Without this guard, every table with positional column names would be labelled 0NF by
    a feature - a generator artifact of exactly the kind Phase 2 exists to remove.
    """
    assert repeating_group_columns([f"col_{i:02d}" for i in range(1, 13)]) == set()


def test_freetext_names_are_recognised():
    assert is_freetext_name("bemerkung")
    assert is_freetext_name("customer_comment")
    assert not is_freetext_name("customer_id")


# --------------------------------------------------------------------------------------
# Dependency discovery
# --------------------------------------------------------------------------------------


def test_dependency_holds_when_the_pair_does_not_split_the_determinant():
    """``a -> b`` holds exactly when adding ``b`` creates no new groups."""
    profile = _profile(
        row_count=100,
        distinct={"a": 10, "b": 5},
        pair_distinct={frozenset({"a", "b"}): 10},
    )

    assert find_dependencies(profile) == [("a", "b")]


def test_dependency_is_rejected_when_the_pair_splits_the_determinant():
    profile = _profile(
        row_count=100,
        distinct={"a": 10, "b": 5},
        pair_distinct={frozenset({"a", "b"}): 40},
    )

    assert find_dependencies(profile) == []


def test_a_unique_determinant_produces_no_findings():
    """
    A key determines everything, trivially.

    Counting those would swamp every dependency feature with noise that says nothing about
    the normal form.
    """
    profile = _profile(
        row_count=100,
        distinct={"a": 100, "b": 5},
        pair_distinct={frozenset({"a", "b"}): 100},
    )

    assert find_dependencies(profile) == []


# --------------------------------------------------------------------------------------
# Key search (decision E1)
# --------------------------------------------------------------------------------------


def test_key_search_does_not_stop_at_the_first_level():
    """
    The bug this test exists for, measured on the generated data.

    ``note_text`` - a free-text comment - is unique and therefore a perfectly valid
    level-1 UCC. Stopping there hides the real composite key ``{orderkey, linenumber}`` at
    level 2, and with it the partial dependency that defines 2NF: every 1NF table then
    looks like a 2NF one.
    """
    profile = _profile(
        row_count=100,
        distinct={"note_text": 100, "orderkey": 30, "linenumber": 7},
        pair_distinct={
            frozenset({"orderkey", "linenumber"}): 100,
            frozenset({"note_text", "orderkey"}): 100,
            frozenset({"note_text", "linenumber"}): 100,
        },
    )

    uccs, no_key = find_uccs(FakeConnection(), "t", profile, max_level=2)

    assert frozenset({"note_text"}) in uccs
    assert frozenset({"orderkey", "linenumber"}) in uccs
    assert not no_key


def test_supersets_of_a_key_are_not_reported():
    """Minimality: ``{note_text, orderkey}`` is unique but contains a smaller UCC."""
    profile = _profile(
        row_count=100,
        distinct={"note_text": 100, "orderkey": 30},
        pair_distinct={frozenset({"note_text", "orderkey"}): 100},
    )

    uccs, _ = find_uccs(FakeConnection(), "t", profile, max_level=2)

    assert uccs == [frozenset({"note_text"})]


def test_no_key_found_is_reported_rather_than_hidden():
    """
    The honest "unknown" state from **E1**.

    It has to reach the training data, or the model learns that a key is always available
    and has nothing to fall back on when one is not.
    """
    profile = _profile(
        row_count=100,
        distinct={"a": 5, "b": 4},
        pair_distinct={frozenset({"a", "b"}): 20},
    )

    uccs, no_key = find_uccs(FakeConnection(), "t", profile, max_level=2)

    assert uccs == []
    assert no_key


def test_level_three_is_pruned_by_cardinality_before_any_sql():
    """
    ``{a,b,c}`` can only be unique if the distinct counts multiply out to the row count.

    Pure arithmetic on numbers already measured - and with three low-cardinality columns
    there is nothing left to ask the database about.
    """
    profile = _profile(
        row_count=1000,
        distinct={"a": 2, "b": 3, "c": 4},
        pair_distinct={
            frozenset({"a", "b"}): 6,
            frozenset({"a", "c"}): 8,
            frozenset({"b", "c"}): 12,
        },
    )
    conn = FakeConnection()

    uccs, no_key = find_uccs(conn, "t", profile, max_level=3)

    assert uccs == []
    assert no_key
    assert conn.executed == []


# --------------------------------------------------------------------------------------
# 2NF versus 3NF
# --------------------------------------------------------------------------------------


def test_determinant_inside_a_composite_key_is_a_partial_dependency():
    """Half the key determining a non-key attribute - the definition of a 2NF violation."""
    uccs = [frozenset({"orderkey", "linenumber"})]

    classes = classify_dependencies([("orderkey", "orderdate")], uccs, prime=frozenset())

    assert classes.partial == [("orderkey", "orderdate")]
    assert classes.transitive == []


def test_determinant_outside_every_key_is_a_transitive_dependency():
    uccs = [frozenset({"orderkey"})]

    classes = classify_dependencies([("custkey", "mktsegment")], uccs, prime=frozenset())

    assert classes.transitive == [("custkey", "mktsegment")]
    assert classes.partial == []


def test_the_strict_variant_drops_a_dependency_whose_dependent_is_prime():
    """
    The textbook condition, and why it is reported separately.

    It is correct when the candidate keys are known and useless when they are discovered:
    a near-unique column forms a valid UCC with almost every other one, the union of all
    UCCs covers the table, and every attribute comes out prime. Measured on the generated
    set, that drove both strict counts to zero for every single table.
    """
    uccs = [frozenset({"orderkey", "linenumber"})]

    classes = classify_dependencies(
        [("orderkey", "orderdate")],
        uccs,
        prime=frozenset({"orderdate"}),
    )

    assert classes.partial == [("orderkey", "orderdate")]
    assert classes.strict_partial == []


def test_thresholds_are_stated_as_constants():
    """Guards the two numbers the 1NF detector turns on, so a silent edit shows up here."""
    assert LIST_VALUE_MIN_SHARE == 0.8
    # Per token, not per value: a list of 80 part ids is 355 characters long and every
    # piece of it is 4. The old per-value cap of 120 rejected it as prose.
    assert LIST_TOKEN_MAX_MEAN_LENGTH == 20
