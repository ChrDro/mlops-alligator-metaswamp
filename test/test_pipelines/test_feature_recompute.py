"""
Tests for the batch-safe feature recomputation in prefect/pk_fk_pipeline.py.

`recompute_table_stats` exists because features are extracted in column batches
(30 columns per query) while several features are *table-level*: unique column counts,
max unique ratio, ranks. Computed per batch they would be wrong for any table wider
than one batch, so the SQL values are recomputed in pandas once all batches are in.

The cross-column features are the subtle ones: `other_unique_columns_in_table` must
exclude the current column, or every column silently counts itself and the value
disagrees with the training data by exactly one.

Also covered: `_build_requeue_clause`, which is what makes a reloaded table get
re-predicted instead of being skipped forever by the queue's dedup.
"""

import pandas as pd
import pytest
from change_events import TableRef
from pk_fk_pipeline import _build_requeue_clause, recompute_table_stats


def _feature_frame(rows: list[dict]) -> pd.DataFrame:
    """
    Minimal per-column feature frame, as produced by the extraction query.

    Only the columns recompute_table_stats reads are included; the real frame has
    ~40 more that it passes through untouched.
    """
    defaults = {
        "null_count": 0,
        # Required, not optional: the extraction SQL always selects null_ratio
        # (pk_fk_pipeline.py:154) and recompute_table_stats ranks on it. Leaving it out
        # here made 11 tests fail the moment the null_ratio_rank fix landed.
        "null_ratio": 0.0,
        "name_ends_with_id": 0,
        "column_type": "varchar",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


# Three columns: two unique (ordinals 1 and 3), one not.
THREE_COLUMNS = [
    {"ordinal_position": 1, "is_unique": 1, "count": 100, "unique_ratio": 1.0},
    {"ordinal_position": 2, "is_unique": 0, "count": 100, "unique_ratio": 0.40},
    {"ordinal_position": 3, "is_unique": 1, "count": 100, "unique_ratio": 0.99},
]


# --- table-level aggregates ---------------------------------------------------


def test_table_level_counts_are_computed_across_all_batches():
    result = recompute_table_stats(_feature_frame(THREE_COLUMNS))

    assert result["table_column_count"].tolist() == [3, 3, 3]
    assert result["table_unique_column_count"].tolist() == [2, 2, 2]
    assert result["table_row_count"].tolist() == [100, 100, 100]
    assert result["table_has_unique_column"].tolist() == [1, 1, 1]
    assert result["table_has_no_single_pk_candidate"].tolist() == [0, 0, 0]


def test_a_table_without_a_unique_column_is_flagged():
    """`table_has_no_single_pk_candidate` is what points the model at composite keys."""
    rows = [
        {"ordinal_position": 1, "is_unique": 0, "count": 100, "unique_ratio": 0.5},
        {"ordinal_position": 2, "is_unique": 0, "count": 100, "unique_ratio": 0.2},
    ]

    result = recompute_table_stats(_feature_frame(rows))

    assert result["table_has_unique_column"].tolist() == [0, 0]
    assert result["table_has_no_single_pk_candidate"].tolist() == [1, 1]
    assert result["is_first_unique_column"].tolist() == [0, 0]


def test_near_unique_columns_use_a_strict_threshold():
    """> 0.95, so a column at exactly 0.95 does not count."""
    rows = [
        {"ordinal_position": 1, "is_unique": 0, "count": 100, "unique_ratio": 0.95},
        {"ordinal_position": 2, "is_unique": 0, "count": 100, "unique_ratio": 0.96},
    ]

    result = recompute_table_stats(_feature_frame(rows))

    assert result["table_near_unique_column_count"].tolist() == [1, 1]


# --- cross-column features ----------------------------------------------------


def test_other_unique_columns_excludes_the_column_itself():
    """
    Two unique columns in the table: each of them sees one *other* unique column,
    while the non-unique column sees two. Counting itself would shift the feature by
    one for exactly the rows the PK models care about most.
    """
    result = recompute_table_stats(_feature_frame(THREE_COLUMNS))

    assert result["other_unique_columns_in_table"].tolist() == [1, 2, 1]


def test_other_near_unique_columns_excludes_the_column_itself():
    result = recompute_table_stats(_feature_frame(THREE_COLUMNS))

    # unique_ratio > 0.95 holds for ordinals 1 and 3.
    assert result["other_near_unique_columns_in_table"].tolist() == [1, 2, 1]


def test_only_the_first_unique_column_is_flagged():
    """
    Composite-key detection leans on this: with several unique columns, the first one
    is the likely surrogate key and the rest are candidates for something else.
    """
    result = recompute_table_stats(_feature_frame(THREE_COLUMNS))

    assert result["is_first_unique_column"].tolist() == [1, 0, 0]


def test_relative_ordinal_position_is_normalised_by_column_count():
    result = recompute_table_stats(_feature_frame(THREE_COLUMNS))

    assert result["relative_ordinal_position"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_unique_ratio_rank_orders_descending_with_ordinal_as_tiebreak():
    rows = [
        {"ordinal_position": 1, "is_unique": 0, "count": 100, "unique_ratio": 0.50},
        {"ordinal_position": 2, "is_unique": 1, "count": 100, "unique_ratio": 1.00},
        {"ordinal_position": 3, "is_unique": 0, "count": 100, "unique_ratio": 0.50},
    ]

    result = recompute_table_stats(_feature_frame(rows))

    # Highest ratio ranks 1; the tie between ordinals 1 and 3 resolves by position.
    assert result["unique_ratio_rank"].tolist() == [2, 1, 3]


def test_unique_ratio_relative_to_max_scales_against_the_best_column():
    rows = [
        {"ordinal_position": 1, "is_unique": 0, "count": 100, "unique_ratio": 0.25},
        {"ordinal_position": 2, "is_unique": 0, "count": 100, "unique_ratio": 0.50},
    ]

    result = recompute_table_stats(_feature_frame(rows))

    assert result["unique_ratio_relative_to_max"].tolist() == pytest.approx([0.5, 1.0])


def test_a_table_of_all_empty_columns_does_not_divide_by_zero():
    """Guard for a table whose columns are all null: max unique ratio is 0."""
    rows = [
        {"ordinal_position": 1, "is_unique": 0, "count": 0, "unique_ratio": 0.0},
        {"ordinal_position": 2, "is_unique": 0, "count": 0, "unique_ratio": 0.0},
    ]

    result = recompute_table_stats(_feature_frame(rows))

    assert result["unique_ratio_relative_to_max"].tolist() == [0.0, 0.0]


def test_a_single_column_table_is_handled():
    rows = [{"ordinal_position": 1, "is_unique": 1, "count": 10, "unique_ratio": 1.0}]

    result = recompute_table_stats(_feature_frame(rows))

    assert result["table_column_count"].tolist() == [1]
    assert result["other_unique_columns_in_table"].tolist() == [0]
    assert result["relative_ordinal_position"].tolist() == [1.0]
    assert result["is_first_unique_column"].tolist() == [1]


# --- null-ratio ranking (regression, fixed 30.07.) ----------------------------


def test_null_ratio_rank_ranks_by_null_ratio():
    """
    Regression test for the 30.07. fix.

    The extraction SQL ranks by `null_ratio ASC, ordinal_position ASC`
    (pk_fk_pipeline.py:136-139), but this Python recompute - which *overwrites* that
    value once all batches are in - used to sort by `ordinal_position` alone. That
    silently degraded `null_ratio_rank` to "position in the table" and
    `is_least_null_in_table` to "is the first column", regardless of nulls; `null_ratio`
    was in the frame all along, the sort key had simply been dropped.

    Nothing fails when this regresses - the feature just quietly stops meaning what its
    name says - so the assertion below is the only guard. It is written so the old
    behaviour ([1, 0]) and the correct one ([0, 1]) are different values.
    """
    rows = [
        # Column 1 is the *most* null; column 2 has no nulls at all.
        {
            "ordinal_position": 1,
            "is_unique": 0,
            "count": 100,
            "unique_ratio": 0.5,
            "null_count": 90,
        },
        {"ordinal_position": 2, "is_unique": 0, "count": 100, "unique_ratio": 0.5, "null_count": 0},
    ]
    frame = _feature_frame(rows)
    frame["null_ratio"] = [0.9, 0.0]

    result = recompute_table_stats(frame)

    # The column with no nulls ranks first and is the one flagged.
    assert result["null_ratio_rank"].tolist() == [2, 1]
    assert result["is_least_null_in_table"].tolist() == [0, 1]


def test_null_ratio_rank_breaks_ties_by_ordinal_position():
    """Equal null ratios must fall back to position, as the SQL's second key does."""
    rows = [
        {"ordinal_position": 1, "is_unique": 0, "count": 100, "unique_ratio": 0.5},
        {"ordinal_position": 2, "is_unique": 0, "count": 100, "unique_ratio": 0.5},
        {"ordinal_position": 3, "is_unique": 0, "count": 100, "unique_ratio": 0.5},
    ]
    frame = _feature_frame(rows)
    frame["null_ratio"] = [0.0, 0.0, 0.0]

    result = recompute_table_stats(frame)

    assert result["null_ratio_rank"].tolist() == [1, 2, 3]
    assert result["is_least_null_in_table"].tolist() == [1, 0, 0]


# --- requeue clause -----------------------------------------------------------


def test_no_requeue_tables_produces_no_clause():
    assert _build_requeue_clause(None) == ("", {})
    assert _build_requeue_clause([]) == ("", {})


def test_requeue_clause_binds_one_predicate_per_table():
    """
    Without this fragment a column is predicted exactly once ever, because the queue
    skips anything already marked processed - so a reloaded table would keep its stale
    prediction.
    """
    tables = [
        TableRef("duckdb", "new_predict_data", "customer"),
        TableRef("duckdb", "new_predict_data", "orders"),
    ]

    clause, params = _build_requeue_clause(tables)

    assert clause.startswith("OR (")
    assert clause.count("f.table_name = :rq_tb_") == 2
    assert params == {
        "rq_db_0": "duckdb",
        "rq_sc_0": "new_predict_data",
        "rq_tb_0": "customer",
        "rq_db_1": "duckdb",
        "rq_sc_1": "new_predict_data",
        "rq_tb_1": "orders",
    }


def test_requeue_clause_does_not_interpolate_table_names():
    hostile = TableRef("duckdb", "new_predict_data", "orders'); DROP TABLE staging--")

    clause, params = _build_requeue_clause([hostile])

    assert hostile.table_name not in clause
    assert params["rq_tb_0"] == hostile.table_name
