"""
Unit tests for the normalform pipeline's table-level aggregation.

`aggregate_to_table` used to collapse per-column predictions with a hard majority
vote (`value_counts().idxmax()`), weighing every column's vote equally regardless of
how confident it was, and breaking an exact split arbitrarily by row order. It now
averages each column's full probability distribution and takes the argmax of that -
these tests are built so the two strategies give *different* answers, and fail if
the hard vote is reintroduced.

`predict_normalform` and `aggregate_to_table` are Prefect tasks; `.fn` unwraps the
raw function so the test calls it directly, with no Prefect engine involved.
"""

import normalform_pipeline as nf
import pandas as pd
import pytest


def _row(table: str, prediction: int, probabilities: dict[str, float]) -> dict:
    return {
        "database": "iceberg",
        "schema": "new_predict_data",
        "table_name": table,
        "column_name": f"col_{table}_{prediction}",
        "prediction": prediction,
        "confidence": max(probabilities.values()),
        "probabilities": probabilities,
    }


def test_confidence_weighted_vote_overrides_a_weak_majority():
    """
    Two columns barely favour class 1 (51%), one column is 90% sure of class 0.
    A hard majority vote picks 1 (2 votes to 1); the confidence-weighted vote must
    pick 0, because the averaged distribution favours it.
    """
    predicted = pd.DataFrame(
        [
            _row("t1", 1, {"0": 0.49, "1": 0.51, "2": 0.0, "3": 0.0}),
            _row("t1", 1, {"0": 0.49, "1": 0.51, "2": 0.0, "3": 0.0}),
            _row("t1", 0, {"0": 0.90, "1": 0.10, "2": 0.0, "3": 0.0}),
        ],
    )

    result = nf.aggregate_to_table.fn(predicted)

    assert len(result) == 1
    assert result.iloc[0]["predicted_normal_form"] == 0
    assert result.iloc[0]["table_name"] == "t1"


def test_confidence_weighted_vote_resolves_an_exact_split_by_strength_not_order():
    """
    A 2-2 split in raw vote counts is not a tie once the columns' confidence is
    taken into account - the stronger pair should decide it, whichever order the
    rows arrive in.
    """
    predicted = pd.DataFrame(
        [
            _row("t2", 3, {"0": 0.05, "1": 0.05, "2": 0.05, "3": 0.85}),
            _row("t2", 2, {"0": 0.10, "1": 0.10, "2": 0.70, "3": 0.10}),
            _row("t2", 3, {"0": 0.05, "1": 0.05, "2": 0.05, "3": 0.85}),
            _row("t2", 2, {"0": 0.10, "1": 0.10, "2": 0.70, "3": 0.10}),
        ],
    )

    result = nf.aggregate_to_table.fn(predicted)

    # Confidence is the winning class's *averaged* probability across all four
    # columns (0.85, 0.10, 0.85, 0.10), not the mean among only the columns that
    # happened to vote for it - that would smuggle the old hard-vote grouping back in.
    assert result.iloc[0]["predicted_normal_form"] == 3
    assert result.iloc[0]["confidence"] == pytest.approx(0.475)


def test_aggregate_drops_columns_the_model_never_scored():
    """A failed API call leaves `prediction`/`probabilities` as None - must not crash."""
    predicted = pd.DataFrame(
        [
            _row("t3", 1, {"0": 0.1, "1": 0.8, "2": 0.05, "3": 0.05}),
            {
                "database": "iceberg",
                "schema": "new_predict_data",
                "table_name": "t3",
                "column_name": "col_failed",
                "prediction": None,
                "confidence": 0.0,
                "probabilities": None,
            },
        ],
    )

    result = nf.aggregate_to_table.fn(predicted)

    assert len(result) == 1
    assert result.iloc[0]["n_columns"] == 1
    assert result.iloc[0]["predicted_normal_form"] == 1


def test_aggregate_returns_empty_frame_when_nothing_scored():
    predicted = pd.DataFrame(
        [
            {
                "database": "iceberg",
                "schema": "new_predict_data",
                "table_name": "t4",
                "column_name": "col_failed",
                "prediction": None,
                "confidence": 0.0,
                "probabilities": None,
            },
        ],
    )

    result = nf.aggregate_to_table.fn(predicted)

    assert result.empty


def test_aggregate_separates_tables_with_the_same_column_name():
    predicted = pd.DataFrame(
        [
            _row("t5", 0, {"0": 0.9, "1": 0.1, "2": 0.0, "3": 0.0}),
            _row("t6", 3, {"0": 0.0, "1": 0.0, "2": 0.1, "3": 0.9}),
        ],
    )

    result = nf.aggregate_to_table.fn(predicted).set_index("table_name")

    assert result.loc["t5", "predicted_normal_form"] == 0
    assert result.loc["t6", "predicted_normal_form"] == 3
