"""
Tests for the grouping the train/test split uses (Phase 1 of TASK_3_PLAN.md).

The split is the measurement. Every honest number in the plan depends on nothing closely
related to a holdout table sitting in training, and there are three ways to get that wrong -
each of which was in the code at some point.
"""

import pandas as pd
import pytest
from task_3_denormalization_train_and_register import split_groups


def _rows(*tables) -> pd.DataFrame:
    """One frame from (table_name, recipe_id, pair_id) triples, two rows per table."""
    records = []
    for table_name, recipe_id, pair_id in tables:
        for position in (1, 2):
            records.append(
                {
                    "table_name": table_name,
                    "meta_recipe_id": recipe_id,
                    "meta_pair_id": pair_id,
                    "ordinal_position": position,
                },
            )
    return pd.DataFrame(records)


def test_one_group_per_row_of_a_table():
    frame = _rows(("t1", "r1", None), ("t2", "r2", None))

    groups = split_groups(frame)

    assert groups.nunique() == 2
    assert groups.iloc[0] == groups.iloc[1], "a table's rows must never straddle the split"


def test_tables_from_one_recipe_share_a_group():
    """
    The same shape at another scale or under another naming is not an unseen schema.

    Grouping by table_name alone let those split, which is the 0.95-against-0.77 gap.
    """
    frame = _rows(("tiny_real", "r1", None), ("sf1_obf", "r1", None), ("other", "r2", None))

    groups = split_groups(frame)

    assert groups.nunique() == 2
    assert set(groups[frame["meta_recipe_id"] == "r1"]) == {groups.iloc[0]}


def test_a_matched_pair_stays_together_across_recipes():
    """
    The injection and its control differ in one column and carry opposite labels.

    They also carry different recipe ids - `x` against `x_control` - so grouping by recipe
    alone put all 24 pairs on both sides of the split. A near-identical table with the
    opposite label in training is the sharpest leak of the three.
    """
    frame = _rows(
        ("injection", "customer_order_columns", "pair_1"),
        ("control", "customer_order_columns_control", "pair_1"),
        ("unrelated", "something_else", None),
    )

    groups = split_groups(frame)

    assert groups.nunique() == 2
    injection = groups[frame["table_name"] == "injection"].iloc[0]
    control = groups[frame["table_name"] == "control"].iloc[0]
    assert injection == control


def test_linking_is_transitive():
    """
    A pair can tie two recipes into one component, and then that whole component has to
    stay whole - including the other tables of both recipes. Hence union-find rather than
    a composite key.
    """
    frame = _rows(
        ("a1", "ra", None),
        ("a2", "ra", "pair_1"),
        ("b1", "rb", "pair_1"),
        ("b2", "rb", None),
        ("c", "rc", None),
    )

    groups = split_groups(frame)

    assert groups.nunique() == 2
    linked = set(groups[frame["table_name"].isin(["a1", "a2", "b1", "b2"])])
    assert len(linked) == 1


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_an_unpaired_table_is_grouped_by_its_recipe_alone(missing):
    frame = _rows(("t1", "r1", missing), ("t2", "r1", missing))

    groups = split_groups(frame)

    assert set(groups) == {groups.iloc[0]}
