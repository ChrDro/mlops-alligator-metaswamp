"""
The quality gate for the normal-form model (Phase 4 of TASK_3_PLAN.md).

There was no gate before this file - CI ran Ruff and pytest and nothing checked model
quality at all. CI has neither MLflow nor Trino, so it cannot retrain to detect a
regression; what it can do is hold the *recorded* baseline to the floors the plan claims.

`data/nf_baseline.json` is written by every training run and committed. If a run comes back
worse, this test fails, and the only way past it is to lower a floor below - which is a
visible edit in a diff rather than a number quietly getting worse.

The floors are derived from the measured values, not from a round target. A gate at 0.99
would be the mistake the old plan warned about: unreachable, so permanently ignored. And
they are stated in the honest unit - see TABLE_ACCURACY_FLOOR.
"""

import json
from pathlib import Path

import pytest


BASELINE_PATH = Path(__file__).resolve().parents[2] / "data" / "nf_baseline.json"

# The headline. Table-level, because the pipeline stores one normal form per table, and a
# 40-column table would otherwise count eight times as much as a five-column one in the
# row-level metric.
#
# Both floors are the measured mean minus two standard deviations, rounded down. Two sigma
# rather than one because the mean is taken over five folds of ~85 tables each, and the
# fold-to-fold spread is real: this run's folds ranged from 0.9341 to 1.0000. A one-sigma
# floor would fail on an unlucky-but-normal fold draw and teach everyone to ignore it.
#
# Measured 2026-08-05: table accuracy 0.9724 +/- 0.0261, column F1 0.9823 +/- 0.0182, on
# 432 tables in 327 split groups. The previous floors (0.80 / 0.85) predate that measurement
# and were far enough below it to catch nothing.
TABLE_ACCURACY_FLOOR = 0.92

# The row-level number, kept as a secondary check. Higher than the table-level one by
# construction, so it is not a substitute for it.
COLUMN_F1_FLOOR = 0.94

# A baseline measured on far fewer tables or groups than the plan describes is not
# comparable to the one the floors were set against.
MIN_TABLES = 400
MIN_SPLIT_GROUPS = 300

# The split has to be the honest one. Recorded rather than assumed, because grouping by
# table_name alone would lift every number above by several points for free.
REQUIRED_GROUP_LINKS = {"meta_recipe_id", "meta_pair_id"}


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not BASELINE_PATH.exists():
        pytest.skip(
            f"{BASELINE_PATH.name} is missing - run "
            "task_3/task_3_denormalization_train_and_register.py to record a baseline",
        )
    return json.loads(BASELINE_PATH.read_text())


def test_table_level_accuracy_clears_the_floor(baseline):
    measured = baseline["cv_table_accuracy_mean"]
    assert measured >= TABLE_ACCURACY_FLOOR, (
        f"table-level accuracy fell to {measured:.4f}, floor is {TABLE_ACCURACY_FLOOR}"
    )


def test_column_level_f1_clears_the_floor(baseline):
    measured = baseline["cv_column_f1_mean"]
    assert measured >= COLUMN_F1_FLOOR, (
        f"column-level F1 fell to {measured:.4f}, floor is {COLUMN_F1_FLOOR}"
    )


def test_the_baseline_is_a_mean_over_folds_not_one_draw(baseline):
    """One split is one draw; with 65 groups in a holdout its spread is several points."""
    assert baseline["n_splits"] >= 5
    assert "cv_table_accuracy_std" in baseline


def test_the_spread_is_recorded_and_not_absurd(baseline):
    """
    A standard deviation this large would mean the mean says little about the next fold.

    Not a quality floor - a sanity check on the measurement itself.
    """
    assert baseline["cv_table_accuracy_std"] < 0.10


def test_the_baseline_was_measured_on_the_honest_split(baseline):
    assert set(baseline["split_group_links"]) == REQUIRED_GROUP_LINKS


def test_the_baseline_covers_enough_data_to_compare(baseline):
    assert baseline["n_tables"] >= MIN_TABLES
    assert baseline["n_split_groups"] >= MIN_SPLIT_GROUPS


def test_the_feature_count_matches_the_serving_contract(baseline):
    """
    The baseline is only meaningful for the feature set that produced it.

    Imported here rather than hard-coded: this is the same contract the request schema and
    the Prefect pipeline read, so a feature added in one place and not the others fails here.
    """
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "prefect"))
    from nf_features import COLUMN_TYPE_DUMMIES, FEATURE_COLUMNS

    assert baseline["n_features"] == len(FEATURE_COLUMNS) + len(COLUMN_TYPE_DUMMIES)
