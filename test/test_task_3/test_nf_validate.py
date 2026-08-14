"""
Tests for the validation harness (Phase 2c of TASK_3_PLAN.md).

The harness is what stands between a generation run and a training run, so its checks have
to fail on the defects they were written for. Each test below reconstructs one of those
defects from the current dataset in miniature and asserts the check catches it - and, just
as important, that a clean dataset passes.

No database is touched: the checks that need one take a connection, the rest take frames.
"""

import pandas as pd
import pytest
from nf_manifest import ManifestRow, build_manifest_row, encode_attribute_sets, encode_fds
from nf_validate import (
    check_key_ranking,
    check_key_search,
    check_label_noise,
    collect_review_reasons,
    decoy_report,
    detect_signatures,
    verify_label_against_discovered,
    verify_labels,
)

from .conftest import FakeConnection


def _table_rows(**overrides):  # noqa: ANN003
    """One table's worth of feature rows, with only what the checks read filled in."""
    defaults = {
        "table_name": "nf_001",
        "target_normal_form": 3,
        "table_ratio_list_like_columns": 0.0,
        "table_repeating_group_ratio": 0.0,
        "table_has_no_ucc_le3": 0,
        "table_column_count": 5,
        "table_row_count": 2000,
        "meta_recipe_id": "orders_core",
        "meta_naming": "real",
    }
    return {**defaults, **overrides}


def _frame(*tables):
    return pd.DataFrame([_table_rows(**table) for table in tables])


# --------------------------------------------------------------------------------------
# Decoy test - the 1NF detector on its own
# --------------------------------------------------------------------------------------


def test_detector_false_positive_fails_the_decoy_test():
    """
    Precision has no tolerance, and the reason is asymmetry.

    A miss loses one table. A false positive flips a whole table to 0NF - the fragility
    simulation in finding 1.5 measured that at F1 0.9939 -> 0.3143.
    """
    frame = _frame(
        {"table_name": "a", "target_normal_form": 0, "table_ratio_list_like_columns": 0.2},
        {"table_name": "b", "target_normal_form": 3, "table_ratio_list_like_columns": 0.1},
    )

    result = decoy_report(frame)

    assert not result.passed
    assert "false positive" in result.offenders["b"]


def test_detector_passes_when_it_stays_quiet_on_the_decoys():
    frame = _frame(
        {"table_name": "a", "target_normal_form": 0, "table_ratio_list_like_columns": 0.2},
        {"table_name": "b", "target_normal_form": 3},
        {"table_name": "c", "target_normal_form": 2},
    )

    result = decoy_report(frame)

    assert result.passed
    assert "precision 1.0000" in result.detail


def test_a_repeating_group_counts_as_a_detection():
    """No separator anywhere - the only evidence is in the column names."""
    frame = _frame(
        {"table_name": "a", "target_normal_form": 0, "table_repeating_group_ratio": 0.4},
    )

    assert decoy_report(frame).passed


# --------------------------------------------------------------------------------------
# Label noise
# --------------------------------------------------------------------------------------


def test_zero_nf_without_a_visible_violation_is_label_noise():
    """
    The exact defect of the six ``tbl_dirty_1nf_*`` tables.

    0NF by label with nothing in the data a feature could see. A model cannot learn that,
    only memorise it.
    """
    frame = _frame({"table_name": "a", "target_normal_form": 0})

    result = check_label_noise(frame)

    assert not result.passed
    assert "no explanation" in result.offenders["a"]


def test_an_obfuscated_repeating_group_is_reported_but_not_a_failure():
    """
    A repeating group lives in the column names. Obfuscate them and the evidence is gone.

    That is a limit, not a bug - so it goes to the review queue instead of failing the run.
    """
    frame = _frame(
        {"table_name": "a", "target_normal_form": 0, "meta_naming": "obfuscated"},
    )

    result = check_label_noise(frame)

    assert result.passed
    assert "obfuscated" in result.offenders["a"]


# --------------------------------------------------------------------------------------
# Signature test
# --------------------------------------------------------------------------------------


def test_a_feature_that_gives_away_the_label_is_caught():
    """
    The check that would have caught finding 1.3.

    ``table_ratio_1nf_violations`` was copied from the label and determined class 0 with
    100% agreement; here the same shape is built deliberately.
    """
    rows = []
    for index in range(40):
        label = index % 4
        rows.append(
            _table_rows(
                table_name=f"t{index}",
                target_normal_form=label,
                # A perfect stand-in for the label.
                table_ratio_list_like_columns=float(label),
                meta_recipe_id=f"recipe_{index // 4}",
            ),
        )

    result = detect_signatures(pd.DataFrame(rows))

    assert not result.passed
    assert "table_ratio_list_like_columns" in result.offenders


def test_features_that_say_nothing_pass_the_signature_test():
    rows = [
        _table_rows(
            table_name=f"t{index}",
            target_normal_form=index % 4,
            table_ratio_list_like_columns=0.0,
            table_repeating_group_ratio=0.0,
            meta_recipe_id=f"recipe_{index // 4}",
        )
        for index in range(40)
    ]

    assert detect_signatures(pd.DataFrame(rows)).passed


# --------------------------------------------------------------------------------------
# Key search (decision E1)
# --------------------------------------------------------------------------------------


def test_tables_without_a_key_are_reported_but_do_not_fail():
    """**E1** settled this: report and carry on, then look at them by hand."""
    frame = _frame(
        {"table_name": "a", "table_has_no_ucc_le3": 1},
        {"table_name": "b", "table_has_no_ucc_le3": 0},
    )

    result = check_key_search(frame)

    assert result.passed
    assert "a" in result.offenders
    assert "b" not in result.offenders


# --------------------------------------------------------------------------------------
# Label verification - the two directions
# --------------------------------------------------------------------------------------


def _manifest_row(**overrides) -> ManifestRow:  # noqa: ANN003
    row = build_manifest_row(
        table_name="nf_001",
        attributes=["a", "b", "c"],
        fds=[({"a"}, {"b", "c"})],
        violates_1nf=False,
        generation_params={},
    )
    return ManifestRow(**{**row.__dict__, **overrides})


def test_a_declared_dependency_that_does_not_hold_is_a_hard_failure():
    """
    The label was computed from dependencies the data does not satisfy - it is wrong.

    ``count(DISTINCT a) != count(DISTINCT (a, b))`` means adding ``b`` split a group of
    ``a``, so ``a -> b`` does not hold.
    """
    conn = FakeConnection(rows=[(10, 40)])  # distinct(a) = 10, distinct(a, b) = 40

    result = verify_labels(conn, [_manifest_row()])

    assert not result.passed
    assert "does not hold" in result.offenders["nf_001"]


def test_a_declared_dependency_that_holds_passes():
    conn = FakeConnection(rows=[(10, 10)])

    assert verify_labels(conn, [_manifest_row()]).passed


def test_discovered_dependencies_that_change_the_label_are_reported():
    """
    Extra dependencies always exist; only the ones that move the label matter.

    Here the table is labelled 3NF, but the data also satisfies ``b -> c`` - a transitive
    dependency that drags it to 2NF.
    """
    row = _manifest_row()
    discovered = {
        "nf_001": {"columns": ["a", "b", "c"], "dependencies": [["b", "c"]]},
    }

    result = verify_label_against_discovered([row], discovered)

    assert not result.passed
    assert "2NF once the" in result.offenders["nf_001"]


def test_discovered_dependencies_that_leave_the_label_alone_pass():
    """
    The reason this check asks about the label instead of about the dependencies.

    ``a`` is the key, so ``a -> c`` is implied by what was already declared and changes
    nothing. Reporting every extra dependency would cry wolf on every run.
    """
    row = _manifest_row()
    discovered = {
        "nf_001": {"columns": ["a", "b", "c"], "dependencies": [["a", "c"]]},
    }

    assert verify_label_against_discovered([row], discovered).passed


# --------------------------------------------------------------------------------------
# The review queue
# --------------------------------------------------------------------------------------


def test_offenders_from_several_checks_land_in_one_note():
    """One row per table in the manifest, so the notes have to be folded together."""
    frame = _frame(
        {"table_name": "a", "target_normal_form": 0, "table_has_no_ucc_le3": 1},
    )

    reasons = collect_review_reasons([check_label_noise(frame), check_key_search(frame)])

    assert "label noise" in reasons["a"]
    assert "key search" in reasons["a"]


def test_a_clean_run_files_nobody():
    frame = _frame({"table_name": "a", "target_normal_form": 3})

    assert collect_review_reasons([check_label_noise(frame), check_key_search(frame)]) == {}


@pytest.mark.parametrize("encoder", [encode_fds, encode_attribute_sets])
def test_encoders_still_round_trip_for_the_harness(encoder):
    """The harness reads the manifest back; a change in encoding would break it silently."""
    assert encoder([]) == "[]"


# --------------------------------------------------------------------------------------
# Key ranking (decision E2)
# --------------------------------------------------------------------------------------


def _row_with_key(*keys) -> ManifestRow:
    """A manifest row whose declared candidate keys are the given ones."""
    row = _manifest_row()
    return ManifestRow(**{**row.__dict__, "candidate_keys": encode_attribute_sets(keys)})


def test_key_ranking_passes_when_the_real_key_is_on_top():
    row = _row_with_key(["orderkey", "linenumber"])
    discovered = {"nf_001": {"uccs": [["linenumber", "orderkey"], ["note"]]}}

    result = check_key_ranking([row], discovered)

    assert result.passed
    assert "ranked first in 1/1" in result.detail


def test_key_ranking_reports_what_outranked_the_real_key():
    """
    The measured failure: a free-text comment is unique, so it is a valid UCC and it
    outranks the real key in every table it appears in.
    """
    row = _row_with_key(["orderkey", "linenumber"])
    discovered = {"nf_001": {"uccs": [["note_text"], ["linenumber", "orderkey"]]}}

    result = check_key_ranking([row], discovered, min_first=0.9)

    assert not result.passed
    assert "ranked 2 of 2" in result.offenders["nf_001"]
    assert "note_text" in result.offenders["nf_001"]


def test_a_key_that_is_not_discovered_at_all_fails_hard():
    """
    Measured at 132/132, so the bar is 1.0.

    A drop here means either the search broke or one of the E2 filters threw away a column
    the real key needs - and a filter that removes the answer is worse than no filter.
    """
    row = _row_with_key(["partkey", "suppkey"])
    discovered = {"nf_001": {"uccs": [["supplycost", "nationkey"]]}}

    result = check_key_ranking([row], discovered)

    assert not result.passed
    assert "not among the 1 discovered UCCs" in result.offenders["nf_001"]


def test_ranking_below_the_bar_fails_while_near_ties_are_tolerated():
    """
    Four of 132 real misses are genuine near-ties, so the bar sits at 0.9, not at 1.0.

    Nine tables ranked first out of ten is inside tolerance; the tenth is still reported.
    """
    rows, discovered = [], {}
    for index in range(10):
        row = _row_with_key(["a", "b"])
        row = ManifestRow(**{**row.__dict__, "table_name": f"t{index}"})
        rows.append(row)
        uccs = [["a", "b"]] if index else [["x"], ["a", "b"]]
        discovered[f"t{index}"] = {"uccs": uccs}

    result = check_key_ranking(rows, discovered)

    assert result.passed
    assert "ranked first in 9/10" in result.detail
    assert "t0" in result.offenders


def test_tables_without_diagnostics_are_skipped():
    assert check_key_ranking([_row_with_key(["a"])], {}).passed
