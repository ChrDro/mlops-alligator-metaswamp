from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def df(csv_path_nf: Path) -> pd.DataFrame:
    """Loads the Normal Form CSV file dynamically."""
    assert csv_path_nf.exists(), f"File not found at path: {csv_path_nf}"
    data = pd.read_csv(csv_path_nf)
    assert len(data) >= 100, f"CSV is too small or empty ({len(data)} rows)"
    return data


# =====================================================================
# 1. SCHEMA VALIDATION
# =====================================================================


def test_required_columns_present(df: pd.DataFrame) -> None:
    """Verifies that all 50 features exist exactly as specified in the schema."""
    expected_columns = {
        "database",
        "schema",
        "table_name",
        "column_name",
        "column_type",
        "number_unique_values",
        "count",
        "null_count",
        "null_ratio",
        "is_unique",
        "ordinal_position",
        "unique_ratio",
        "is_non_null",
        "is_first_column",
        "relative_ordinal_position",
        "is_first_unique_column",
        "table_column_count",
        "table_unique_column_count",
        "table_row_count",
        "other_unique_columns_in_table",
        "table_has_unique_column",
        "table_has_no_single_pk_candidate",
        "table_near_unique_column_count",
        "table_id_named_column_count",
        "table_non_null_column_count",
        "table_max_unique_ratio",
        "table_integer_column_count",
        "unique_ratio_rank",
        "null_ratio_rank",
        "is_least_null_in_table",
        "unique_ratio_relative_to_max",
        "other_near_unique_columns_in_table",
        "name_ends_with_id",
        "name_contains_key",
        "name_contains_table_name",
        "name_is_singular_table_id",
        "name_length",
        "table_avg_unique_ratio",
        "table_avg_null_ratio",
        "table_ratio_of_pk_candidates",
        "is_this_col_violating_1nf",
        "is_composite_key_part",
        "table_contains_1nf_violation",
        "table_has_composite_pk",
        "target_normal_form",
        "is_this_col_partial_dependency",
        "table_ratio_composite_key_cols",
        "table_ratio_1nf_violations",
        "table_std_unique_ratio",
        "table_has_partial_dependency",
    }
    missing = expected_columns - set(df.columns)
    assert not missing, f"Missing columns in the CSV: {missing}"


# =====================================================================
# 2. DATA TYPES & BOUNDS
# =====================================================================


def test_binary_and_ratio_bounds(df: pd.DataFrame) -> None:
    """Validates that boolean flags are strictly binary (0/1) and ratios stay within [0, 1]."""
    binary_flags = [
        "is_unique",
        "is_non_null",
        "is_first_column",
        "is_first_unique_column",
        "table_has_unique_column",
        "table_has_no_single_pk_candidate",
        "is_least_null_in_table",
        "name_ends_with_id",
        "name_contains_key",
        "name_contains_table_name",
        "name_is_singular_table_id",
        "is_this_col_violating_1nf",
        "is_composite_key_part",
        "table_contains_1nf_violation",
        "table_has_composite_pk",
        "is_this_col_partial_dependency",
        "table_has_partial_dependency",
    ]
    for flag in binary_flags:
        assert df[flag].isin([0, 1]).all(), f"Flag '{flag}' contains invalid values (not 0 or 1)."

    ratios = [
        "null_ratio",
        "unique_ratio",
        "relative_ordinal_position",
        "table_max_unique_ratio",
        "unique_ratio_relative_to_max",
        "table_avg_unique_ratio",
        "table_avg_null_ratio",
        "table_ratio_of_pk_candidates",
        "table_ratio_composite_key_cols",
        "table_ratio_1nf_violations",
    ]
    for ratio in ratios:
        assert df[ratio].between(0.0, 1.0).all(), (
            f"Ratio '{ratio}' out of bounds (must be between 0 and 1)."
        )


# =====================================================================
# 3. CONSISTENZ-CHECKS (Business Logic)
# =====================================================================


def test_column_level_consistency(df: pd.DataFrame) -> None:
    """Checks logical mathematical dependencies on a column level."""
    # If is_unique=1, unique_ratio MUST be 1.0 (and vice versa)
    assert df[(df["is_unique"] == 1) & (df["unique_ratio"] < 1.0)].empty, (
        "Inconsistency: is_unique=1 but unique_ratio < 1.0"
    )
    assert df[(df["unique_ratio"] == 1.0) & (df["is_unique"] == 0)].empty, (
        "Inconsistency: unique_ratio=1.0 but is_unique=0"
    )

    # If is_non_null=1, null_count and null_ratio MUST be 0
    assert df[(df["is_non_null"] == 1) & (df["null_count"] > 0)].empty, (
        "Inconsistency: is_non_null=1 but null_count > 0"
    )

    # Mathematical validity of unique_values and total count
    # unique_ratio should match number_unique_values / count
    valid_counts = df[df["count"] > 0]
    calculated_ratio = (valid_counts["number_unique_values"] / valid_counts["count"]).round(4)
    assert (valid_counts["unique_ratio"].round(4) == calculated_ratio).all(), (
        "Mathematical mismatch: unique_ratio does not equal number_unique_values / count"
    )


def test_table_level_group_consistency(df: pd.DataFrame) -> None:
    """Ensures that table-level aggregated metrics are identical for all columns of the same table (prevents data leakage)."""
    table_group_cols = [
        "table_column_count",
        "table_unique_column_count",
        "table_row_count",
        "table_has_unique_column",
        "table_has_no_single_pk_candidate",
        "table_max_unique_ratio",
        "table_contains_1nf_violation",
        "table_has_composite_pk",
    ]
    for col in table_group_cols:
        inconsistent = df.groupby(["database", "schema", "table_name"])[col].nunique()
        assert (inconsistent == 1).all(), (
            f"Inconsistency found: '{col}' varies within the columns of the exact same table!"
        )


# =====================================================================
# 4. NORMAL FORM-SPECIFIC LOGIC
# =====================================================================


def test_normal_form_logic_constraints(df: pd.DataFrame) -> None:
    """Validates strict relational database theory constraints for Normal Forms."""
    # 1NF Violation: If a column violates 1NF, the entire table must be flagged as containing a 1NF violation
    nf1_violators = df[
        (df["is_this_col_violating_1nf"] == 1) & (df["table_contains_1nf_violation"] == 0)
    ]
    assert nf1_violators.empty, (
        "Logical error: Column violates 1NF, but table_contains_1nf_violation is 0"
    )

    # Partial Dependency (2NF Trigger): Can only exist if the table has a composite primary key
    partial_dep_errors = df[
        (df["is_this_col_partial_dependency"] == 1) & (df["table_has_composite_pk"] == 0)
    ]
    assert partial_dep_errors.empty, (
        "Theoretical mismatch: Partial dependency reported without a composite primary key structure"
    )

    # If table_has_partial_dependency=1, at least one column within that table must have is_this_col_partial_dependency=1
    tables_with_partial_dep = df[df["table_has_partial_dependency"] == 1]
    if not tables_with_partial_dep.empty:
        has_violating_col = tables_with_partial_dep.groupby("table_name")[
            "is_this_col_partial_dependency"
        ].max()
        assert (has_violating_col == 1).all(), (
            "Table reports a partial dependency, but no individual column is flagged as a partial dependency"
        )


def test_target_variable(df: pd.DataFrame) -> None:
    """Ensures that the target label (target_normal_form) is completely defined and valid."""
    assert df["target_normal_form"].notna().all(), (
        "target_normal_form contains illegal NULL values!"
    )

    # Adjust the allowed target classes below based on your dataset labels (e.g., [1, 2, 3] or ['1NF', '2NF', '3NF'])
    allowed_targets = [0, 1, 2, 3]
    assert df["target_normal_form"].isin(allowed_targets).all(), (
        f"Unexpected Normal Form class in target: {df['target_normal_form'].unique()}"
    )
