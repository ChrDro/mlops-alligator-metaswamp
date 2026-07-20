# test_training_data_quality_subject_area.py
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def df(csv_path_subject: Path) -> pd.DataFrame:  # <--- Diese Fixture DARF NICHT gelöscht werden!
    """
    Loads the CSV file dynamically from the provided command-line path
    and performs basic data integrity checks.
    """
    assert csv_path_subject.exists(), f"File not found at path: {csv_path_subject}"
    data = pd.read_csv(csv_path_subject)
    assert len(data) >= 100, f"CSV is too small or empty ({len(data)} rows)"
    return data


# =====================================================================
# 1. SCHEMA VALIDATION (Ab hier folgen deine ganz normalen Tests...)
# =====================================================================

def test_required_columns_present(df: pd.DataFrame) -> None:
    """Verifies that all 41 columns required for subject area assignment exist."""
    expected_columns = {
        "database", "schema", "table_name", "column_name", "column_type",
        "number_unique_values", "count", "null_count", "null_ratio", "is_unique",
        "ordinal_position", "unique_ratio", "is_non_null", "is_first_column",
        "relative_ordinal_position", "is_first_unique_column", "table_column_count",
        "table_unique_column_count", "table_row_count", "other_unique_columns_in_table",
        "table_has_unique_column", "table_has_no_single_pk_candidate",
        "table_near_unique_column_count", "table_id_named_column_count",
        "table_non_null_column_count", "table_max_unique_ratio", "table_integer_column_count",
        "unique_ratio_rank", "null_ratio_rank", "is_least_null_in_table",
        "unique_ratio_relative_to_max", "other_near_unique_columns_in_table",
        "name_ends_with_id", "name_contains_key", "name_contains_table_name",
        "name_is_singular_table_id", "name_length", "pk_target", "composite_pk_target",
        "fk_target", "composite_fk_target",
    }
    missing = expected_columns - set(df.columns)
    assert not missing, f"Missing columns in the provided CSV file: {missing}"

# ... (restliche Tests: test_naming_and_metadata_integrity, etc. einfach so lassen wie vorhin)



# =====================================================================
# 2. STRING AND METADATA QUALITY
# =====================================================================


def test_naming_and_metadata_integrity(df: pd.DataFrame) -> None:
    """Validates that text fields are not corrupt, as names drive subject area logic."""
    text_fields = ["database", "schema", "table_name", "column_name", "column_type"]
    for field in text_fields:
        assert df[field].notna().all(), f"Null values found in critical naming column: {field}"

        # Uses is_string_dtype to support both legacy 'object' and modern Pandas 2.0+ string engines
        assert pd.api.types.is_string_dtype(df[field]), f"Column {field} must be of string type"

    # Verify name length calculation is correct
    assert (df["name_length"] == df["column_name"].str.len()).all(), \
        "Data anomaly: 'name_length' does not match the actual character count of 'column_name'"


# =====================================================================
# 3. LOGICAL BOUNDS & BINARY FLAGS
# =====================================================================


def test_binary_flags_and_ratios(df: pd.DataFrame) -> None:
    """Validates that indicator flags are strictly binary and ratios are bounded."""
    binary_flags = [
        "is_unique", "is_non_null", "is_first_column", "is_first_unique_column",
        "table_has_unique_column", "table_has_no_single_pk_candidate", "is_least_null_in_table",
        "name_ends_with_id", "name_contains_key", "name_contains_table_name",
        "name_is_singular_table_id",
    ]
    for flag in binary_flags:
        assert df[flag].isin([0, 1]).all(), f"Flag '{flag}' contains invalid values (must be 0 or 1)"

    ratios = ["null_ratio",
              "unique_ratio",
              "relative_ordinal_position",
              "table_max_unique_ratio",
              "unique_ratio_relative_to_max",
              ]
    for ratio in ratios:
        assert df[ratio].between(0.0, 1.0).all(), f"Ratio '{ratio}' is out of mathematical bounds (0.0 to 1.0)"


# =====================================================================
# 4. TABLE-LEVEL AGGREGATION CONSISTENCY
# =====================================================================


def test_table_level_consistency(df: pd.DataFrame) -> None:
    """Ensures aggregated table-level columns are uniform for all rows of the same table (prevents data leakage)."""
    table_aggregated_columns = [
        "table_column_count", "table_unique_column_count", "table_row_count",
        "table_has_unique_column", "table_has_no_single_pk_candidate",
        "table_near_unique_column_count", "table_id_named_column_count",
        "table_non_null_column_count", "table_max_unique_ratio", "table_integer_column_count",
    ]

    # Check that value variations equal exactly 1 unique value within the same table group
    for col in table_aggregated_columns:
        variations = df.groupby(["database", "schema", "table_name"])[col].nunique()
        assert (variations == 1).all(), f"Data Leakage Risk: '{col}' varies within the columns of the exact same table!"


# =====================================================================
# 5. KEY TARGETS LOGIC VALIDATION (For Multi-Task Learning)
# =====================================================================


def test_target_relationship_constraints(df: pd.DataFrame) -> None:
    """Enforces logical connections between the primary and foreign key targets."""
    target_flags = ["pk_target", "composite_pk_target", "fk_target", "composite_fk_target"]
    for target in target_flags:
        assert df[target].isin([0, 1]).all(), f"Target column '{target}' contains invalid non-binary values"

    # A single column cannot logically be a standalone PK and a composite PK part at the same time
    pk_overlap = df[(df["pk_target"] == 1) & (df["composite_pk_target"] == 1)]
    assert pk_overlap.empty, "Logical conflict: Column marked as both single PK and composite PK target simultaneously"

    # A single column cannot logically be a standalone FK and a composite FK part at the same time
    fk_overlap = df[(df["fk_target"] == 1) & (df["composite_fk_target"] == 1)]
    assert fk_overlap.empty, "Logical conflict: Column marked as both single FK and composite FK target simultaneously"
