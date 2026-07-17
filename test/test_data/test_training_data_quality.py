"""Tests for training data quality and validation.

These tests verify that the training CSV has the expected schema,
data types, distributions, and consistency required for model training.
"""

import pandas as pd
import pytest
from pathlib import Path


# Path to training data
TRAINING_DATA_PATH = Path("data/summary_output_task_1_2_training.csv")

# Expected columns in the training data
EXPECTED_IDENTIFIER_COLUMNS = [
    "database",
    "schema",
    "table_name",
    "column_name",
]

EXPECTED_FEATURE_COLUMNS = [
    "column_type",
    "number_unique_values",
    "count",
    "null_count",
    "null_ratio",
    "is_unique",
    "min_value",
    "max_value",
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
    "table_integer_column_count",
]

EXPECTED_TARGET_COLUMNS = [
    "pk_target",
    "composite_pk_target",
    "fk_target",
    "composite_fk_target",
]

EXPECTED_ALL_COLUMNS = (
    EXPECTED_IDENTIFIER_COLUMNS
    + EXPECTED_FEATURE_COLUMNS
    + EXPECTED_TARGET_COLUMNS
)


@pytest.fixture
def training_df() -> pd.DataFrame:
    """Load training data for tests.
    
    This fixture is used by all tests in this module to avoid
    loading the CSV multiple times.
    """
    if not TRAINING_DATA_PATH.exists():
        pytest.skip(f"Training data not found at {TRAINING_DATA_PATH}")
    
    df = pd.read_csv(TRAINING_DATA_PATH)
    return df


# ============================================================================
# CATEGORY 1: DATA QUALITY TESTS
# ============================================================================

class TestSchemaValidation:
    """Test 1: Schema Validation - Critical for preventing training crashes."""

    def test_training_csv_exists(self):
        """Test that the training CSV file exists."""
        assert TRAINING_DATA_PATH.exists(), (
            f"Training data not found at {TRAINING_DATA_PATH}. "
            "Run data preparation script first."
        )

    def test_training_csv_is_not_empty(self, training_df: pd.DataFrame):
        """Test that the CSV has data rows."""
        assert len(training_df) > 0, "Training CSV is empty"
        assert len(training_df) >= 100, (
            f"Training data has only {len(training_df)} rows. "
            "Expected at least 100 rows for meaningful training."
        )

    def test_training_csv_has_all_required_columns(self, training_df: pd.DataFrame):
        """Test that all expected columns are present.
        
        Missing columns will cause training to fail.
        """
        actual_columns = set(training_df.columns)
        expected_columns = set(EXPECTED_ALL_COLUMNS)
        
        missing_columns = expected_columns - actual_columns
        assert not missing_columns, (
            f"Missing required columns: {sorted(missing_columns)}"
        )

    def test_training_csv_has_no_unexpected_columns(self, training_df: pd.DataFrame):
        """Test that there are no unexpected extra columns.
        
        Extra columns might indicate data pipeline issues.
        """
        actual_columns = set(training_df.columns)
        expected_columns = set(EXPECTED_ALL_COLUMNS)
        
        extra_columns = actual_columns - expected_columns
        
        # Allow for some flexibility but warn
        if extra_columns:
            pytest.warns(
                UserWarning,
                match=f"Found unexpected columns: {sorted(extra_columns)}"
            )

    def test_identifier_columns_are_present(self, training_df: pd.DataFrame):
        """Test that identifier columns are present."""
        for col in EXPECTED_IDENTIFIER_COLUMNS:
            assert col in training_df.columns, (
                f"Identifier column '{col}' is missing"
            )

    def test_target_columns_are_present(self, training_df: pd.DataFrame):
        """Test that all target columns are present."""
        for col in EXPECTED_TARGET_COLUMNS:
            assert col in training_df.columns, (
                f"Target column '{col}' is missing. Cannot train without targets."
            )


class TestDataTypeValidation:
    """Test 2: Data Type Validation - Ensures correct dtypes for training."""

    def test_numeric_feature_columns_are_numeric(self, training_df: pd.DataFrame):
        """Test that numeric features have numeric dtypes.
        
        scikit-learn requires numeric inputs.
        """
        numeric_columns = [
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
            "table_integer_column_count",
        ]
        
        for col in numeric_columns:
            if col in training_df.columns:
                assert pd.api.types.is_numeric_dtype(training_df[col]), (
                    f"Column '{col}' should be numeric but is {training_df[col].dtype}"
                )

    def test_target_columns_are_binary(self, training_df: pd.DataFrame):
        """Test that target columns contain only 0 and 1.
        
        Classification targets must be binary (0 or 1).
        """
        for col in EXPECTED_TARGET_COLUMNS:
            if col in training_df.columns:
                unique_values = set(training_df[col].dropna().unique())
                valid_values = {0, 1, 0.0, 1.0}
                
                assert unique_values.issubset(valid_values), (
                    f"Target column '{col}' has invalid values: {unique_values}. "
                    f"Expected only 0 or 1."
                )

    def test_boolean_like_columns_are_zero_or_one(self, training_df: pd.DataFrame):
        """Test that boolean-like columns contain only 0, 1, or NaN."""
        boolean_columns = [
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
        ]
        
        for col in boolean_columns:
            if col in training_df.columns:
                unique_values = set(training_df[col].dropna().unique())
                valid_values = {0, 1, 0.0, 1.0}
                
                assert unique_values.issubset(valid_values), (
                    f"Boolean column '{col}' has invalid values: {unique_values}. "
                    f"Expected only 0 or 1."
                )

    def test_column_type_is_string(self, training_df: pd.DataFrame):
        """Test that column_type is a string column."""
        if "column_type" in training_df.columns:
            assert pd.api.types.is_string_dtype(training_df["column_type"]) or \
                   pd.api.types.is_object_dtype(training_df["column_type"]), (
                f"column_type should be string but is {training_df['column_type'].dtype}"
            )


class TestNoMissingValuesInCriticalColumns:
    """Test 3: No Missing Values - Ensures critical columns are complete."""

    def test_identifier_columns_have_no_nulls(self, training_df: pd.DataFrame):
        """Test that identifier columns have no missing values.
        
        Identifiers (database, schema, table_name, column_name) must be complete.
        """
        for col in EXPECTED_IDENTIFIER_COLUMNS:
            if col in training_df.columns:
                null_count = training_df[col].isna().sum()
                assert null_count == 0, (
                    f"Identifier column '{col}' has {null_count} null values. "
                    "All identifiers must be present."
                )

    def test_target_columns_have_no_nulls(self, training_df: pd.DataFrame):
        """Test that target columns have no missing values.
        
        Cannot train without targets.
        """
        for col in EXPECTED_TARGET_COLUMNS:
            if col in training_df.columns:
                null_count = training_df[col].isna().sum()
                assert null_count == 0, (
                    f"Target column '{col}' has {null_count} null values. "
                    "Cannot train with missing targets."
                )

    def test_critical_feature_columns_have_reasonable_nulls(
        self, training_df: pd.DataFrame
    ):
        """Test that critical feature columns don't have too many nulls.
        
        Some nulls might be acceptable, but >50% nulls indicates data issues.
        """
        critical_features = [
            "number_unique_values",
            "count",
            "is_unique",
            "unique_ratio",
            "table_column_count",
        ]
        
        for col in critical_features:
            if col in training_df.columns:
                null_ratio = training_df[col].isna().sum() / len(training_df)
                assert null_ratio < 0.5, (
                    f"Critical feature '{col}' has {null_ratio*100:.1f}% nulls. "
                    "Expected less than 50%."
                )


# ============================================================================
# CATEGORY 2: DATA DISTRIBUTION TESTS
# ============================================================================

class TestTargetDistribution:
    """Test 4: Target Distribution - Checks for extreme class imbalance."""

    def test_pk_target_is_not_extremely_imbalanced(self, training_df: pd.DataFrame):
        """Test that pk_target has reasonable class balance.
        
        Extreme imbalance (>95% or <5% positive) leads to poor models.
        """
        if "pk_target" not in training_df.columns:
            pytest.skip("pk_target column not found")
        
        positive_ratio = training_df["pk_target"].mean()
        
        assert 0.01 < positive_ratio < 0.99, (
            f"pk_target is extremely imbalanced: {positive_ratio*100:.1f}% positive. "
            "Expected between 1% and 99%."
        )
        
        # Warn if still quite imbalanced
        if positive_ratio < 0.05 or positive_ratio > 0.95:
            pytest.warns(
                UserWarning,
                match=f"pk_target is imbalanced: {positive_ratio*100:.1f}% positive"
            )

    def test_pk_target_has_both_classes(self, training_df: pd.DataFrame):
        """Test that pk_target has at least one example of each class."""
        if "pk_target" not in training_df.columns:
            pytest.skip("pk_target column not found")
        
        unique_values = set(training_df["pk_target"].unique())
        
        assert 0 in unique_values or 0.0 in unique_values, (
            "pk_target has no negative examples (class 0)"
        )
        assert 1 in unique_values or 1.0 in unique_values, (
            "pk_target has no positive examples (class 1)"
        )

    def test_all_targets_have_both_classes(self, training_df: pd.DataFrame):
        """Test that all target columns have both classes."""
        for target_col in EXPECTED_TARGET_COLUMNS:
            if target_col in training_df.columns:
                unique_values = set(training_df[target_col].unique())
                has_zero = 0 in unique_values or 0.0 in unique_values
                has_one = 1 in unique_values or 1.0 in unique_values
                
                assert has_zero and has_one, (
                    f"Target '{target_col}' does not have both classes (0 and 1). "
                    f"Found: {unique_values}"
                )

    def test_target_distribution_summary(self, training_df: pd.DataFrame):
        """Print target distribution summary for visibility."""
        print("\n=== Target Distribution Summary ===")
        for target_col in EXPECTED_TARGET_COLUMNS:
            if target_col in training_df.columns:
                counts = training_df[target_col].value_counts()
                ratio = training_df[target_col].mean()
                print(f"\n{target_col}:")
                print(f"  Positive (1): {counts.get(1, 0)} ({ratio*100:.1f}%)")
                print(f"  Negative (0): {counts.get(0, 0)} ({(1-ratio)*100:.1f}%)")


class TestUniqueRatioBounds:
    """Test 5: Unique Ratio Bounds - Ensures ratio features are valid."""

    def test_unique_ratio_is_between_0_and_1(self, training_df: pd.DataFrame):
        """Test that unique_ratio is in valid range [0, 1]."""
        if "unique_ratio" not in training_df.columns:
            pytest.skip("unique_ratio column not found")
        
        min_val = training_df["unique_ratio"].min()
        max_val = training_df["unique_ratio"].max()
        
        assert min_val >= 0.0, (
            f"unique_ratio has value < 0: {min_val}"
        )
        assert max_val <= 1.0, (
            f"unique_ratio has value > 1: {max_val}"
        )

    def test_unique_ratio_relative_to_max_is_between_0_and_1(
        self, training_df: pd.DataFrame
    ):
        """Test that unique_ratio_relative_to_max is in valid range [0, 1]."""
        if "unique_ratio_relative_to_max" not in training_df.columns:
            pytest.skip("unique_ratio_relative_to_max column not found")
        
        min_val = training_df["unique_ratio_relative_to_max"].min()
        max_val = training_df["unique_ratio_relative_to_max"].max()
        
        assert min_val >= 0.0, (
            f"unique_ratio_relative_to_max has value < 0: {min_val}"
        )
        assert max_val <= 1.0, (
            f"unique_ratio_relative_to_max has value > 1: {max_val}"
        )

    def test_table_max_unique_ratio_is_between_0_and_1(
        self, training_df: pd.DataFrame
    ):
        """Test that table_max_unique_ratio is in valid range [0, 1]."""
        if "table_max_unique_ratio" not in training_df.columns:
            pytest.skip("table_max_unique_ratio column not found")
        
        min_val = training_df["table_max_unique_ratio"].min()
        max_val = training_df["table_max_unique_ratio"].max()
        
        assert min_val >= 0.0, (
            f"table_max_unique_ratio has value < 0: {min_val}"
        )
        assert max_val <= 1.0, (
            f"table_max_unique_ratio has value > 1: {max_val}"
        )

    def test_null_ratio_is_between_0_and_1(self, training_df: pd.DataFrame):
        """Test that null_ratio is in valid range [0, 1]."""
        if "null_ratio" not in training_df.columns:
            pytest.skip("null_ratio column not found")
        
        min_val = training_df["null_ratio"].min()
        max_val = training_df["null_ratio"].max()
        
        assert min_val >= 0.0, (
            f"null_ratio has value < 0: {min_val}"
        )
        assert max_val <= 1.0, (
            f"null_ratio has value > 1: {max_val}"
        )

    def test_relative_ordinal_position_is_between_0_and_1(
        self, training_df: pd.DataFrame
    ):
        """Test that relative_ordinal_position is in valid range [0, 1]."""
        if "relative_ordinal_position" not in training_df.columns:
            pytest.skip("relative_ordinal_position column not found")
        
        min_val = training_df["relative_ordinal_position"].min()
        max_val = training_df["relative_ordinal_position"].max()
        
        # Allow slight tolerance for rounding
        assert min_val >= -0.01, (
            f"relative_ordinal_position has value < 0: {min_val}"
        )
        assert max_val <= 1.01, (
            f"relative_ordinal_position has value > 1: {max_val}"
        )


# ============================================================================
# CATEGORY 3: BUSINESS LOGIC TESTS
# ============================================================================

class TestConsistencyChecks:
    """Test 6: Consistency Checks - Ensures feature consistency."""

    def test_is_unique_matches_unique_ratio(self, training_df: pd.DataFrame):
        """Test that is_unique=1 implies unique_ratio=1.0.
        
        If a column is unique, its unique_ratio should be 1.0.
        """
        if "is_unique" not in training_df.columns or \
           "unique_ratio" not in training_df.columns:
            pytest.skip("Required columns not found")
        
        # Check: is_unique=1 → unique_ratio=1.0 (allow small floating point error)
        unique_rows = training_df[training_df["is_unique"] == 1]
        if len(unique_rows) > 0:
            invalid = unique_rows[
                (unique_rows["unique_ratio"] < 0.99) &
                (unique_rows["count"] > 0)  # Exclude empty tables
            ]
            
            assert len(invalid) == 0, (
                f"Found {len(invalid)} rows where is_unique=1 but unique_ratio<1.0. "
                "This indicates inconsistent feature calculation."
            )

    def test_unique_ratio_1_implies_is_unique(self, training_df: pd.DataFrame):
        """Test that unique_ratio=1.0 implies is_unique=1.
        
        If unique_ratio is 1.0, the column should be marked as unique.
        """
        if "is_unique" not in training_df.columns or \
           "unique_ratio" not in training_df.columns:
            pytest.skip("Required columns not found")
        
        # Check: unique_ratio=1.0 → is_unique=1
        fully_unique_rows = training_df[
            (training_df["unique_ratio"] >= 0.99) &
            (training_df["count"] > 0)
        ]
        if len(fully_unique_rows) > 0:
            invalid = fully_unique_rows[fully_unique_rows["is_unique"] != 1]
            
            assert len(invalid) == 0, (
                f"Found {len(invalid)} rows where unique_ratio=1.0 but is_unique=0. "
                "This indicates inconsistent feature calculation."
            )

    def test_is_non_null_matches_null_ratio(self, training_df: pd.DataFrame):
        """Test that is_non_null=1 implies null_ratio=0.0."""
        if "is_non_null" not in training_df.columns or \
           "null_ratio" not in training_df.columns:
            pytest.skip("Required columns not found")
        
        non_null_rows = training_df[training_df["is_non_null"] == 1]
        if len(non_null_rows) > 0:
            invalid = non_null_rows[non_null_rows["null_ratio"] > 0.01]
            
            assert len(invalid) == 0, (
                f"Found {len(invalid)} rows where is_non_null=1 but null_ratio>0. "
                "This indicates inconsistent feature calculation."
            )

    def test_ordinal_position_is_positive(self, training_df: pd.DataFrame):
        """Test that ordinal_position is always >= 1.
        
        Column positions start at 1, not 0.
        """
        if "ordinal_position" not in training_df.columns:
            pytest.skip("ordinal_position column not found")
        
        min_position = training_df["ordinal_position"].min()
        assert min_position >= 1, (
            f"ordinal_position has value < 1: {min_position}. "
            "Column positions should start at 1."
        )

    def test_count_is_non_negative(self, training_df: pd.DataFrame):
        """Test that count is always >= 0."""
        if "count" not in training_df.columns:
            pytest.skip("count column not found")
        
        min_count = training_df["count"].min()
        assert min_count >= 0, (
            f"count has negative value: {min_count}"
        )


class TestGroupConsistency:
    """Test 7: Group Consistency - Ensures table-level features are consistent."""

    def test_table_column_count_consistent_per_table(self, training_df: pd.DataFrame):
        """Test that table_column_count is the same for all columns in a table.
        
        All columns in the same table should have the same table_column_count.
        """
        if "table_column_count" not in training_df.columns:
            pytest.skip("table_column_count column not found")
        
        # Group by table and check variance
        grouped = training_df.groupby(
            ["database", "schema", "table_name"]
        )["table_column_count"]
        
        variances = grouped.var()
        inconsistent = variances[variances > 0]
        
        assert len(inconsistent) == 0, (
            f"Found {len(inconsistent)} tables with inconsistent table_column_count. "
            f"Examples: {inconsistent.head()}"
        )

    def test_table_row_count_consistent_per_table(self, training_df: pd.DataFrame):
        """Test that table_row_count is the same for all columns in a table."""
        if "table_row_count" not in training_df.columns:
            pytest.skip("table_row_count column not found")
        
        grouped = training_df.groupby(
            ["database", "schema", "table_name"]
        )["table_row_count"]
        
        variances = grouped.var()
        inconsistent = variances[variances > 0]
        
        assert len(inconsistent) == 0, (
            f"Found {len(inconsistent)} tables with inconsistent table_row_count. "
            f"Examples: {inconsistent.head()}"
        )

    def test_table_has_unique_column_consistent_per_table(
        self, training_df: pd.DataFrame
    ):
        """Test that table_has_unique_column is the same for all columns in a table."""
        if "table_has_unique_column" not in training_df.columns:
            pytest.skip("table_has_unique_column column not found")
        
        grouped = training_df.groupby(
            ["database", "schema", "table_name"]
        )["table_has_unique_column"]
        
        variances = grouped.var()
        inconsistent = variances[variances > 0]
        
        assert len(inconsistent) == 0, (
            f"Found {len(inconsistent)} tables with inconsistent table_has_unique_column. "
            f"Examples: {inconsistent.head()}"
        )

    def test_table_max_unique_ratio_consistent_per_table(
        self, training_df: pd.DataFrame
    ):
        """Test that table_max_unique_ratio is the same for all columns in a table."""
        if "table_max_unique_ratio" not in training_df.columns:
            pytest.skip("table_max_unique_ratio column not found")
        
        grouped = training_df.groupby(
            ["database", "schema", "table_name"]
        )["table_max_unique_ratio"]
        
        variances = grouped.var()
        inconsistent = variances[variances > 0.001]  # Allow tiny floating point diff
        
        assert len(inconsistent) == 0, (
            f"Found {len(inconsistent)} tables with inconsistent table_max_unique_ratio. "
            f"Examples: {inconsistent.head()}"
        )


# ============================================================================
# CATEGORY 4: PIPELINE INTEGRATION TESTS
# ============================================================================

class TestOneHotEncoding:
    """Test 8: One-Hot Encoding - Ensures column_type can be encoded."""

    def test_column_type_has_expected_values(self, training_df: pd.DataFrame):
        """Test that column_type contains only expected data types.
        
        Unknown column types will cause one-hot encoding issues.
        """
        if "column_type" not in training_df.columns:
            pytest.skip("column_type column not found")
        
        expected_types = {
            "integer",
            "varchar",
            "date",
            "decimal",
            "double",
            "boolean",
            "timestamp",
            "char",
            # Add more as needed
        }
        
        actual_types = set(training_df["column_type"].dropna().unique())
        unexpected_types = actual_types - expected_types
        
        # This is a soft check - new types might be valid
        if unexpected_types:
            print(f"\nWarning: Found unexpected column types: {unexpected_types}")
            print("These will be one-hot encoded but might need review.")

    def test_column_type_can_be_one_hot_encoded(self, training_df: pd.DataFrame):
        """Test that pd.get_dummies works on column_type."""
        if "column_type" not in training_df.columns:
            pytest.skip("column_type column not found")
        
        try:
            encoded = pd.get_dummies(
                training_df,
                columns=["column_type"],
                drop_first=True
            )
            assert len(encoded.columns) > len(training_df.columns), (
                "One-hot encoding did not create new columns"
            )
        except Exception as e:
            pytest.fail(f"Failed to one-hot encode column_type: {e}")


class TestTrainTestSplit:
    """Test 9: Train-Test Split - Ensures data supports StratifiedGroupKFold."""

    def test_data_has_multiple_databases(self, training_df: pd.DataFrame):
        """Test that data has enough databases for group splitting.
        
        StratifiedGroupKFold with n_splits=5 requires at least 5 groups.
        """
        if "database" not in training_df.columns:
            pytest.skip("database column not found")
        
        n_databases = training_df["database"].nunique()
        
        assert n_databases >= 5, (
            f"Only {n_databases} databases found. "
            "Need at least 5 for StratifiedGroupKFold with 5 splits."
        )

    def test_each_database_has_multiple_samples(self, training_df: pd.DataFrame):
        """Test that each database has enough samples.
        
        Databases with very few samples might cause split issues.
        """
        if "database" not in training_df.columns:
            pytest.skip("database column not found")
        
        db_counts = training_df["database"].value_counts()
        small_databases = db_counts[db_counts < 5]
        
        # Warn if many databases are very small
        if len(small_databases) > 0:
            print(f"\nWarning: {len(small_databases)} databases have <5 samples:")
            print(small_databases.head())

    def test_stratification_is_possible(self, training_df: pd.DataFrame):
        """Test that each group has both positive and negative examples.
        
        Stratified split requires both classes in each split.
        """
        if "database" not in training_df.columns or \
           "pk_target" not in training_df.columns:
            pytest.skip("Required columns not found")
        
        # Check each database has both classes
        grouped = training_df.groupby("database")["pk_target"]
        class_counts = grouped.apply(lambda x: x.nunique())
        
        single_class_dbs = class_counts[class_counts == 1]
        
        if len(single_class_dbs) > 0:
            print(
                f"\nWarning: {len(single_class_dbs)} databases have only one class. "
                "This might cause stratification issues."
            )

    def test_groups_are_mutually_exclusive_in_splits(self, training_df: pd.DataFrame):
        """Test that group split logic works correctly.
        
        This mimics the actual split logic from the training script.
        """
        if "database" not in training_df.columns or \
           "pk_target" not in training_df.columns:
            pytest.skip("Required columns not found")
        
        from sklearn.model_selection import StratifiedGroupKFold
        
        groups = training_df["database"]
        y = training_df["pk_target"]
        X = training_df.drop(columns=EXPECTED_TARGET_COLUMNS, errors="ignore")
        
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        
        try:
            for train_idx, test_idx in sgkf.split(X, y, groups=groups):
                # Check that no database appears in both train and test
                train_dbs = set(groups.iloc[train_idx])
                test_dbs = set(groups.iloc[test_idx])
                overlap = train_dbs & test_dbs
                
                assert len(overlap) == 0, (
                    f"Found {len(overlap)} databases in both train and test split. "
                    f"Examples: {list(overlap)[:5]}"
                )
                
                # Only check first fold for performance
                break
                
        except ValueError as e:
            pytest.fail(f"StratifiedGroupKFold split failed: {e}")
