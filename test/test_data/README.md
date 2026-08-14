# Training Data Quality Tests
## Overview

This test suite validates the quality of the training CSV before the model is trained. It prevents common errors caused by poor data quality, missing columns, or inconsistent features.

**File:** `data/summary_output_task_1_2_training.csv`
**Total Tests:** 37 Tests in 9 categories

---

## 🎯 Why These Tests?

Since your data originates from a database, the following issues can occur:
- ✅ **CSV-Import** can change data types.
- ✅ **Query logic** can produce faulty features.
- ✅ **New Data** can have different distributions.
- ✅ **Schema changes** can remove columns.

These tests catch such issues early before the training starts.

---

## 📊 Test Categories

### Category 1: Data Quality Tests (6 Tests) ⭐⭐⭐

#### `TestSchemaValidation`
- ✅ CSV exists.
- ✅ CSV is not empty (minimum 100 rows).
- ✅ All required columns are present.
- ✅ No unexpected extra columns.
- ✅ Identifier columns are present.
- ✅ Target columns are present.

**Prevents:** Training crashes due to missing columns.

---

#### `TestDataTypeValidation` (4 Tests)
- ✅ Numerical features are numerical.
- ✅ Target columns are binary (0 or 1).
- ✅ Boolean-like columns are 0 or 1.
- ✅ column_type is a string.

**Prevents:**  Type errors during training.

---

#### `TestNoMissingValuesInCriticalColumns` (3 Tests)
- ✅ Identifiers have no NULL values.
- ✅ Targets have no NULL values.
- ✅ Critical features have less than 50% NULL values.

**Prevents:** Training errors due to missing values.

---

### **Category 2: Data Distribution Tests** (9 Tests)  ⭐⭐

#### `TestTargetDistribution`
- ✅ `pk_target` is not extremely unbalanced (<1% or >99%).
- ✅ `pk_target` contains both classes (0 and 1).
- ✅ All targets contain both classes.
- ✅ Print target distribution summary.

**Prevents:** Poor models caused by extreme class imbalance.

---

#### `TestUniqueRatioBounds` (5 Tests)
- ✅ `unique_ratio` is between 0 and 1.
- ✅ `unique_ratio_relative_to_max` is between 0 and 1.
- ✅ `table_max_unique_ratio` is between 0 and 1.
- ✅ `null_ratio` is between 0 and 1.
- ✅ `relative_ordinal_position` is between 0 and 1.

**Prevents:** Faulty feature calculations.

---

### **Category 3: Business Logic Tests** (9 Tests) ⭐⭐

#### `TestConsistencyChecks` (5 Tests)
- ✅ `is_unique=1` → `unique_ratio=1.0`
- ✅ `unique_ratio=1.0` → `is_unique=1`
- ✅ `is_non_null=1` → `null_ratio=0.0`
- ✅ `ordinal_position` >= 1
- ✅ `count` >= 0

**Prevents:** Inconsistent features (Data Leakage)

---

#### `TestGroupConsistency` (4 Tests)
- ✅ `table_column_count` is identical for all columns of the same table.
- ✅ `table_row_count` is identical for all columns of the same table.
- ✅ `table_has_unique_column` is identical for all columns of the same table.
- ✅ `table_max_unique_ratio` is identical for all columns of the same table.

**Prevents:** Faulty aggregations (Data Leakage)

---

### **Category 4: Pipeline Integration Tests** (6 Tests) ⭐

#### `TestOneHotEncoding` (2 Tests)
- ✅ `column_type` contains expected values.
- ✅ `pd.get_dummies()` works correctly.

**Prevents:** Crashes during one-hot encoding.

---

#### `TestTrainTestSplit` (4 Tests)
- ✅ At least 5 databases available (for 5-fold split).
- ✅ Each database contains multiple samples.
- ✅ `Stratification` is possible (both classes present per DB).
- ✅ `StratifiedGroupKFold` works correctly (no DB overlap).

**Prevents:** Crashes during train-test split.

---

## 🚀 Tests execution

### All Data Quality Tests
```bash
pytest tests/test_data/test_training_data_quality.py -v
```

### Only Schema Tests (schnell)
```bash
pytest tests/test_data/test_training_data_quality.py::TestSchemaValidation -v
```

### Only Distribution Tests
```bash
pytest tests/test_data/test_training_data_quality.py::TestTargetDistribution -v
```

### With detailed Output
```bash
pytest tests/test_data/test_training_data_quality.py -v -s
```
(The `-s` flag shows Print-Statements like Target Distribution Summary)

---

## 📈 Example Output

### Successful Test Run
```
tests/test_data/test_training_data_quality.py::TestSchemaValidation::test_training_csv_exists PASSED
tests/test_data/test_training_data_quality.py::TestSchemaValidation::test_training_csv_is_not_empty PASSED
...
tests/test_data/test_training_data_quality.py::TestTargetDistribution::test_target_distribution_summary PASSED

=== Target Distribution Summary ===

pk_target:
  Positiv (1): 1234 (11.9%)
  Negativ (0): 9136 (88.1%)

composite_pk_target:
  Positiv (1): 456 (4.4%)
  Negativ (0): 9914 (95.6%)

...

============================== 37 passed in 2.45s ===============================
```

### Aborted Test (Example)
```
FAILED tests/test_data/test_training_data_quality.py::TestSchemaValidation::test_training_csv_has_all_required_columns

AssertionError: Missing required columns: ['ordinal_position', 'pk_target']
```

---

## 🐛 Common Errors and Solutions

### "Training data not found"
**Problem:** CSV file does not exist
**Solution:** Run the data pipeline
```bash
python src/get_trino_summaries_task_1_2_training.py
```

### "pk_target is extremely imbalanced: 0.2% positive"
**Problem:** Too few positive examples
**Solution:** Collect more data or adjust the sampling strategy

### "Missing required columns: ['unique_ratio']"
**Problem:** Column is missing from the CSV
**Solution:** Check the query in the data pipeline and rerun it

### "is_unique=1 but unique_ratio<1.0"
**Problem:** Inconsistent feature calculation
**Solution:** Verify the logic in `get_trino_summaries`

### "StratifiedGroupKFold split failed"
**Problem:** Not enough databases or data is too unbalanced
**Solution:** Not enough databases or data is too unbalanced

---

## 🔄 Integration in CI/CD

These tests should run in your CI/CD pipeline **before** the model training starts:

```yaml
# .github/workflows/train-model.yml
jobs:
  validate-data:
    runs-on: ubuntu-latest
    steps:
      - name: Validate Training Data
        run: pytest tests/test_data/test_training_data_quality.py -v

  train-model:
    needs: validate-data  # Nur wenn Data Quality OK
    runs-on: ubuntu-latest
    steps:
      - name: Train Model
        run: python task_1/task_1_pk_train_and_register.py
```

---

## 🔄 Integration in Prefect Pipeline

Add a data quality check as the first task:

```python
from prefect import flow, task
import pytest


@task
def validate_training_data():
    """Run data quality tests before training."""
    exit_code = pytest.main(["tests/test_data/test_training_data_quality.py", "-v"])
    if exit_code != 0:
        raise ValueError("Data quality tests failed!")


@task
def train_model():
    # ... existing training code
    pass


@flow
def training_pipeline():
    validate_training_data()  # Runs first
    train_model()  # Only runs if validation passes
```

---

## 📝 Extend The Tests

### Add NewFeatures
When adding new features to the CSV:

1. **Aktualisiere `EXPECTED_FEATURE_COLUMNS`** (Zeile 27)
```python
EXPECTED_FEATURE_COLUMNS = [
    # ... existing columns
    "new_feature_name",  # Add here
]
```

2. **Optional: Add specific Tests**
```python
def test_new_feature_is_valid(self, training_df: pd.DataFrame):
    """Test that new_feature has valid range."""
    assert training_df["new_feature"].min() >= 0
    assert training_df["new_feature"].max() <= 100
```

### Add New Targets
For new Target columns (z.B. Task 3):

1. **Update `EXPECTED_TARGET_COLUMNS`** (rows 67)
```python
EXPECTED_TARGET_COLUMNS = [
    "pk_target",
    "composite_pk_target",
    "fk_target",
    "composite_fk_target",
    "normalization_target",  # Add new target
]
```

Tests run automatically for all targets!

---

## 🎯 Best Practices

### When these tests should run:
1. ✅ **Before every training run** - Local checks
2. ✅ **In CI/CD** - Automated validation
3. ✅ **After data pipeline updates** - Regression Tests
4. ✅ **For new data sources** - Integration Tests

### What to do if tests fail:
1. **Do not skip!** - Fixing errors early saves time
2. **Find the root cause** - Is it the DB query or the CSV import?
3. **Adjust the test** - Only if the requirements have changed
4. **Fix the data** - Usually, it is a data quality issue

---

## 📚 Additional Resources

- **Great Expectations:** For production-grade data testing
- **Pandera:** Schema validation for Pandas DataFrames
- **dbt Tests:** For testing directly inside the database

---

## 🔗 Related Tests

- `tests/test_api/` - API Endpoint Tests
- `tests/test_models/` - Model Loading Tests
- `tests/README.md` - General Test Overview

---

**Success with your data quality endeavors! 🚀**
