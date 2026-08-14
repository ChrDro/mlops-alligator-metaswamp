# Training Data Quality Tests: Normal Form Classification

This test suite validates the quality of the schema profiling training dataset (`data/summary_output_task_1_2_training.csv`) before the classification model is trained. It enforces data integrity, mathematical constraints, and relational database logic to prevent common errors caused by poor data quality, missing features, or data leakage.

## 🎯 Why These Tests?

Since the training features are aggregated and profile data across databases, schemas, and tables (e.g., via Trino summaries), several issues can introduce corrupted data or structural inconsistencies:

*   **CSV Import Side-Effects:** Schema changes or parsing issues can accidentally convert data types or shift ranks.
*   **Query Logic Errors:** Complex structural aggregations can produce conflicting feature flags.
*   **Data Leakage:** Inconsistent global metrics within identical tables break cross-validation.
*   **Theoretical Deviations:** Machine learning models rely on rigorous mathematical foundations (e.g., partial dependencies cannot exist without a composite key).

These tests run automatically to catch data issues early before the resource-intensive training step begins.

---

## 📊 Test Suite Categories

The pipeline validates **37 distinct tests** categorized into 4 core groups:

### 1. Schema & Data Type Validation ⭐⭐⭐
*   **`test_required_columns_present`**: Ensures all 50 structural, rank, relational, and target columns are present in the source file. Prevents pipeline crashes.
*   **`test_binary_and_ratio_bounds`**: Guarantees that flags (`is_unique`, `is_composite_key_part`, etc.) are strictly binary (0 or 1) and that all ratios sit strictly between `0.0` and `1.0`.

### 2. Data Distribution & Targets ⭐⭐
*   **`test_target_variable`**: Assures that `target_normal_form` contains valid labels and no unexpected NULL values.
*   **Target Distribution Balance**: Protects model performance by validating that the distribution is not severely unbalanced (<1% or >99%) and that both classes exist.

### 3. Business Logic & Column Consistency ⭐⭐⭐
*   **`test_column_level_consistency`**: Verifies strict relational dependencies on a column level:
    *   If `is_unique=1` ➔ `unique_ratio` must be exactly `1.0`.
    *   If `is_non_null=1` ➔ `null_count` and `null_ratio` must be exactly `0.0`.
    *   Ensures `unique_ratio` aligns mathematically with `number_unique_values / count`.

### 4. Relational Normal Form Logic ⭐⭐⭐
*   **`test_table_level_group_consistency`**: Checks that table-level aggregates (e.g., `table_row_count`, `table_column_count`, `table_has_composite_pk`) remain perfectly identical across all columns belonging to the exact same table. This completely prevents data leakage during validation splits.
*   **`test_normal_form_logic_constraints`**: Enforces strict database theory conditions:
    *   If a single column is marked as a 1NF violator (`is_this_col_violating_1nf=1`), the parent table flag must reflect this.
    *   A partial dependency flag (`is_this_col_partial_dependency=1`) can *only* exist if the table is verified to have a composite primary key (`table_has_composite_pk=1`).

---

## 🚀 Execution & Integration

These checks are designed to run automatically in your **CI/CD pipeline as the very first task** before model training triggers.

### Running Locally
To run the verification suite manually, install dependencies and execute the script via terminal:

```bash
# Install dependencies
pip install pytest pandas

# Run tests with detailed verbose output
pytest test_normal_form_data.py -v

# Run tests showing print statements for data distributions
pytest test_normal_form_data.py -v -s
```

### When to Execute
1.  ✅ **Before Every Training Run** – Local validation during development.
2.  ✅ **Inside CI/CD Workflow** – Automated block before deploying a model pipeline.
3.  ✅ **After Data Pipeline Updates** – Regression testing if the Trino extractor query changes.
4.  ✅ **For New Data Sources** – Structural check whenever profiling new databases.

---

## 🛠️ Common Errors & Troubleshooting

### ❌ `"Training data not found"`
*   **Problem:** The source summary CSV path is invalid or missing.
*   **Solution:** Execute the data aggregation pipeline to regenerate `summary_output_task_1_2_training.csv`.

### ❌ `"Missing columns in the CSV"`
*   **Problem:** An essential feature (e.g., `unique_ratio`) is absent.
*   **Solution:** Verify the latest schema changes in the query pipeline and rerun the data extraction.

### ❌ `"Inconsistency: is_unique=1 but unique_ratio < 1.0"`
*   **Problem:** Feature calculation error during table profiling.
*   **Solution:** Check the calculation logic inside your extraction summary modules (e.g., `get_trino_summaries`).

### ❌ `"Theoretical mismatch: Partial dependency reported without a composite primary key"`
*   **Problem:** Relational rules are broken. A partial functional dependency (2NF violation) was flagged on a table that lacks a multi-column key.
*   **Solution:** Inspect the key extraction rules. A column might have been incorrectly flagged, or the primary key candidate detection failed to register the composite structure.

---

## 📚 Additional Resources
*   **Pandera:** Lightweight schema validation for pandas DataFrames.
*   **Great Expectations:** For advanced production-grade pipeline monitoring.
