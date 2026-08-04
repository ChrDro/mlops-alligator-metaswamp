"""Pydantic request and response models for the prediction API."""

from pydantic import BaseModel


# This model describes the JSON body accepted by POST /predict.
# Each field is one feature used by the registered MLflow model.
#
# Phase 0 of TASK_3_PLAN.md cut the set from 50 to 29: 21 features carried no
# information of their own (constant in the training data, bit-for-bit identical to
# another column, or a threshold/ratio derivation of one). They are commented out
# rather than deleted so the drop reason travels with the schema. Field ORDER matches
# the training frame (CSV order minus the drops, dummies appended), which is what the
# MLflow signature records.
#
# This list is the single source of truth for three consumers:
#   - the API contract enforced here,
#   - MODEL_FEATURES in prefect/normalform_pipeline.py (kept in sync by hand),
#   - evidently_service/build_monitoring_references.py, which reads model_fields
#     directly and therefore needs no edit of its own.
class NormalForm(BaseModel):
    # These fields mirror the training features used by the demo model.
    # Keeping the schema small makes it easier to inspect payloads in curl,
    # send_data.py, and the Evidently report.
    # number_unique_values: int             # raw value behind unique_ratio
    # count: int                            # == table_row_count
    # null_count: int                       # constant 0 in the training set
    # null_ratio: float                     # constant 0 in the training set
    # is_unique: int                        # == (unique_ratio == 1 and null_count == 0)
    # ordinal_position: int                 # raw value behind relative_ordinal_position
    unique_ratio: float
    # is_non_null: int                      # constant 1 in the training set
    # is_first_column: int                  # == (ordinal_position == 1)
    relative_ordinal_position: float
    is_first_unique_column: int
    table_column_count: int
    # table_unique_column_count: int        # raw value behind table_ratio_of_pk_candidates
    table_row_count: int
    # other_unique_columns_in_table: int    # == table_unique_column_count - is_unique
    # table_has_unique_column: int          # == (table_unique_column_count > 0)
    # table_has_no_single_pk_candidate: int # == 1 - table_has_unique_column
    table_near_unique_column_count: int
    table_id_named_column_count: int
    # table_non_null_column_count: int      # == table_column_count (no NULLs in training)
    table_max_unique_ratio: float
    table_integer_column_count: int
    unique_ratio_rank: int
    # null_ratio_rank: int                  # == ordinal_position (no NULLs in training)
    # is_least_null_in_table: int           # == is_first_column (dito)
    # unique_ratio_relative_to_max: float   # == unique_ratio / table_max_unique_ratio
    # other_near_unique_columns_in_table: int  # == table_near_unique_column_count - (ur > .95)
    name_ends_with_id: int
    # name_contains_key: int                # constant 0 in the training set
    # name_contains_table_name: int         # constant 0 in the training set
    # name_is_singular_table_id: int        # constant 0 in the training set
    name_length: int
    table_avg_unique_ratio: float
    # table_avg_null_ratio: float           # constant 0 in the training set
    table_ratio_of_pk_candidates: float
    # Table-level aggregates below are deliberately KEPT. They look derivable via a
    # groupby, but the model sees one row per request and cannot aggregate - dropping
    # them collapsed holdout F1 from 0.9939 to 0.7710.
    is_this_col_violating_1nf: int
    is_composite_key_part: int
    table_has_composite_pk: int
    is_this_col_partial_dependency: int
    table_ratio_composite_key_cols: float
    table_ratio_1nf_violations: float
    table_std_unique_ratio: float
    table_has_partial_dependency: int
    column_type_char: bool
    column_type_date: bool
    column_type_decimal: bool
    column_type_double: bool
    column_type_integer: bool
    column_type_timestamp: bool
    column_type_varchar: bool


# The response reuses every request feature and appends the model output.
# The same shape is also forwarded to Evidently for monitoring.
class NormalFormPrediction(NormalForm):
    prediction: int
    probability: float
