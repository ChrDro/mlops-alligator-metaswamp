"""Pydantic request and response models for the prediction API."""

from pydantic import BaseModel


# This model describes the JSON body accepted by POST /predict_normalform.
# Each field is one feature used by the registered MLflow model.
#
# GENERATED from nf_features.FEATURE_COLUMNS - do not hand-edit. The field ORDER is the
# order the training frame has, which is what the MLflow signature records, and getting it
# wrong shifts every value onto the wrong feature without any error.
#
# Phase 2 of TASK_3_PLAN.md replaced the previous 29-feature set wholesale. Those features
# were metadata heuristics (name_ends_with_id, table_has_composite_pk, ...) plus two that
# were copied from the label itself; these are measured on the data. The old set is not
# commented out here the way Phase 0's drops were - none of it survived, so a per-field
# reason would say the same thing 29 times. See TASK_3_PLAN.md 1.1-1.8.
#
# Three consumers read this list:
#   - the API contract enforced here,
#   - MODEL_FEATURES in prefect/normalform_pipeline.py (imported from nf_features, not copied),
#   - evidently_service/build_monitoring_references.py, which reads model_fields directly.
class NormalForm(BaseModel):
    ordinal_position: int
    col_unique_ratio: float
    col_null_ratio: float
    col_list_like_ratio: float
    col_mean_token_length: float
    col_mean_token_count: float
    col_separator_density: float
    col_is_freetext_name: int
    col_violates_1nf: int
    col_in_repeating_group: int
    col_in_candidate_key: int
    col_is_prime: int
    col_determines_count: int
    col_depends_on_count: int
    table_row_count: int
    table_sampled: int
    table_sample_ratio: float
    table_search_truncated: int
    table_pair_coverage: float
    table_ucc_search_coverage: float
    table_column_count: int
    table_max_list_like_ratio: float
    table_ratio_list_like_columns: float
    table_repeating_group_ratio: float
    table_constant_column_ratio: float
    table_has_no_ucc_le3: int
    table_candidate_key_count: int
    table_key_size: int
    table_composite_key_count: int
    table_prime_ratio: float
    table_fd_count: int
    table_fd_ratio: float
    table_partial_fd_count: int
    table_partial_fd_ratio: float
    table_transitive_fd_count: int
    table_transitive_fd_ratio: float
    table_strict_partial_fd_count: int
    table_strict_transitive_fd_count: int
    table_near_fd_count: int
    table_near_fd_ratio: float
    table_max_near_fd_strength: float
    table_near_ucc_count: int
    table_has_near_key_but_no_key: int

    # One-hot column type. `bigint` is the reference category and has
    # no field of its own: all-zero dummies mean that type. An unseen type (decimal,
    # timestamp) therefore reads as the reference - see E3 in TASK_3_PLAN.md.
    column_type_date: bool
    column_type_double: bool
    column_type_integer: bool
    column_type_varchar: bool


# The response reuses every request feature and appends the model output.
# The same shape is also forwarded to Evidently for monitoring.
class NormalFormPrediction(NormalForm):
    prediction: int
    probability: float
    # The full per-class distribution, keyed by class label as a string ("0".."3").
    # `probability` alone is just its max - the normalform pipeline's table-level
    # aggregation needs the whole thing to weigh a confident column's vote more than
    # an unsure one's, which max() on its own cannot do.
    probabilities: dict[str, float]
