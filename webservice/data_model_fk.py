"""Pydantic request and response models for the prediction API."""

from pydantic import BaseModel


# This model describes the JSON body accepted by POST /predict_fk.
# Order matters as predict.py hands same frame to the
# raw sklearn estimator for predict_proba, which compares feature names positionally.
#
# The seven cross-table features below `table_integer_column_count` were added on
# 2026-08-05 and raise the model's CV F1 from 0.696 to 0.810. They sit between the
# single-table features and the column_type dummies because that is where the training
# script produces them: src/cross_table_features.py appends them to the raw frame, and
# pd.get_dummies then moves the column_type dummies to the end.
#
# Unlike every other feature here, these seven describe the *other* tables in the same
# schema, so prefect/pk_fk_pipeline.py computes them against the whole schema rather
# than the table being profiled. A caller that fills them with zeros is telling the model
# "this schema has no other tables", which is a valid input but throws away the signal.
class ForeignKey(BaseModel):
    number_unique_values: int
    null_count: int
    null_ratio: float
    is_unique: int
    ordinal_position: int
    unique_ratio: float
    is_first_column: int
    relative_ordinal_position: float
    is_first_unique_column: int
    table_column_count: int
    table_unique_column_count: int
    table_row_count: int
    table_has_unique_column: int
    table_near_unique_column_count: int
    table_non_null_column_count: int
    table_max_unique_ratio: float
    unique_ratio_rank: int
    null_ratio_rank: int
    name_ends_with_id: int
    name_contains_key: int
    name_contains_table_name: int
    name_is_singular_table_id: int
    name_length: int
    table_integer_column_count: int
    n_other_tables_with_same_column_name: int
    name_unique_in_other_table: int
    is_non_unique_and_name_unique_elsewhere: int
    name_references_other_table_exact: int
    name_references_other_table_fuzzy: int
    name_ends_with_id_no_underscore: int
    name_ends_with_code_or_num: int
    column_type_boolean: bool
    column_type_date: bool
    column_type_decimal: bool
    column_type_double: bool
    column_type_integer: bool
    column_type_varchar: bool


# The response reuses every request feature and appends the model output.
# The same shape is also forwarded to Evidently for monitoring.
class ForeignKeyPrediction(ForeignKey):
    prediction: int
    probability: float
