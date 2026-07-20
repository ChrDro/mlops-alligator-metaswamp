"""Pydantic request and response models for the prediction API."""

from pydantic import BaseModel


# This model describes the JSON body accepted by POST /predict.
# Each field is one feature used by the registered MLflow model.
class PrimaryKey(BaseModel):
    # These fields mirror the training features used by the demo model.
    # Keeping the schema small makes it easier to inspect payloads in curl,
    # send_data.py, and the Evidently report.
    number_unique_values: int
    count: int
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
    table_has_no_single_pk_candidate: int
    table_near_unique_column_count: int
    table_id_named_column_count: int
    table_non_null_column_count: int
    table_max_unique_ratio: float
    unique_ratio_rank: int
    null_ratio_rank: int
    is_least_null_in_table: int
    unique_ratio_relative_to_max: float
    name_ends_with_id: int
    name_contains_table_name: int
    name_is_singular_table_id: int
    name_length: int
    column_type_boolean: bool
    column_type_date: bool
    column_type_decimal: bool
    column_type_double: bool
    column_type_integer: bool
    column_type_varchar: bool


# The response reuses every request feature and appends the model output.
# The same shape is also forwarded to Evidently for monitoring.
class PrimaryKeyPrediction(PrimaryKey):
    prediction: int
    probability: float
