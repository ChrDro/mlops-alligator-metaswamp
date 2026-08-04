"""Pydantic request and response models for the subject-area endpoint."""

from pydantic import BaseModel


# Unlike the other five request models, these two fields are text rather than column
# statistics: a subject area is a property of what a table is *called* and what it
# holds, not of how unique its values are. Field order matches the order the model
# signature was inferred from - see FEATURE_COLUMNS in the task_4 training script.
class SubjectArea(BaseModel):
    table_name: str
    # The table's column names, comma-separated - the same shape the training script
    # builds by joining `column_name` per table, e.g. "order_id, customer_id, total".
    columns: str


# The response reuses the request fields and appends the model output, matching the
# other endpoints. `prediction` is the subject-area name (a string here, an int class
# elsewhere) and `probability` is the model's confidence in that name.
class SubjectAreaPrediction(SubjectArea):
    prediction: str
    probability: float
