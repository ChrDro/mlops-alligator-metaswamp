"""Pydantic request and response models for the streaming trigger endpoint."""

from pydantic import BaseModel, Field


class NewDataNotification(BaseModel):
    """
    Body of POST /events/new-data.

    Deliberately minimal: the caller says *where* it loaded data, not *what*
    changed. Working out what changed is the change detector's job, so a loader
    does not have to know anything about features, tables or the queue.
    """

    schema_name: str = Field(
        default="new_predict_data",
        description="Source schema that was loaded.",
        # 'schema' is reserved on BaseModel, so the field is named schema_name
        # internally while the JSON key stays the more natural 'schema'.
        alias="schema",
        serialization_alias="schema",
    )
    note: str | None = Field(
        default=None,
        description="Optional free-text hint from the caller, stored for traceability.",
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [{"schema": "new_predict_data", "note": "nightly load finished"}],
        },
    }


class NewDataAccepted(BaseModel):
    """Response of POST /events/new-data."""

    status: str
    schema_name: str = Field(serialization_alias="schema")
    detail: str

    model_config = {"populate_by_name": True}
