"""Pydantic response models for the health endpoints."""

from typing import Literal

from pydantic import BaseModel, Field


class LivenessStatus(BaseModel):
    """Response of GET /health/live."""

    status: Literal["alive"] = "alive"


class ReadinessStatus(BaseModel):
    """Response of GET /health/ready."""

    status: Literal["ok", "degraded", "unavailable"] = Field(
        description=(
            "ok when every model resolves, degraded when only some do (the service "
            "still serves those), unavailable when none do."
        ),
    )
    alias: str = Field(description="Registry alias the service serves.")
    models: dict[str, str | None] = Field(
        description=(
            "Registered model name to the version the alias resolves to, or null when "
            "the model, the alias or the registry itself is unavailable."
        ),
    )
    detail: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "degraded",
                    "alias": "dev",
                    "models": {"pk_model": "3", "fk_model": None},
                    "detail": "1 of 2 models resolve for alias 'dev'. Not resolving: fk_model.",
                },
            ],
        },
    }
