"""Pydantic schemas for the VLM Temporal Operation Intelligence API."""

from pydantic import BaseModel, Field
from typing import Optional


OPERATION_CLASSES = [
    "Box Setup",
    "Inner Packing",
    "Tape",
    "Put Items",
    "Pack",
    "Wrap",
    "Label",
    "Final Check",
    "Idle",
    "Unknown",
]


class TemporalSegment(BaseModel):
    start_frame: int = Field(..., ge=0, description="Start frame index of the operation")
    end_frame: int = Field(..., ge=0, description="End frame index of the operation")


class PredictionResponse(BaseModel):
    clip_id: str = Field(..., description="Unique identifier for the video clip")
    dominant_operation: str = Field(..., description="Predicted operation class")
    temporal_segment: TemporalSegment = Field(
        ..., description="Frame boundaries of the operation"
    )
    anticipated_next_operation: str = Field(
        ..., description="Predicted next operation in the workflow"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = False
    model_name: str = ""
