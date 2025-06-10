from typing import List

from pydantic import BaseModel, Field


class EyeCoordinates(BaseModel):
    normalized: List[float] = Field(
        ...,
        example=[0.37, 0.51, 0.61, 0.51],
        description="Normalized [lx, ly, rx, ry] coordinates in [0,1] range",
    )
    pixel_original_ref_approx: List[float] = Field(
        ...,
        example=[65.9, 111.2, 108.6, 111.2],
        description="Approx. pixel coordinates [lx, ly, rx, ry] relative to typical original image dimensions (e.g., 178x218)",
    )


class PredictionResponse(BaseModel):
    age: float = Field(..., example=35.7, description="Predicted age in years")
    gender: str = Field(
        ..., example="female", description="Predicted gender ('male' or 'female')"
    )
    gender_confidence: float = Field(
        ..., example=0.9987, description="Confidence score for gender prediction (0-1)"
    )
    eye_coordinates: EyeCoordinates = Field(
        ..., description="Predicted eye coordinates"
    )
    model_type: str = Field(
        ...,
        example="keras",
        description="Type of model used for prediction ('keras' or 'pytorch')",
    )


class ErrorResponse(BaseModel):
    detail: str = Field(..., example="Model not available or error during prediction.")
