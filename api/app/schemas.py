from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1)


class PredictionResponse(BaseModel):
    label: str
    score: float
