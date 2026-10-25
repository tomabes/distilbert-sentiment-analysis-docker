from fastapi import FastAPI

from app.model import predict_sentiment
from app.schemas import PredictionRequest, PredictionResponse


app = FastAPI(title="DistilBERT Inference API")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    prediction = predict_sentiment(request.text)
    return PredictionResponse(**prediction)
