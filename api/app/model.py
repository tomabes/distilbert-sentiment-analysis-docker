import os
from functools import lru_cache

from transformers import pipeline


MODEL_DIR = os.getenv("MODEL_DIR", "/models/distilbert-sst2")


@lru_cache(maxsize=1)
def load_model():
    return pipeline(
        "sentiment-analysis",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
    )


def predict_sentiment(text: str) -> dict[str, float | str]:
    classifier = load_model()
    result = classifier(text)[0]

    return {
        "label": result["label"],
        "score": float(result["score"]),
    }
