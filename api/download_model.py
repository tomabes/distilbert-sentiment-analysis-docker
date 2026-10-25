import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_DIR = os.getenv("MODEL_DIR", "/models/distilbert-sst2")


def download_model() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)


if __name__ == "__main__":
    download_model()
