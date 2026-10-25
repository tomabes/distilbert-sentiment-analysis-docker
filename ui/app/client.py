import os

import requests


API_URL = os.getenv("API_URL", "http://api:8000")


def request_prediction(text: str) -> str:
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text},
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    return f'{data["label"]} ({data["score"]:.4f})'
