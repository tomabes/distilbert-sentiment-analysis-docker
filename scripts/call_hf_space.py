import os

from gradio_client import Client


HF_SPACE = os.environ["HF_SPACE"]


def main() -> None:
    client = Client(HF_SPACE)
    result = client.predict(
        "I really enjoyed this assignment.",
        api_name="/classify_text",
    )
    print(result)


if __name__ == "__main__":
    main()
