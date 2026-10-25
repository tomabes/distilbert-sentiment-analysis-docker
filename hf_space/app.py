import gradio as gr
import spaces
from transformers import pipeline


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

POSITIVE_REVIEW = (
    "I really enjoyed this product. "
    "It was easy to use and worked perfectly."
)
NEGATIVE_REVIEW = (
    "This was a terrible experience. "
    "The product failed and the support was slow."
)


classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=0,
)


def predict_sentiment(text: str) -> str:
    result = classifier(text)[0]
    return f'{result["label"]} ({float(result["score"]):.4f})'


@spaces.GPU(duration=30)
def classify_text(text: str) -> str:
    cleaned_text = text.strip()

    if not cleaned_text:
        return "Please enter some text."

    return predict_sentiment(cleaned_text)


@spaces.GPU(duration=30)
def use_positive_review() -> tuple[str, str]:
    return POSITIVE_REVIEW, predict_sentiment(POSITIVE_REVIEW)


@spaces.GPU(duration=30)
def use_negative_review() -> tuple[str, str]:
    return NEGATIVE_REVIEW, predict_sentiment(NEGATIVE_REVIEW)


with gr.Blocks(title="DistilBERT Sentiment Analysis") as demo:
    gr.Markdown("# DistilBERT Sentiment Analysis")

    review_text = gr.Textbox(
        label="Review text",
        lines=4,
        placeholder="Type an English review here.",
    )

    with gr.Row():
        positive_button = gr.Button("Positive review")
        negative_button = gr.Button("Negative review")
        analyze_button = gr.Button("Analyze", variant="primary")

    prediction_output = gr.Textbox(label="Prediction")

    positive_button.click(
        fn=use_positive_review,
        outputs=[review_text, prediction_output],
    )
    negative_button.click(
        fn=use_negative_review,
        outputs=[review_text, prediction_output],
    )
    analyze_button.click(
        fn=classify_text,
        inputs=review_text,
        outputs=prediction_output,
        api_name="classify_text",
    )
    review_text.submit(
        fn=classify_text,
        inputs=review_text,
        outputs=prediction_output,
    )


if __name__ == "__main__":
    demo.launch()