import gradio as gr

from app.client import request_prediction


POSITIVE_REVIEW = "I really enjoyed this product. It was easy to use and worked perfectly."
NEGATIVE_REVIEW = "This was a terrible experience. The product failed and the support was slow."


def classify_text(text: str) -> str:
    cleaned_text = text.strip()

    if not cleaned_text:
        return "Please enter some text."

    return request_prediction(cleaned_text)


def use_positive_review() -> tuple[str, str]:
    return POSITIVE_REVIEW, classify_text(POSITIVE_REVIEW)


def use_negative_review() -> tuple[str, str]:
    return NEGATIVE_REVIEW, classify_text(NEGATIVE_REVIEW)


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
    )
    review_text.submit(
        fn=classify_text,
        inputs=review_text,
        outputs=prediction_output,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
