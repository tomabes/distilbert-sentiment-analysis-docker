# Docker NLP Project

This project runs sentiment analysis with a pretrained DistilBERT model.

The architecture uses two Docker containers:

- `api`: FastAPI service that loads the model and performs inference.
- `ui`: Gradio interface that sends HTTP requests to the API service.

The model receives English review text and returns a sentiment label plus a confidence score.

## Run with Docker Compose

```bash
docker compose build
docker compose up
```

Open the interface:

```text
http://localhost:7860
```

Test the API directly:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I really enjoyed this assignment."}'
```

## Run the API container manually

First build the API image:

```bash
docker build -t distilbert-api:latest ./api
```

Then run:

```bash
bash scripts/run_api_container.sh
```

## Project improvements

The Dockerfiles are organized to make good use of Docker cache layers:

1. Copy dependency files.
2. Install dependencies.
3. Download the pretrained model.
4. Copy application code.

The Compose file also includes memory limits, shared memory configuration, service dependency, and an API health check.

## Hugging Face Spaces Part

The `hf_space/` directory [Yabusiki/distilbert-sentiment-analysis](https://huggingface.co/spaces/Yabusiki/distilbert-sentiment-analysis) contains a standalone Gradio version for Hugging Face Spaces.

After uploading it to a Space, the online app can also be called through the Gradio API.

The deployed application is publicly available in Hugging Face Spaces: https://yabusiki-distilbert-sentiment-analysis.hf.space/
