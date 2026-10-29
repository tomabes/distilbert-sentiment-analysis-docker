# DistilBERT Sentiment Analysis with Docker

Containerized NLP application for classifying English review text as positive or negative. A FastAPI inference service and a Gradio user interface run as separate Docker services; a standalone version is deployed on Hugging Face Spaces.

[Try the live Hugging Face Space](https://yabusiki-distilbert-sentiment-analysis.hf.space/) · [View the Space repository](https://huggingface.co/spaces/Yabusiki/distilbert-sentiment-analysis)

![DistilBERT sentiment-analysis interface on Hugging Face Spaces](report_assets/Screenshot%202026-10-25%20at%2011.02.21.png)

## Architecture

```text
Browser -> Gradio UI container -> FastAPI container -> DistilBERT model
```

- `api`: loads the pretrained model and exposes `/predict` and `/health` endpoints;
- `ui`: collects review text and sends HTTP requests to the API service;
- Docker Compose configures service dependency, health checking, shared memory and resource limits;
- `hf_space`: provides a standalone Gradio deployment for Hugging Face Spaces.

The API returns a sentiment label and confidence score. The Dockerfiles separate dependency installation, model download and application code to improve build-cache reuse.

## Run with Docker Compose

```bash
docker compose build
docker compose up
```

Open `http://localhost:7860` for the interface. Test the API directly:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I really enjoyed this assignment."}'
```

## Run the API container manually

```bash
docker build -t distilbert-api:latest ./api
bash scripts/run_api_container.sh
```

## Repository structure

```text
api/            # FastAPI application, model download and Dockerfile
ui/             # Gradio client and Dockerfile
hf_space/       # standalone Hugging Face Spaces application
scripts/        # local API tests and container helpers
report_en.pdf   # architecture, experiments and discussion
docker-compose.yml
```

The complete implementation and resource experiments are documented in the [English project report](report_en.pdf).

## Tools

Python, DistilBERT, Transformers, PyTorch, FastAPI, Gradio, Docker, Docker Compose and Hugging Face Spaces.
