# Deployment of DistilBERT for Sentiment Analysis in Docker and Hugging Face Spaces

## Abstract

This work deploys the pre-trained DistilBERT deep learning model for binary sentiment analysis of English-language reviews. The main local implementation is divided into two Docker containers. The first container contains the model and the FastAPI inference service, while the second contains only the Gradio graphical interface and an HTTP client. The containers are built and launched together using Docker Compose. In addition, a standalone cloud application was created in Hugging Face Spaces using Gradio, ZeroGPU, and a published API endpoint.

The solution was verified in three ways: with a direct HTTP request to the local FastAPI through the terminal, with inference through the local graphical interface, and with a remote call through the terminal and the Hugging Face Space web interface using the Gradio API. The state of the containers and their CPU, memory, network, block I/O, and process consumption were also analyzed.

To obtain the maximum grade, improvements were also made to the Dockerfile and Compose configuration, and resource limit parameters were set.

---

## 1. Task Definition and Compliance with the Requirements

The objective of the assignment is to gain practical experience in selecting, configuring, and deploying a deep learning architecture in Docker containers.


| Requirement | Implementation | Verification |
|---|---|---|
| Pre-trained DL model | `distilbert-base-uncased-finetuned-sst-2-english` | The model loads and returns a label and score |
| Inference container | FastAPI + Transformers + PyTorch | Image `distilbert-api:latest` |
| Separate GUI container | Gradio + Requests, without the model | Image `distilbert-ui:latest` |
| Docker Compose | Services `api` and `ui` | Both containers are running, and the API has the `healthy` status |
| Direct local inference | `POST /predict` via `curl` | `POSITIVE`, score `0.99986` |
| Graphical inference | Page `localhost:7860` | `NEGATIVE (0.9998)` for a negative example |
| Resource management | `mem_limit`, `shm_size` | Parameters are set separately for the two services |
| Dockerfile improvements | Separation of dependency, model, and code layers; health check | Docker cache and API readiness control are used |
| Cloud deployment | Hugging Face Space on ZeroGPU | The public interface works |
| Online API | Gradio Client from a temporary Docker container | `POSITIVE (0.9999)` |

---

## 2. Selected Model and Inference Task

### 2.1. Model

A compact version of the BERT model, fine-tuned for sentiment analysis on SST-2 (Stanford Sentiment Treebank), was used:

```text
distilbert-base-uncased-finetuned-sst-2-english
```

The model is fine-tuned for binary sentiment classification (`finetuned-sst-2`). An English-language review is provided as input. The output contains:

- `label`: `POSITIVE` or `NEGATIVE`;
- `score`: the model's confidence in the predicted class.

DistilBERT was selected because it is already pre-trained and publicly available, and its result is easy to verify manually. No training was performed.

---

## 3. Overall Project Architecture

Main architecture:

```text
User in a browser
        |
        | http://localhost:7860
        v
+-------------------------+
| UI container            |
| Gradio + requests       |
| distilbert-ui:latest    |
+-------------------------+
        |
        | POST http://api:8000/predict
        | JSON: {"text": "..."}
        v
+-------------------------+
| API container           |
| FastAPI + Transformers  |
| PyTorch + DistilBERT    |
| distilbert-api:latest   |
+-------------------------+
        |
        v
 {"label": "...", "score": ...}
```

Cloud version:

```text
User or Gradio Client
        |
        v
+----------------------------------+
| Hugging Face Space               |
| Gradio + Transformers + model    |
| @spaces.GPU -> ZeroGPU           |
+----------------------------------+
```

Hugging Face Space: https://huggingface.co/spaces/Yabusiki/distilbert-sentiment-analysis.

### 3.1. File Structure

```text
docker-dl-assignment/
├── api/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── model.py
│   │   └── schemas.py
│   ├── .dockerignore
│   ├── Dockerfile
│   ├── download_model.py
│   └── requirements.txt
├── ui/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── main.py
│   ├── .dockerignore
│   ├── Dockerfile
│   └── requirements.txt
├── hf_space/
│   ├── README.md
│   ├── app.py
│   └── requirements.txt
├── scripts/
│   ├── call_hf_space.py
│   ├── run_api_container.sh
│   └── test_local_api.sh
├── report_assets/
├── .dockerignore
├── docker-compose.yml
└── README.md
```

### 3.2. Local Environment Setup

The project was completed on a MacBook Air with an M1 processor, 8 CPU cores, 8 GB of unified memory, and ARM64 architecture. On macOS, Linux containers are not run by the macOS kernel, so Docker Desktop was installed. In addition to the graphical interface, it provides the Docker CLI accessed by the terminal; the Docker daemon, which manages images and containers; a Linux VM in which Linux containers run; and virtual storage for images, layers, and containers.

While Docker Desktop was running, commands were entered in Terminal.

The availability of the client tools was checked with the following commands:

```bash
docker --version
docker compose version
```

The following versions were obtained:

```text
Docker version 29.9.2
Docker Compose version v5.6.1
```

The project directory was then opened:

```bash
cd '/Users/toma/Downloads/STUDY & MASTERS/+UJM/M2 2026:27/M2 1st semester/AI Infrastructure/docker-dl-assignment'
```

The list of services was checked:

```bash
docker compose config --services
```

Result:

```text
api
ui
```

This command confirms that the Compose file can be parsed syntactically and defines both services.

---

## 5. `api` Inference Service

### 5.1. Purpose

The API container is responsible only for ML inference:

1. it accepts an HTTP request containing text;
2. validates the JSON structure;
3. loads the locally stored tokenizer and model;
4. runs the Transformers pipeline;
5. returns `label` and `score`.

### 5.2. Dependencies in `api/requirements.txt`

- **FastAPI (fastapi)** — a Python framework for creating an HTTP API; it describes routes and data.
- **Uvicorn (uvicorn[standard])** — an ASGI server that listens on a port and forwards requests to the FastAPI application.
- **Transformers (transformers)** — loading the tokenizer, model, and ML pipeline.
- **PyTorch (torch)** — the computation backend for the neural network.

### 5.3. Model Preloading

The `api/download_model.py` file is executed while the image is being built:

```python
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_DIR = os.getenv("MODEL_DIR", "/models/distilbert-sst2")

def download_model() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
```

`AutoTokenizer.from_pretrained` downloads the tokenization rules and vocabulary; `AutoModelForSequenceClassification.from_pretrained` downloads the architecture and weights; `save_pretrained` saves both components inside the image.

The advantage of this solution is that the model does not need to be downloaded from the Internet again every time the container starts. The cost of this solution is a larger image and a longer initial build.

### 5.4. Model Loading and Prediction

The inference logic is defined in `api/app/model.py`:

```python
@lru_cache(maxsize=1)
def load_model():
    return pipeline(
        "sentiment-analysis",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
    )
```

The pipeline receives a local path inside the image. The `@lru_cache(maxsize=1)` decorator stores the first pipeline that is created and returns the same object. Therefore, the model is loaded into memory once rather than for every HTTP request.

Prediction function:

```python
def predict_sentiment(text: str) -> dict[str, float | str]:
    classifier = load_model()
    result = classifier(text)[0]

    return {
        "label": result["label"],
        "score": float(result["score"]),
    }
```

In this function, `load_model()` returns the cached pipeline; the tokenizer converts the text into token IDs; PyTorch runs DistilBERT; the pipeline creates a list of results; `[0]` selects the result for the first text; and the function returns a Python dictionary.

### 5.5. Data Schemas

`api/app/schemas.py` contains Pydantic models:

```python
class PredictionRequest(BaseModel):
    text: str = Field(min_length=1)

class PredictionResponse(BaseModel):
    label: str
    score: float
```

`PredictionRequest` requires a non-empty string field named `text`. If the JSON has an incorrect structure, FastAPI returns a validation error before calling the model. `PredictionResponse` defines the response format: a string label and a numeric score.

### 5.6. HTTP Routes

`api/app/main.py` creates the application:

```python
app = FastAPI(title="DistilBERT Inference API")
```

The health-check route indicates that the web service is available:

```python
@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
```

Inference route:

```python
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    prediction = predict_sentiment(request.text)
    return PredictionResponse(**prediction)
```

FastAPI converts the input JSON into `PredictionRequest`. `request.text` is passed to the model. The expression `PredictionResponse(**prediction)` expands the dictionary as named arguments.

### 5.7. API Dockerfile

The Dockerfile contains the following key instructions:

- `FROM python:3.11-slim` — the base Linux image with Python 3.11. The `slim` variant reduces the base layer by excluding system components that are not needed for the project.
- `PYTHONDONTWRITEBYTECODE=1` — do not create `.pyc` files.
- `PYTHONUNBUFFERED=1` — output logs to the Docker stream.
- `HF_HOME=/models` — use `/models` for Hugging Face data.
- `MODEL_DIR` — a single path where the script saves and the application loads the model.
- `WORKDIR /app` — subsequent relative paths are resolved from `/app`.
- `COPY requirements.txt .` and a separate `RUN pip install` — the dependency layer is cached independently of code changes.
- `RUN python download_model.py` — the model becomes part of the image.
- `COPY app ./app` is performed last among the copy operations, so changing the application code does not require reinstalling the libraries and downloading the model again.
- `EXPOSE 8000` documents the internal port.
- `HEALTHCHECK` calls `/health` every 30 seconds.
- `ENTRYPOINT` starts Uvicorn and specifies the `app` object from the `app.main` module.
- `CMD` adds the address and port. `0.0.0.0` is required so that the server accepts requests not only from within its own process.

The layers are ordered so that dependencies that change infrequently precede code that changes more frequently.

### 5.8. Build

The build was started with the following command:

```bash
docker compose build
```

![Start of the Docker image build](report_assets/Screenshot%202026-10-24%20at%2016.25.46.png)

*Figure 1. Docker Compose reads two Dockerfiles, obtains the `python:3.11-slim` base image, and starts building the API and UI services.*

![Built local Docker images](report_assets/Screenshot%202026-10-25%20at%2000.24.08.png)

*Figure 2. Two local images in Docker Desktop: the large `distilbert-api` and the significantly smaller `distilbert-ui`.*

In the screenshot, Docker Desktop shows `10.35 GB` for the API and `681.52 MB` for the UI. A possible future improvement is to explicitly install a CPU-only build of PyTorch for the local API. This would reduce the image size and disk usage without slowing down local inference, because an NVIDIA GPU is not available to the container on the Mac.

### 5.9. Direct Launch of the API Container

The `scripts/run_api_container.sh` script was used to create the container through the terminal:

```bash
docker run \
  -d \
  --rm \
  -p 8000:8000 \
  --shm-size=1g \
  --memory=4g \
  --name "$containerName" \
  distilbert-api:latest
```

Parameters:

- `-d` — run in the background;
- `--rm` — remove the container after it stops without removing the image;
- `-p 8000:8000` — port `8000` on the Mac is forwarded to port `8000` in the container;
- `--shm-size=1g` — the shared memory segment `/dev/shm` is increased to 1 GiB;
- `--memory=4g` — the memory limit;
- `--name` — a clear container name.

### 5.10. Direct Inference Through the API

The request was executed from the Mac terminal after Compose was launched:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I really enjoyed this assignment."}' \
  | python3 -m json.tool
```

`curl` accesses port 8000 on the host. Docker forwards the request to the API container. The header indicates that the request body is JSON. The `-d` parameter passes the `text` field. The `python3 -m json.tool` command formats the received JSON.

![Direct local API inference](report_assets/Screenshot%202026-10-25%20at%2000.50.16.png)

*Figure 3. Successful POST request to the local FastAPI: label `POSITIVE`, score `0.9998620748519897`.*

---

## 6. `ui` Graphical Service

### 6.1. Purpose and Isolation Principle

The UI container is responsible only for the user interface and HTTP requests to the API. It does not contain PyTorch, Transformers, DistilBERT weights, or a local ML inference function. The UI sends text over the network and receives the completed JSON response. The ML operation is performed in another container.

### 6.2. UI Dependencies in `ui/requirements.txt`

- **Gradio (gradio)** builds the web interface.
- **Requests (requests)** sends an HTTP POST request to FastAPI.

### 6.3. HTTP Client

`ui/app/client.py`:

```python
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
```

`API_URL` is obtained from an environment variable. The default value is `http://api:8000`, where `api` is the name of the Compose service. Docker Compose creates a shared network and an embedded DNS that resolves the name `api` to the internal IP address of the API container. From the Mac, `http://localhost:8000` is used.
`requests.post` serializes `json={"text": text}` and sends it to `/predict`. `timeout=30`. `raise_for_status()` converts 4xx/5xx HTTP responses into an exception. After a successful response, `response.json()` converts the JSON into a Python dictionary.

### 6.4. Interface Handlers

The main function in `ui/app/main.py`:

```python
def classify_text(text: str) -> str:
    cleaned_text = text.strip()

    if not cleaned_text:
        return "Please enter some text."

    return request_prediction(cleaned_text)
```

`strip()` removes spaces from the beginning and end. If the string is empty after cleaning, the HTTP request is not sent. For non-empty text, the FastAPI client is called.

Two helper functions insert prepared reviews and perform analysis at the same time:

```python
def use_positive_review() -> tuple[str, str]:
    return POSITIVE_REVIEW, classify_text(POSITIVE_REVIEW)

def use_negative_review() -> tuple[str, str]:
    return NEGATIVE_REVIEW, classify_text(NEGATIVE_REVIEW)
```

The tuple contains two results: the text for the input field and the prediction for the output field.

### 6.5. Building the Gradio UI

```python
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
```

`gr.Blocks` defines the page. `Textbox` stores the input, `Row` arranges the buttons horizontally, and the second `Textbox` displays the result.

The button event connects the components to the Python function:

```python
analyze_button.click(
    fn=classify_text,
    inputs=review_text,
    outputs=prediction_output,
)
```

On click, the contents of `review_text` are passed to `classify_text`, and the returned string is placed in `prediction_output`. `review_text.submit(...)` makes it possible to run the same function by submitting the form from the keyboard.

Launch:

```python
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

`0.0.0.0` allows connections from outside the container. If only `127.0.0.1` were used, the server would be available only inside the UI container.

### 6.6. UI Dockerfile

The UI Dockerfile is simpler than the API Dockerfile because it does not download the model. The `python -m app.main` command launches `main.py`.

### 6.7. GUI Verification

After launch, the following address was opened in a browser:

```text
http://localhost:7860
```

A negative review was entered:

```text
This product was disappointing and difficult to use
```

![Local graphical inference](report_assets/Screenshot%202026-10-25%20at%2000.52.30.png)

*Figure 4. The local Gradio UI received the `NEGATIVE (0.9998)` result from the separate FastAPI container.*

---

## 7. Orchestration with Docker Compose

Docker Compose is part of the Docker CLI for describing and jointly launching multiple services. Its configuration is stored in the `docker-compose.yml` YAML file.

### 7.2. API Service

```yaml
api:
  build:
    context: ./api
  image: distilbert-api:latest
  container_name: distilbert_api
  ports:
    - "8000:8000"
  shm_size: "1gb"
  mem_limit: "4g"
```

- `context: ./api` defines the build context. Docker sees the files inside the `api` directory.
- `image` specifies the name of the built image.
- `container_name` specifies the name of the instance.
- `ports` publishes the API on the Mac.
- `shm_size` increases shared memory.
- `mem_limit` specifies the upper RAM limit.

### 7.3. UI Service

```yaml
ui:
  build:
    context: ./ui
  image: distilbert-ui:latest
  container_name: distilbert_ui
  ports:
    - "7860:7860"
  environment:
    - API_URL=http://api:8000
  depends_on:
    api:
      condition: service_healthy
  shm_size: "512mb"
  mem_limit: "1g"
```

`API_URL` passes the internal API address to the UI. `depends_on` with `condition: service_healthy` creates the API first and waits for a successful Docker health check, because starting the process does not yet guarantee that the web server is ready to accept requests.

The UI receives lower limits because it does not store the model and PyTorch.

### 7.4. Launch

```bash
docker compose up
```

Compose created a network and two containers:

![Launching services with Docker Compose](report_assets/Screenshot%202026-10-25%20at%2000.15.57.png)

*Figure 5. The API starts Uvicorn, the health check returns `200 OK`, and the UI is then allowed to start.*

### 7.5. Status Verification

```bash
docker compose ps
```

It showed the state of the containers:

![Container status](report_assets/Screenshot%202026-10-25%20at%2000.46.04.png)

*Figure 6. Both containers are in the `Up` state; the API additionally has the `healthy` status; ports 8000 and 7860 are published on the host.*

### 7.6. Resource Consumption

A one-time snapshot was obtained with the following command:

```bash
docker stats --no-stream
```

![Container resource statistics](report_assets/Screenshot%202026-10-25%20at%2001.13.15.png)

*Figure 7. Snapshot of CPU, memory, network and block I/O, and the number of processes for the two containers.*

Observed values:

| Metric | UI | API | Interpretation |
|---|---:|---:|---|
| CPU | 7.38% | 0.25% | Instantaneous snapshot: the UI may have been serving the browser, while the API had already completed inference |
| Memory usage | 126.7 MiB | 991.2 MiB | The API holds PyTorch, Transformers, the tokenizer, and model weights |
| Memory limit | 1 GiB | 3.825 GiB | The UI received its limit; the API is limited by the memory actually available to the Docker VM |
| Memory % | 12.37% | 25.31% | The percentage is calculated relative to the limit shown for the container |
| Network I/O | 83.5 kB / 428 kB | 8.2 kB / 4.77 kB | The UI serves browser resources; the API exchanges compact JSON |
| Block I/O | 126 MB / 12.3 kB | 1.25 GB / 4.1 kB | The API read significantly more data because of the libraries and model weights |
| PIDs | 11 | 29 | The ML runtime and API server use more processes/threads |

---

## 8. Hugging Face Spaces

### 8.1. Purpose of the Cloud Part

The additional part demonstrates that the model can be deployed outside the local computer and called through a public online API. The `Yabusiki/distilbert-sentiment-analysis` Space was created with the Gradio SDK, ZeroGPU hardware, and public visibility.

### 8.2. Difference in the Cloud Architecture

The local part must have two containers, so the UI calls FastAPI. In the Space, the interface and model are placed in one Python application:

```text
Gradio event -> Python function -> Transformers pipeline -> result
```

### 8.3. Dependencies in `hf_space/requirements.txt`

Only `transformers` is installed because Gradio is already defined by the SDK configuration, while the infrastructure is provided by ZeroGPU.

### 8.4. Pipeline Loading

```python
classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=0,
)
```

Here, the model is specified by its Hub name rather than by a local path. When the Space started, the following files were downloaded:

- `config.json`;
- `model.safetensors`, approximately 268 MB in size;
- `tokenizer_config.json`;
- `vocab.txt`.

`device=0` refers to the first GPU. The pipeline is created once at startup.

### 8.5. Inference and ZeroGPU

This function runs the model and formats the result:

```python
def predict_sentiment(text: str) -> str:
    result = classifier(text)[0]
    return f'{result["label"]} ({float(result["score"]):.4f})'
```

Main handler:

```python
@spaces.GPU(duration=30)
def classify_text(text: str) -> str:
    cleaned_text = text.strip()

    if not cleaned_text:
        return "Please enter some text."

    return predict_sentiment(cleaned_text)
```

`@spaces.GPU(duration=30)` informs ZeroGPU that the function needs a temporary GPU. `duration=30` is the requested upper GPU-time interval.

### 8.6. Online GUI Inference

The following text was entered in the public Space:

```text
The application is simple, fast and very useful
```

![Inference in Hugging Face Space](report_assets/Screenshot%202026-10-25%20at%2011.02.21.png)

*Figure 8. The public Hugging Face Space runs on ZeroGPU and returns `POSITIVE (0.9997)`.*

### 8.10. Online API Inference from a Docker Container

The `scripts/call_hf_space.py` script uses the Gradio Client:

```python
HF_SPACE = os.environ["HF_SPACE"]

def main() -> None:
    client = Client(HF_SPACE)
    result = client.predict(
        "I really enjoyed this assignment.",
        api_name="/classify_text",
    )
    print(result)
```

The Space name is passed through an environment variable. The client connects to the published Space and calls the endpoint.

```bash
docker run --rm \
  -e HF_SPACE=Yabusiki/distilbert-sentiment-analysis \
  --mount "type=bind,source=$PWD/scripts,target=/scripts,readonly" \
  distilbert-ui:latest \
  python /scripts/call_hf_space.py
```

The parameters mean:

- `--rm` — remove the temporary container after the call;
- `-e HF_SPACE=...` — pass the Space name;
- `type=bind` — mount an existing Mac directory;
- `source=$PWD/scripts` — the source on the host;
- `target=/scripts` — the path inside the container;
- `readonly` — prevent the container from modifying the local scripts;
- `distilbert-ui:latest` — use the lightweight image in which Gradio Client is already installed as part of Gradio;
- the last line starts the API client.

![Remote call to the Hugging Face API from a Docker container](report_assets/Screenshot%202026-10-25%20at%2011.09.32.png)

*Figure 9. A temporary local Docker container connected to the public Space and received `POSITIVE (0.9999)`.*

---

## 9. Comparison of Local and Cloud Approaches

| Characteristic | Local Docker Containers | Hugging Face Space |
|---|---|---|
| Architecture | Two services: UI and API | One Gradio application |
| GUI and model isolation | Explicit, through separate containers | Not used |
| API | FastAPI `/predict` | Gradio `/classify_text` |
| Model | Embedded in the API image during the build | Downloaded from the Hub when the environment starts |
| Computation | CPU in the Docker VM on the Mac | ZeroGPU |
| Access | `localhost` | Public Internet address |
| Orchestration | Docker Compose | Managed by Hugging Face |
| Resource control | `mem_limit`, `shm_size` | Determined by the selected Space hardware |
| Reproducibility | Images fix the environment | README fixes Python, Gradio, Transformers |
| Main value | Demonstration of containerization and separation of responsibilities | Demonstration of cloud deployment and an online API |

The local version better demonstrates Docker architectural principles: separate service responsibilities, an internal network, a health check, startup order, and resource limits. The cloud version is simpler for the end user because it does not require a local Docker installation and provides a public link.

---

## 10. Limitations and Possible Improvements

### 10.1. CPU-only PyTorch for the Local API

Rebuilding the local API with a CPU-only version of PyTorch could potentially:

- reduce the image size;
- remove unnecessary CUDA/NVIDIA libraries;
- reduce the risk of running out of disk space again;
- avoid reducing the performance of this local run, because an NVIDIA GPU was not used on the Mac.

### 10.2. Pinning Dependency Versions

The requirements files use package names without versions. This is convenient at the MVP stage, but the same build may receive new library versions in the future. For reproducibility, the verified versions of FastAPI, Uvicorn, Transformers, PyTorch, Gradio, and Requests can be pinned.

---

## 12. Conclusion

As a result, a sentiment analysis application based on pre-trained DistilBERT was created and verified. In the main version, ML inference and the graphical interface are isolated in separate Docker containers and connected through a local FastAPI using Docker Compose. The Dockerfiles are organized with layer caching in mind, the model is saved in the API image during the build, and Compose defines the health dependency, RAM limits, and shared memory.

A direct request to FastAPI returned `POSITIVE` with a confidence of `0.99986`, while the local Gradio interface returned `NEGATIVE (0.9998)` for a negative text. The `docker stats` analysis confirmed the expected separation of resources: the inference container uses more memory, block I/O, and processes, while the UI container remains relatively lightweight.

In addition, the application was published in Hugging Face Spaces on ZeroGPU. Its operation was verified both through the public interface (`POSITIVE (0.9997)`) and through the remote Gradio API called from a temporary Docker container (`POSITIVE (0.9999)`).

The project demonstrated the main principles of deploying ML systems: a reproducible environment, separation of service responsibilities, network communication, readiness control, resource management, and moving an application to a cloud environment.
