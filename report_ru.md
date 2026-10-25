# Развёртывание DistilBERT для анализа тональности в Docker и Hugging Face Spaces

## Аннотация

В работе развёрнута предварительно обученная модель глубокого обучения DistilBERT для бинарного анализа тональности англоязычных отзывов. Основная локальная реализация разделена на два Docker-контейнера. Первый контейнер содержит модель и FastAPI-сервис инференса, второй — только графический интерфейс Gradio и HTTP-клиент. Контейнеры собираются и запускаются совместно с помощью Docker Compose. Дополнительно создано самостоятельное облачное приложение в Hugging Face Spaces, использующее Gradio, ZeroGPU и опубликованный API endpoint.

Работоспособность решения проверена тремя способами: прямым HTTP-запросом к локальному FastAPI через терминал, инференсом через локальный графический интерфейс и удалённым вызовом через терминал и веб-интерфейс Hugging Face Space через Gradio API. Также проанализированы состояние контейнеров и потребление ими CPU, оперативной памяти, сети, дискового ввода-вывода и процессов.

Для максимальной оценки также сделаны улучшения Dockerfile и Compose-конфигурации и заданы параметры ограничения ресурсов.

---

## 1. Постановка задачи и соответствие требованиям

Цель задания — получить практический опыт выбора, конфигурации и развёртывания архитектуры глубокого обучения в Docker-контейнерах.


| Требование | Реализация | Проверка |
|---|---|---|
| Предварительно обученная DL-модель | `distilbert-base-uncased-finetuned-sst-2-english` | Модель загружается и выдаёт метку и score |
| Контейнер инференса | FastAPI + Transformers + PyTorch | Образ `distilbert-api:latest` |
| Отдельный GUI-контейнер | Gradio + Requests, без модели | Образ `distilbert-ui:latest` |
| Docker Compose | Сервисы `api` и `ui` | Оба контейнера запущены, API имеет статус `healthy` |
| Прямой локальный инференс | `POST /predict` через `curl` | `POSITIVE`, score `0.99986` |
| Графический инференс | Страница `localhost:7860` | `NEGATIVE (0.9998)` для негативного примера |
| Управление ресурсами | `mem_limit`, `shm_size` | Параметры заданы отдельно для двух сервисов |
| Улучшения Dockerfile | Разделение слоёв зависимостей, модели и кода; health check | Используется кэш Docker и контроль готовности API |
| Облачное развёртывание | Hugging Face Space на ZeroGPU | Публичный интерфейс работает |
| Онлайн API | Gradio Client из временного Docker-контейнера | `POSITIVE (0.9999)` |

---

## 2. Выбранная модель и задача инференса

### 2.1. Модель

Использована компактная версия модели BERT, дообученная для анализа тональности на SST-2 (Stanford Sentiment Treebank):

```text
distilbert-base-uncased-finetuned-sst-2-english
```

Модель дообучена на бинарной классификации тональности (`finetuned-sst-2`). На вход подаётся англоязычный отзыв. На выходе возвращаются:

- `label`: `POSITIVE` или `NEGATIVE`;
- `score`: уверенность модели в предсказанном классе.

DistilBERT выбрана, поскольку уже предварительно обучена и доступна публично, а результат легко проверять вручную. Обучение не проводилось.

---

## 3. Общая архитектура проекта

Основная архитектура:

```text
Пользователь в браузере
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

Облачная версия:

```text
Пользователь или Gradio Client
        |
        v
+----------------------------------+
| Hugging Face Space               |
| Gradio + Transformers + модель   |
| @spaces.GPU -> ZeroGPU           |
+----------------------------------+
```

Hugging Face Space: https://huggingface.co/spaces/Yabusiki/distilbert-sentiment-analysis.

### 3.1. Структура файлов

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

### 3.2. Организация локальной среды

Проект выполнен на MacBook Air с процессором M1, 8 CPU cores, 8 GB unified memory, ARM64. На macOS Linux-контейнеры не запускаются ядром macOS, поэтому было установлено приложение Docker Desktop. Помимо графического интерфейа, он дает Docker CLI, к которому обращается терминал; Docker daemon, управляющий образами и контейнерами; Linux VM, внутри которой выполняются Linux-контейнеры; и виртуальное хранилище образов, слоёв и контейнеров.

Пока Docker Desktop запущен, команды вводились в Terminal. 

Наличие клиентских инструментов проверено командами:

```bash
docker --version
docker compose version
```

Получены версии:

```text
Docker version 29.9.2
Docker Compose version v5.6.1
```

Затем был выполнен переход в папку проекта:

```bash
cd '/Users/toma/Downloads/STUDY & MASTERS/+UJM/M2 2026:27/M2 1st semester/AI Infrastructure/docker-dl-assignment'
```

Состав сервисов проверен:

```bash
docker compose config --services
```

Результат:

```text
api
ui
```

Эта команда подтверждает, что Compose-файл синтаксически читается и определяет оба сервиса.

---

## 5. Сервис инференса `api`

### 5.1. Назначение

API-контейнер отвечает только за ML-инференс:

1. принимает HTTP-запрос с текстом;
2. проверяет структуру JSON;
3. загружает локально сохранённые tokenizer и модель;
4. запускает Transformers pipeline;
5. возвращает `label` и `score`.

### 5.2. Зависимости `api/requirements.txt`

- **FastAPI (fastapi)** — Python-фреймворк для создания HTTP API, описывает маршруты и данные.
- **Uvicorn (uvicorn[standard])** — ASGI-сервер, который слушает порт и передаёт запросы FastAPI-приложению.
- **Transformers (transformers)** — загрузка tokenizer, модели и ML-pipeline.
- **PyTorch (torch)** — вычислительный backend для нейронной сети.

### 5.3. Предварительная загрузка модели

Файл `api/download_model.py` выполняется во время сборки образа:

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

`AutoTokenizer.from_pretrained` загружает правила токенизации и словарь;
`AutoModelForSequenceClassification.from_pretrained` загружает архитектуру и веса; `save_pretrained` сохраняет оба компонента внутри образа.

Преимущество такого решения — при каждом запуске контейнера не требуется заново скачивать модель из интернета. Цена этого решения — больший размер образа и более долгая первоначальная сборка.

### 5.4. Загрузка модели и предсказание

В `api/app/model.py` определена логика инференса:

```python
@lru_cache(maxsize=1)
def load_model():
    return pipeline(
        "sentiment-analysis",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
    )
```

Pipeline получает локальный путь внутри образа. Декоратор `@lru_cache(maxsize=1)` сохраняет первый созданный pipeline и возвращает тот же объект. Поэтому модель загружается в память один раз, а не для каждого HTTP-запроса.

Функция предсказания:

```python
def predict_sentiment(text: str) -> dict[str, float | str]:
    classifier = load_model()
    result = classifier(text)[0]

    return {
        "label": result["label"],
        "score": float(result["score"]),
    }
```

В ней `load_model()` возвращает закэшированный pipeline; tokenizer переводит текст в token IDs; PyTorch запускает DistilBERT; pipeline формирует список результатов; `[0]` выбирает результат для первого текста; функция возвращает словарь Python.

### 5.5. Схемы данных

`api/app/schemas.py` содержит Pydantic-модели:

```python
class PredictionRequest(BaseModel):
    text: str = Field(min_length=1)

class PredictionResponse(BaseModel):
    label: str
    score: float
```

`PredictionRequest` требует непустое строковое поле `text`. Если JSON имеет неправильную структуру, FastAPI вернёт ошибку валидации до вызова модели. `PredictionResponse` фиксирует формат ответа: строковая метка и числовой score. 

### 5.6. HTTP-маршруты

`api/app/main.py` создаёт приложение:

```python
app = FastAPI(title="DistilBERT Inference API")
```

Маршрут проверки состояния отвечает, что веб-сервис доступен:

```python
@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
```

Маршрут инференса:

```python
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    prediction = predict_sentiment(request.text)
    return PredictionResponse(**prediction)
```

FastAPI преобразует входной JSON в `PredictionRequest`. `request.text` передаётся модели. Выражение `PredictionResponse(**prediction)` разворачивает словарь как именованные аргументы.

### 5.7. Dockerfile API

Dockerfile содержит ключевые инструкции:

- `FROM python:3.11-slim` — базовый Linux-образ с Python 3.11. Вариант `slim` уменьшает базовый слой, исключая системные компоненты, ненужные для проекта.
- `PYTHONDONTWRITEBYTECODE=1` — не создавать файлы `.pyc`.
- `PYTHONUNBUFFERED=1` — выводить логи в поток Docker.
- `HF_HOME=/models` — использовать `/models` для данных Hugging Face.
- `MODEL_DIR` — единый путь, по которому скрипт сохраняет, а приложение загружает модель.
- `WORKDIR /app` — дальнейшие относительные пути отсчитываются от `/app`.
- `COPY requirements.txt .` и отдельный `RUN pip install` — слой зависимостей кэшируется независимо от изменений кода.
- `RUN python download_model.py` — модель становится частью образа.
- `COPY app ./app` выполняется последним среди копирований, поэтому изменение прикладного кода не заставляет заново устанавливать библиотеки и скачивать модель.
- `EXPOSE 8000` документирует внутренний порт.
- `HEALTHCHECK` обращается к `/health` каждые 30 секунд.
- `ENTRYPOINT` запускает Uvicorn и указывает объект `app` из модуля `app.main`.
- `CMD` добавляет адрес и порт. `0.0.0.0` обязателен, чтобы сервер принимал запросы не только изнутри своего процесса.

Порядок слоёв такой что редко меняющиеся зависимости идут раньше часто меняющегося кода.

### 5.8. Сборка

Сборка была запущена командой:

```bash
docker compose build
```

![Начало сборки Docker-образов](report_assets/Screenshot%202026-10-24%20at%2016.25.46.png)

*Рисунок 1. Docker Compose читает два Dockerfile, получает базовый образ `python:3.11-slim` и начинает строить сервисы API и UI.*

![Собранные локальные Docker-образы](report_assets/Screenshot%202026-10-25%20at%2000.24.08.png)

*Рисунок 2. Два локальных образа в Docker Desktop: тяжёлый `distilbert-api` и существенно меньший `distilbert-ui`.*

На скриншоте Docker Desktop показывает `10.35 GB` для API и `681.52 MB` для UI. Возможное будущее улучшение — явно установить CPU-only сборку PyTorch для локального API. Это уменьшило бы образ и расход диска, не замедлив локальный инференс: на Mac NVIDIA GPU контейнеру всё равно не предоставляется.

### 5.9. Прямой запуск API-контейнера

Для создания контейнера через терминал использовался скрипт `scripts/run_api_container.sh`:

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

Параметры:

- `-d` — запуск в фоне;
- `--rm` — удалить контейнер после остановки, не удаляя образ;
- `-p 8000:8000` — порт Mac `8000` перенаправляется на порт `8000` контейнера;
- `--shm-size=1g` — общий сегмент памяти `/dev/shm` увеличен до 1 GiB;
- `--memory=4g` — лимит оперативной памяти;
- `--name` — понятное имя контейнера.

### 5.10. Прямой инференс через API

Compose запрос был выполнен из терминала Mac:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I really enjoyed this assignment."}' \
  | python3 -m json.tool
```

`curl` обращается к порту 8000 хоста. Docker перенаправляет запрос в API-контейнер. Заголовок сообщает, что тело запроса является JSON. Параметр `-d` передаёт поле `text`. Команда `python3 -m json.tool` форматирует полученный JSON.

![Прямой локальный API-инференс](report_assets/Screenshot%202026-10-25%20at%2000.50.16.png)

*Рисунок 3. Успешный POST-запрос к локальному FastAPI: метка `POSITIVE`, score `0.9998620748519897`.*

---

## 6. Графический сервис `ui`

### 6.1. Назначение и принцип изоляции

UI-контейнер отвечает только за пользовательский интерфейс и HTTP-запросы к API. В нём нет PyTorch, Transformers, весов DistilBERT и функции локального ML-инференса. UI отправляет текст по сети и получает уже готовый JSON. ML-операция выполняется в другом контейнере.

### 6.2. Зависимости UI `ui/requirements.txt`

- **Gradio (gradio)** строит веб-интерфейс.
- **Requests (requests)** выполняет HTTP POST к FastAPI.

### 6.3. HTTP-клиент

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

`API_URL` берётся из переменной окружения. Значение по умолчанию — `http://api:8000`, где `api` является именем Compose-сервиса. Docker Compose создаёт общую сеть и встроенный DNS, который преобразует имя `api` во внутренний IP API-контейнера. С Mac используется `http://localhost:8000`.
`requests.post` сериализует `json={"text": text}` и отправляет его на `/predict`. `timeout=30`. `raise_for_status()` превращает HTTP-ответы 4xx/5xx в исключение. После успешного ответа `response.json()` преобразует JSON в словарь Python.

### 6.4. Обработчики интерфейса

Основная функция в `ui/app/main.py`:

```python
def classify_text(text: str) -> str:
    cleaned_text = text.strip()

    if not cleaned_text:
        return "Please enter some text."

    return request_prediction(cleaned_text)
```

`strip()` удаляет пробелы с начала и конца. Если после очистки строка пуста, HTTP-запрос не выполняется. Для непустого текста вызывается клиент FastAPI.

Две вспомогательные функции вставляют готовые отзывы и одновременно выполняют анализ:

```python
def use_positive_review() -> tuple[str, str]:
    return POSITIVE_REVIEW, classify_text(POSITIVE_REVIEW)

def use_negative_review() -> tuple[str, str]:
    return NEGATIVE_REVIEW, classify_text(NEGATIVE_REVIEW)
```

Кортеж содержит два результата: текст для входного поля и предсказание для выходного поля.

### 6.5. Построение Gradio UI

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

`gr.Blocks` задаёт страницу. `Textbox` хранит ввод, `Row` располагает кнопки горизонтально, а второй `Textbox` отображает результат.

Событие кнопки связывает компоненты с Python-функцией:

```python
analyze_button.click(
    fn=classify_text,
    inputs=review_text,
    outputs=prediction_output,
)
```

При клике содержимое `review_text` передаётся в `classify_text`, а возвращённая строка попадает в `prediction_output`. `review_text.submit(...)` позволяет запустить ту же функцию отправкой формы с клавиатуры.

Запуск:

```python
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

`0.0.0.0` позволяет принимать соединения извне контейнера. Если использовать только `127.0.0.1`, сервер будет доступен лишь внутри UI-контейнера.

### 6.6. Dockerfile UI

Dockerfile UI проще API Dockerfile, поскольку не скачивает модель. Команда `python -m app.main` запускает `main.py`.

### 6.7. Проверка GUI

После запуска в браузере был открыт адрес:

```text
http://localhost:7860
```

Введён негативный отзыв:

```text
This product was disappointing and difficult to use
```

![Локальный графический инференс](report_assets/Screenshot%202026-10-25%20at%2000.52.30.png)

*Рисунок 4. Локальный Gradio UI получил от отдельного FastAPI-контейнера результат `NEGATIVE (0.9998)`.*

---

## 7. Оркестрация через Docker Compose

Docker Compose — часть Docker CLI для описания и совместного запуска нескольких сервисов. Конфигурация хранится в YAML-файле `docker-compose.yml`.

### 7.2. Сервис API

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

- `context: ./api` определяет контекст сборки. Docker видит файлы внутри папки `api`.
- `image` задаёт имя собранного образа.
- `container_name` задаёт имя экземпляра.
- `ports` публикует API на Mac.
- `shm_size` увеличивает shared memory.
- `mem_limit` задаёт верхний лимит RAM.

### 7.3. Сервис UI

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

`API_URL` передаёт UI внутренний адрес API. `depends_on` с `condition: service_healthy` создаёт API первым и ждёт положительного Docker health check, потому что запуск процесса ещё не гарантирует, что веб-сервер готов принимать запросы.

UI получает меньшие лимиты, потому что не хранит модель и PyTorch.

### 7.4. Запуск

```bash
docker compose up
```

Compose создал сеть и два контейнера:

![Запуск сервисов через Docker Compose](report_assets/Screenshot%202026-10-25%20at%2000.15.57.png)

*Рисунок 5. API запускает Uvicorn, health check возвращает `200 OK`, после чего UI получает разрешение на запуск.*

### 7.5. Проверка состояния

```bash
docker compose ps
```

Показала состояние контейнеров:

![Состояние контейнеров](report_assets/Screenshot%202026-10-25%20at%2000.46.04.png)

*Рисунок 6. Оба контейнера находятся в состоянии `Up`; API дополнительно имеет статус `healthy`; порты 8000 и 7860 опубликованы на хосте.*

### 7.6. Потребление ресурсов

Одноразовый снимок был получен командой:

```bash
docker stats --no-stream
```

![Статистика ресурсов контейнеров](report_assets/Screenshot%202026-10-25%20at%2001.13.15.png)

*Рисунок 7. Снимок CPU, памяти, сетевого и дискового ввода-вывода и числа процессов для двух контейнеров.*

Наблюдаемые значения:

| Метрика | UI | API | Интерпретация |
|---|---:|---:|---|
| CPU | 7.38% | 0.25% | Моментальный снимок: UI мог обслуживать браузер, а API уже завершил инференс |
| Memory usage | 126.7 MiB | 991.2 MiB | API держит PyTorch, Transformers, tokenizer и веса модели |
| Memory limit | 1 GiB | 3.825 GiB | UI получил свой лимит; API ограничен фактически доступной памятью Docker VM |
| Memory % | 12.37% | 25.31% | Процент считается относительно показанного для контейнера лимита |
| Network I/O | 83.5 kB / 428 kB | 8.2 kB / 4.77 kB | UI обслуживает браузерные ресурсы; API обменивается компактным JSON |
| Block I/O | 126 MB / 12.3 kB | 1.25 GB / 4.1 kB | API прочитал намного больше данных из-за библиотек и весов модели |
| PIDs | 11 | 29 | ML runtime и сервер API используют больше процессов/потоков |

---

## 8. Hugging Face Spaces

### 8.1. Цель облачной части

Дополнительная часть демонстрирует, что модель можно развернуть вне локального компьютера и вызвать через публичный онлайн API. Был создан Space `Yabusiki/distilbert-sentiment-analysis` с SDK Gradio, hardware ZeroGPU и visibility public.

### 8.2. Отличие облачной архитектуры

Локальная часть обязана иметь два контейнера, поэтому UI вызывает FastAPI. В Space интерфейс и модель размещены в одном Python-приложении:

```text
Gradio event -> Python function -> Transformers pipeline -> result
```

### 8.3. Зависимости `hf_space/requirements.txt`

Только `transformers` устанавливается, так как Gradio уже определен SDK-конфигурацией, а  инфраструктуру предоставляет ZeroGPU.

### 8.4. Загрузка pipeline

```python
classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=0,
)
```

Здесь модель указывается по имени Hub, а не по локальному пути. При старте Space были загружены:

- `config.json`;
- `model.safetensors` размером около 268 MB;
- `tokenizer_config.json`;
- `vocab.txt`.

`device=0` означает первый GPU. Pipeline создаётся один раз при старте.

### 8.5. Инференс и ZeroGPU

Эта функция выполняет модель и форматирует результат:

```python
def predict_sentiment(text: str) -> str:
    result = classifier(text)[0]
    return f'{result["label"]} ({float(result["score"]):.4f})'
```

Основной обработчик:

```python
@spaces.GPU(duration=30)
def classify_text(text: str) -> str:
    cleaned_text = text.strip()

    if not cleaned_text:
        return "Please enter some text."

    return predict_sentiment(cleaned_text)
```

`@spaces.GPU(duration=30)` сообщает ZeroGPU, что функции нужен временный GPU. `duration=30` — запрошенный верхний интервал GPU-времени.

### 8.6. Онлайн GUI-инференс

В публичном Space был введён текст:

```text
The application is simple, fast and very useful
```

![Инференс в Hugging Face Space](report_assets/Screenshot%202026-10-25%20at%2011.02.21.png)

*Рисунок 8. Публичный Hugging Face Space работает на ZeroGPU и возвращает `POSITIVE (0.9997)`.*

### 8.10. Онлайн API-инференс из Docker-контейнера

Скрипт `scripts/call_hf_space.py` использует Gradio Client:

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

Имя Space передаётся через переменную окружения. Клиент подключается к опубликованному Space и вызывает endpoint.

```bash
docker run --rm \
  -e HF_SPACE=Yabusiki/distilbert-sentiment-analysis \
  --mount "type=bind,source=$PWD/scripts,target=/scripts,readonly" \
  distilbert-ui:latest \
  python /scripts/call_hf_space.py
```

Параметры означают:

- `--rm` — удалить временный контейнер после вызова;
- `-e HF_SPACE=...` — передать имя Space;
- `type=bind` — подключить существующую папку Mac;
- `source=$PWD/scripts` — источник на хосте;
- `target=/scripts` — путь внутри контейнера;
- `readonly` — запретить контейнеру изменять локальные скрипты;
- `distilbert-ui:latest` — использовать лёгкий образ, где уже установлен Gradio Client как часть Gradio;
- последняя строка запускает API-клиент.

![Удалённый вызов Hugging Face API из Docker-контейнера](report_assets/Screenshot%202026-10-25%20at%2011.09.32.png)

*Рисунок 9. Временный локальный Docker-контейнер подключился к публичному Space и получил `POSITIVE (0.9999)`.*

---

## 9. Сравнение локального и облачного подходов

| Характеристика | Локальные Docker-контейнеры | Hugging Face Space |
|---|---|---|
| Архитектура | Два сервиса: UI и API | Одно Gradio-приложение |
| Изоляция GUI и модели | Явная, через разные контейнеры | Не используется |
| API | FastAPI `/predict` | Gradio `/classify_text` |
| Модель | Встроена в API-образ при build | Скачивается из Hub при запуске среды |
| Вычисления | CPU Docker VM на Mac | ZeroGPU |
| Доступ | `localhost` | Публичный интернет-адрес |
| Оркестрация | Docker Compose | Управляется Hugging Face |
| Контроль ресурсов | `mem_limit`, `shm_size` | Определяется выбранным hardware Space |
| Воспроизводимость | Образы фиксируют среду | README фиксирует Python, Gradio, Transformers |
| Основная ценность | Демонстрация контейнеризации и разделения обязанностей | Демонстрация облачного deployment и онлайн API |

Локальная версия лучше демонстрирует архитектурные принципы Docker: отдельные обязанности сервисов, внутреннюю сеть, health check, порядок запуска и ресурсные ограничения. Облачная версия проще для конечного пользователя, поскольку не требует локальной установки Docker и предоставляет публичную ссылку.

---

## 10. Ограничения и возможные улучшения

### 10.1. CPU-only PyTorch для локального API

Если перестроить локальный API с CPU-only версией PyTorch, это потенциально:

- уменьшит размер образа;
- устранит ненужные CUDA/NVIDIA-библиотеки;
- снизит риск повторного переполнения диска;
- не ухудшит производительность данного локального запуска, поскольку NVIDIA GPU на Mac не использовался.

### 10.2. Закрепление версий зависимостей

Файлы requirements используют имена пакетов без версий. Это удобно на этапе MVP, но одна и та же сборка в будущем может получить новые версии библиотек. Для воспроизводимости можно зафиксировать проверенные версии FastAPI, Uvicorn, Transformers, PyTorch, Gradio и Requests.

---

## 12. Вывод

В результате создано и проверено приложение анализа тональности на основе предобученной DistilBERT. Основная версия: ML-инференс и графический интерфейс изолированы в отдельных Docker-контейнерах и связаны через локальный FastAPI с помощью Docker Compose. Dockerfile организованы с учётом кэширования слоёв, модель сохраняется в API-образ во время сборки, а Compose задаёт health dependency, RAM limits и shared memory.

Прямой запрос к FastAPI вернул `POSITIVE` с уверенностью `0.99986`, а локальный Gradio-интерфейс вернул `NEGATIVE (0.9998)` для негативного текста. Анализ `docker stats` подтвердил ожидаемое разделение ресурсов: inference-контейнер использует больше памяти, дискового ввода-вывода и процессов, а UI-контейнер остаётся относительно лёгким.

Дополнительно приложение опубликовано в Hugging Face Spaces на ZeroGPU. Работоспособность проверена как через публичный интерфейс (`POSITIVE (0.9997)`), так и через удалённый Gradio API, вызванный из временного Docker-контейнера (`POSITIVE (0.9999)`).

Проект продемонстрировал основные принципы развёртывания ML-систем: воспроизводимое окружение, разделение обязанностей сервисов, сетевое взаимодействие, контроль готовности, управление ресурсами и перенос приложения в облачную среду.
