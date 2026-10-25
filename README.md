# InsightCart
 

### AI-Powered Shopping Intelligence Platform with MLOps

**Tagline**

> **Shop Smarter with AI-Driven Product Insights**

---

# Project Overview

**InsightCart** is an AI-powered Chrome Extension and MLOps platform that helps online shoppers make informed purchasing decisions by automatically analyzing product reviews using Natural Language Processing (NLP) and Machine Learning.

Instead of manually reading hundreds or thousands of reviews, users can instantly understand the strengths, weaknesses, overall sentiment, and trustworthiness of a product. The extension extracts reviews from the current shopping page and sends them to an AI backend that performs multiple analyses, including review summarization, fake review detection, topic extraction, sentiment analysis, and personalized buying recommendations.

Unlike a traditional machine learning application, InsightCart is designed as an end-to-end local MLOps project. It includes automated model training, experiment tracking, model versioning, containerized deployment, continuous integration, monitoring, and local Docker orchestration without requiring any cloud credentials or hosted infrastructure.

The goal of the project is to demonstrate the complete lifecycle of developing, deploying, monitoring, and maintaining a machine learning application locally on a MacBook using Docker Compose, MLflow, DVC, Prometheus, and Grafana.

---

# Problem Statement

Online shopping platforms often contain thousands of customer reviews, making it difficult for buyers to quickly understand product quality.

Current problems include:

-  Reading hundreds of reviews takes significant time. 
-  Fake reviews reduce trust in product ratings. 
-  Customers struggle to identify the most common complaints and positive feedback. 
-  Star ratings alone do not accurately reflect user experiences. 
-  Product reviews are often long and repetitive. 

InsightCart addresses these challenges by using Artificial Intelligence and Machine Learning to provide concise, trustworthy, and actionable insights.

---

# Project Objectives

The primary objectives of InsightCart are:

-  Develop an intelligent Chrome Extension for e-commerce websites. 
-  Automatically analyze product reviews. 
-  Detect fake or suspicious reviews. 
-  Generate AI-powered summaries. 
-  Perform sentiment analysis on customer opinions. 
-  Recommend whether a product is worth purchasing. 
-  Build an end-to-end MLOps pipeline. 
-  Run the application locally with Docker Compose. 
-  Track machine learning experiments using MLflow. 
-  Automate local validation with GitHub Actions. 
-  Monitor deployed models and APIs with Prometheus and Grafana. 

---

# Key Features

## Review Summarization

Generate a concise summary of thousands of customer reviews within seconds.

Example:

> Customers appreciate the battery life and display quality, while several users report heating issues during gaming.

---

## Sentiment Analysis

Analyze customer opinions and classify reviews as:

-  Positive 
-  Neutral 
-  Negative 

The extension also displays the percentage distribution of each sentiment.

---

## Common Pros

Automatically identify frequently mentioned positive features.

Example:

-  Excellent Battery 
-  Premium Design 
-  Fast Charging 
-  Smooth Performance 

---

## Common Complaints

Automatically identify frequently mentioned issues.

Example:

-  Heating Problem 
-  Camera Quality 
-  Slow Updates 
-  Poor Speakers 

---

## Fake Review Detection

Detect suspicious reviews using a custom machine learning model trained on publicly available fake-review datasets.

The extension provides:

-  Fake Review Percentage 
-  Trust Score 
-  Genuine Review Ratio 

---

## Topic Extraction

Automatically classify reviews into major topics such as:

-  Battery 
-  Camera 
-  Display 
-  Gaming 
-  Software 
-  Performance 
-  Charging 
-  Build Quality 

---

## AI Buying Recommendation

Generate an intelligent recommendation based on review quality.

Examples:

-  Recommended 
-  Recommended for Students 
-  Good for Gaming 
-  Not Recommended 

---

## Product Trust Score

Generate an overall product quality score between 0 and 100 based on multiple AI signals.

---

## Review Search (Future Enhancement)

Users can ask natural language questions.

Example:

> Does this phone overheat?

The AI searches relevant reviews and generates a concise answer.

---

# Workflow

```
```

```
User Opens Amazon Product Page

        │

Chrome Extension Extracts Product Information

        │

Extract Review Text

        │

Send Reviews to FastAPI Backend

        │

Preprocessing Pipeline

        │

──────────────────────────────────────────
│              AI Pipeline               │
│                                        │
│ Sentiment Analysis                     │
│ Review Summarization                   │
│ Topic Extraction                       │
│ Fake Review Detection                  │
│ Product Recommendation                 │
──────────────────────────────────────────

        │

Generate Final Product Insights

        │

Store Results in PostgreSQL

        │

Return Response to Chrome Extension

        │

Interactive Dashboard Displayed
```

---

# System Architecture

```
```

```
Chrome Extension (React + Manifest V3)

        │

Content Script

        │

Background Service Worker

        │

REST API

        │

FastAPI Backend

        │

────────────────────────────────────────────

Text Preprocessing

↓

Sentiment Model

↓

Summarization Model

↓

Fake Review Detection Model

↓

Recommendation Engine

↓

Database

↓

MLflow

↓

Docker

↓

Monitoring

────────────────────────────────────────────
```

---

# Local Technology Stack

## Frontend

-  React
-  Vite
-  Tailwind CSS
-  JavaScript
-  Chrome Extension Manifest V3

---

## Backend

-  FastAPI
-  Python
-  Pydantic
-  Uvicorn

---

## Database

-  SQLite for the default local app configuration
-  Optional PostgreSQL or Neon database when a project-specific external DB is configured

Stores:

-  Product information
-  Review history
-  User feedback
-  Prediction results
-  Model metadata

---

# Machine Learning Stack

## Pre-trained Models (Hugging Face)

These models are used because they provide high-quality NLP capabilities without requiring large-scale training.

### Review Summarization

Suggested models:

- `facebook/bart-large-cnn` (high quality, heavier) 
- `google/flan-t5-base` or `google/flan-t5-small` (lighter alternatives) 

Purpose:

-  Summarize large collections of reviews into concise paragraphs. 

---

### Sentiment Analysis

Suggested model:

- `cardiffnlp/twitter-roberta-base-sentiment-latest` 

Purpose:

-  Predict Positive, Neutral, or Negative sentiment. 

---

### Embeddings (Optional)

Suggested model:

- `sentence-transformers/all-MiniLM-L6-v2` 

Purpose:

-  Semantic search and future review Q&A. 

---

# Custom Machine Learning Models

Unlike summarization and sentiment, these models will be trained by you.

## Fake Review Detection

Possible algorithms:

-  XGBoost 
-  LightGBM 
-  Random Forest 
-  Logistic Regression 
-  Fine-tuned DistilBERT (advanced) 

Purpose:

-  Predict whether a review is genuine or fake. 

---

## Product Recommendation Model (Optional)

Predict:

-  Buy 
-  Consider 
-  Avoid 

based on review statistics and product features.

---

# Datasets

## Amazon Reviews Dataset

Contains:

-  Review Text 
-  Rating 
-  Product ID 
-  Review Date 
-  Verified Purchase 
-  Helpful Votes 

Uses:

-  Topic extraction 
-  Summarization experiments 
-  Sentiment evaluation 
-  Product analytics 

---

## Fake Review Dataset

Contains:

-  Review Text 
-  Label (Fake/Genuine) 

Used for:

-  Training the fake review detection model. 

---

## Additional Datasets (Optional)

-  Yelp Reviews Dataset 
-  IMDB Review Dataset (for experimentation) 
-  Amazon Customer Reviews (category-specific subsets) 

---

# MLOps Pipeline

The project follows a complete MLOps lifecycle.

### Step 1

Collect and preprocess datasets.

↓

### Step 2

Train multiple ML models.

↓

### Step 3

Track experiments using MLflow.

↓

### Step 4

Compare metrics.

↓

### Step 5

Register the best-performing model.

↓

### Step 6

Package the model using Docker.

↓

### Step 7

Run the local Docker Compose stack on your MacBook.

↓

### Step 8

Monitor production with Prometheus and Grafana.

↓

### Step 9

Collect user feedback.

↓

### Step 10

Retrain models periodically.

---

# MLflow

MLflow is used to manage the machine learning lifecycle.

### Experiment Tracking

Track:

-  Accuracy 
-  Precision 
-  Recall 
-  F1 Score 
-  Training Time 
-  Hyperparameters 

---

### Model Registry

Maintain multiple versions of trained models.

Example:

-  Version 1 
-  Version 2 
-  Version 3 

Rollback to a previous version if necessary.

---

### Artifact Storage

Store:

-  Model files 
-  Evaluation reports 
-  Confusion matrices 
-  Feature importance plots 

---

# Docker

The project runs locally with Docker Compose.

Containers include:

-  FastAPI backend
-  MLflow tracking server
-  Prometheus
-  Grafana
-  Optional local PostgreSQL or external database configuration

Docker Compose orchestrates the local MLOps stack for development and validation.

---

# CI/CD (GitHub Actions)

Whenever code is pushed to GitHub:

1.  Install dependencies.
2.  Run linting.
3.  Execute unit tests.
4.  Validate the Docker configuration.
5.  Build the Docker image locally on the runner.
6.  Report failures clearly without any cloud credentials or deployment steps.

---

# Local Monitoring

## Prometheus

Prometheus scrapes the FastAPI metrics endpoint with the service name and port from the Docker network.

## Grafana

Grafana connects to Prometheus as its datasource and persists dashboards locally in Docker volumes.

## MLflow

MLflow stores experiment metadata and trace data locally in the configured tracking store, which can be SQLite or a local file-backed directory.

---

# Monitoring

## Prometheus

Collect metrics such as:

-  API request count 
-  Request latency 
-  Error rate 
-  Prediction frequency 

i want to make this project so first give dataset to implement train model machine learning 


```
InsightCart/
│
├── app/
│   ├── main.py
│   ├── routes.py
│   ├── schemas.py
│   ├── models.py
│   └── __init__.py
│
├── artifacts/
│   ├── models/
│   │   ├── best_linear_svm_calibrated.pkl
│   │   ├── tfidf_vectorizer.pkl
│   │   └── category_encoder.pkl
│   │
│   ├── reports/
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.png
│   │   └── model_comparison.csv
│   │
│   └── plots/
│       ├── accuracy_comparison.png
│       ├── f1_comparison.png
│       └── roc_auc_comparison.png
│
├── chrome_extension/
│   ├── manifest.json
│   ├── popup/
│   ├── background/
│   ├── content/
│   ├── assets/
│   └── icons/
│
├── config/
│   ├── config.yaml
│   ├── params.yaml
│   └── schema.yaml
│
├── data/
│   ├── raw/
│   │   └── fake_reviews_dataset.csv
│   │
│   ├── processed/
│   │   ├── clean_dataset.csv
│   │   └── preprocessed_dataset.csv
│   │
│   └── external/
│
├── docs/
│   ├── architecture.png
│   ├── workflow.png
│   └── screenshots/
│
├── logs/
│
├── notebooks/
│   └── 01_AI_Generated_Review_Detection.ipynb
│
├── src/
│   │
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── pipeline/
│   │   ├── review_detection_training_pipeline.py
│   │   └── review_detection_inference_pipeline.py
│   │
│   ├── utils/
│   │   ├── common.py
│   │   └── preprocessing.py
│   │
│   ├── constants.py
│   ├── exception.py
│   ├── logger.py
│   └── __init__.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_training.py
│   └── test_prediction.py
│
├── .dvc/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── requirements.txt
├── setup.py
├── README.md
├── LICENSE
└── .gitignore
```

---

## Reproducible Candidate Training

The DVC `train_candidate` stage uses the DVC-tracked raw CSV and writes all
candidate files under `artifacts/experiments/dvc-candidate/`. It does not touch
production model files. Repeated `dvc repro` runs reuse the stage cache while
its dataset and code dependencies remain unchanged. The preprocessing stage
drops incomplete required rows and exact duplicate records before splitting;
the stratified 60/10/10/20 train/validation/promotion/test split fits TF-IDF,
category encoding, and rating scaling on training rows only. Evaluation reports
binary precision, recall, and F1,
accuracy, confusion matrix, dataset SHA-256, and a fingerprint of the held-out
records.

```bash
dvc status
dvc pull data/raw/fake_reviews_dataset.csv.dvc
sha256sum data/raw/fake_reviews_dataset.csv
dvc repro train_candidate
```

For an isolated run without DVC caching, use
`python src/pipeline/run_v2_experiment.py --dataset data/raw/fake_reviews_dataset.csv`.
Candidate selection uses validation F1 with ROC-AUC as a tie-breaker; the
selected candidate is evaluated once on the held-out test split. The test set
is not used for model selection. MLflow logging failures block a successful
evaluation record. Metrics and candidate metadata are stored in the training
experiment, separate from the API tracing experiment.

Promotion is evaluated on a dedicated promotion holdout, separate from both
the candidate-selection validation split and the final test split. The final
test set is reported once and is not used for promotion decisions.
Production promotion is opt-in and requires `--promote`. The active production
model is evaluated on the same promotion rows as the candidate. The gate fails
closed unless all four metrics are available, the evaluation identity matches,
and the candidate strictly improves the configured primary metric. Metric
floors default to `0` and are technical defaults,
not approved business requirements; configure `MODEL_MIN_ACCURACY`,
`MODEL_MIN_PRECISION`, `MODEL_MIN_RECALL`, `MODEL_MIN_F1`,
`MODEL_PRIMARY_METRIC` (default `f1`), and `MODEL_MINIMUM_IMPROVEMENT` after
confirming acceptable false-positive/false-negative costs. A single held-out
score is not a statistical reliability guarantee.

```bash
MODEL_MIN_F1=0.90 MODEL_MINIMUM_IMPROVEMENT=0.01 \
        python src/pipeline/run_v2_experiment.py --promote
```

This command reruns training and only promotes after the gate passes. The API
loads `artifacts/models/production_model_bundle.pkl` when present; verify the
loaded `model_version` in the API startup log. A rejected candidate leaves the
bundle and its checksum unchanged.

After an approved local promotion, record the new production artifact versions
and upload them to the configured DVC remote before sharing that model with a
clean checkout:

```bash
dvc add artifacts/models/production_model_bundle.pkl \
        artifacts/models/best_linear_svm_calibrated.pkl \
        artifacts/models/category_encoder.pkl \
        artifacts/models/tfidf_vectorizer.pkl \
        artifacts/models/rating_scaler.pkl
dvc push
```

The approved bundle is the API activation artifact; the separate DVC update is
the versioned distribution step. Keep the `.dvc` pointer updates in the same
review as the promotion decision.

### Audit and Promotion Commands

Check the current dataset and DVC state, then run the cached candidate stage:

```bash
dvc status
shasum -a 256 data/raw/fake_reviews_dataset.csv
dvc repro train_candidate
```

Inspect candidate and same-holdout production metrics in
`artifacts/experiments/dvc-candidate/model_metadata.json` and
`evaluation_results.json`. Query recent training runs from the local MLflow
database:

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db ./venv/bin/python -c \
        'import mlflow; mlflow.set_tracking_uri("sqlite:///mlflow.db"); e=mlflow.get_experiment_by_name("InsightCart AI Generated Review Detection"); print(mlflow.search_runs([e.experiment_id])[ ["status", "metrics.Accuracy", "metrics.Precision", "metrics.Recall", "metrics.F1 Score", "tags.promotion_decision", "tags.promotion_reason"] ].to_string(index=False) if e else "No training experiment")'
```

To try promotion explicitly, first record the current production bundle checksum,
then run into a separate candidate directory. The command re-evaluates the
active model on the promotion holdout and only activates a candidate that passes
the configured gate:

```bash
shasum -a 256 artifacts/models/production_model_bundle.pkl 2>/dev/null || true
MODEL_MIN_ACCURACY=0 MODEL_MIN_PRECISION=0 MODEL_MIN_RECALL=0 MODEL_MIN_F1=0 \
        MODEL_PRIMARY_METRIC=f1 MODEL_MINIMUM_IMPROVEMENT=0.01 \
        ./venv/bin/python src/pipeline/run_v2_experiment.py \
                --dataset data/raw/fake_reviews_dataset.csv \
                --output-dir artifacts/experiments/manual-candidate --promote
shasum -a 256 artifacts/models/production_model_bundle.pkl 2>/dev/null || true
```

For a rejected candidate, the before/after bundle checksums must match. Confirm
the active version without exposing the serialized model contents:

```bash
./venv/bin/python -c 'import os, joblib; p="artifacts/models/production_model_bundle.pkl"; print(joblib.load(p).get("model_version", "bundle") if os.path.exists(p) else "legacy:" + os.path.basename("artifacts/models/best_linear_svm_calibrated.pkl"))'
```

If no promoted bundle exists, the API uses the legacy DVC-tracked model,
vectorizer, encoder, and scaler files. The API logs `model_name` and
`model_version` when loading a promoted bundle. In CI, training is blocked until the repository has a
reachable `DVC_REMOTE_URL` secret; GitHub Actions has not been run remotely as
part of this local audit.

### Hosted Training Prerequisite

The checked-in DVC remote is local (`../dvc_storage`) and cannot be reached by
GitHub-hosted runners. Push-triggered and manual training require a repository
secret named `DVC_REMOTE_URL` pointing to an accessible DVC remote. Configure
any provider-specific authentication as GitHub secrets/environment variables.
The workflow pulls the tracked raw dataset, runs `dvc repro train_candidate`,
and uploads candidate results; it never promotes or commits model artifacts.
Without that remote configuration, hosted training is intentionally blocked.

## API Tracing

MLflow 3 application traces are stored separately from training runs in the
`InsightCart-API-Tracing` experiment. Run commands from the repository root so
the relative `sqlite:///mlflow.db` URI points to this project's existing
`mlflow.db`. On macOS with Docker Desktop, the host MLflow process must listen
on a Docker-reachable interface, and its Host-header allowlist must include the
Docker Desktop hostname. This does not disable MLflow's security middleware:

```bash
./venv/bin/mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 \
        --allowed-hosts localhost,127.0.0.1,host.docker.internal:5000
```

Use this host binding only on a trusted development machine and restrict access
with the host firewall. Do not expose the tracking server publicly. For local
FastAPI execution, use `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`; Docker
Compose already overrides that with `http://host.docker.internal:5000` for the
API container. Verify the effective setting without dumping the full
environment:

```bash
docker compose exec -T insightcart-api sh -c \
        'printf "MLFLOW_TRACING_ENABLED=%s\\n" "$MLFLOW_TRACING_ENABLED"; case "$MLFLOW_TRACKING_URI" in http://host.docker.internal:5000) echo MLFLOW_TRACKING_URI=expected;; *) echo MLFLOW_TRACKING_URI=unexpected;; esac'
```

The image copies `artifacts/` at build time; it does not pull or mount model
files at runtime. In a clean checkout, hydrate the DVC-managed production model
files before building (with the configured DVC remote credentials available to
the build operator):

```bash
dvc pull artifacts/models/best_linear_svm_calibrated.pkl.dvc \
        artifacts/models/category_encoder.pkl.dvc \
        artifacts/models/tfidf_vectorizer.pkl.dvc
```

Rebuild and start the existing API service after backend source changes:

```bash
docker compose up --build -d insightcart-api
curl -fsS http://localhost:8000/health
```

The extension's actual request is `POST http://localhost:8000/api/v1/review/predict/batch`
with `{ "reviews": [...] }`. Each review includes `review` (3-5000 chars),
`rating` (1-5), and `category`; the extractor also supplies `id`, `title`,
`reviewer`, `date`, `verified_purchase`, and `helpful_votes`, which the request
schema ignores when not declared. To test in Chrome, open DevTools Network,
open a supported Amazon product or review page, open the InsightCart popup, and
click **Analyze Reviews**. Confirm the batch URL, request JSON, and HTTP 200.
The extension service worker console logs the route, status, and returned trace
ID without logging review content. Backend route and trace lifecycle messages
are available via `docker compose logs -f insightcart-api`.

Copy the `X-MLflow-Trace-Id` response header from that request and query the
same host tracking server. The verifier does not create synthetic traces:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
        ./venv/bin/python scripts/verify_mlflow_tracing.py --route batch --trace-id <TRACE_ID>
```

Open the `InsightCart-API-Tracing` experiment's **Traces** view in the MLflow
UI. For a local (non-Docker) API, start it with
`MLFLOW_TRACKING_URI=http://127.0.0.1:5000 ./venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`.
CI disables tracing for ordinary tests so they do not require an external
tracking server.

## Project Steps:

Based on your current progress, you've completed the Machine Learning + Core MLOps pipeline for the first AI feature.

✅ Phase 1: Research & Experimentation (Completed)
Dataset selection
EDA
Data cleaning
Text preprocessing
Feature engineering
Train-test split
Train 7 models
Hyperparameter tuning
Calibrated Linear SVM
MLflow experiment tracking
Notebook (01_AI_Generated_Review_Detection.ipynb)
✅ Phase 2: Project Structure (Completed)
Project folders
src/
artifacts/
config/
utils/
components/
pipeline/
✅ Phase 3: DVC (Completed)
DVC initialized
Dataset tracked
Model tracking
Local DVC storage
✅ Phase 4: Production Code (Completed)
src/
│
├── logger.py
├── exception.py
├── constants.py
│
├── utils/
│   ├── common.py
│   └── preprocessing.py
│
├── components/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_transformation.py
│   ├── model_trainer.py
│   ├── model_evaluation.py
│   └── model_pusher.py
│
└── pipeline/
    └── review_detection_training_pipeline.py

At this point, you have a production training pipeline.

🚀 Phase 5: Model Serving (Next)

This is the next major milestone.

Step 13

Create:

src/pipeline/review_detection_inference_pipeline.py

Purpose:

User Review

↓

Load Model

↓

Load TF-IDF

↓

Load Category Encoder

↓

Clean Text

↓

Create Feature Vector

↓

Predict

↓

Return JSON

This replaces the prediction function you wrote in the notebook.

Step 14

Create:

app/
│
├── main.py
├── schemas.py
├── routes.py
└── services.py

Build a FastAPI backend.

Example endpoints:

POST /predict

POST /health

GET /metrics
Step 15

Move prediction logic from notebook

Notebook:

predict_review(...)

↓

Production:

ReviewDetectionInferencePipeline.predict(...)
Step 16

Test API using Swagger

http://localhost:8000/docs

Example request

{
  "review": "Excellent battery life and premium build quality.",
  "rating": 5,
  "category": "Electronics_5"
}

Example response

{
  "review_type": "Human-Written Review",
  "confidence": 94.63,
  "reason": "The writing style resembles authentic human-written reviews."
}
🚀 Phase 6: Chrome Extension

This is where InsightCart starts becoming your actual product.

chrome-extension/
│
├── manifest.json
├── background.js
├── content.js
├── popup.html
├── popup.js
├── popup.css
├── assets/
└── icons/

Workflow

Amazon Product Page

↓

Extract Reviews

↓

Call FastAPI

↓

Receive Prediction

↓

Show Extension Popup
🚀 Phase 7: Docker

Create

Dockerfile

docker-compose.yml

Run

FastAPI

↓

MLflow

↓

PostgreSQL

↓

Prometheus

↓

Grafana

inside containers.

🚀 Phase 8: CI/CD

GitHub Actions

Lint

↓

Unit Tests

↓

Build Docker

↓

Push Image

🚀 Phase 9: Monitoring

Integrate

Prometheus
Grafana

Monitor:

API latency
Prediction count
Error rate
CPU
Memory
🚀 Phase 11: Expand InsightCart

Once the review detection feature is complete, add more AI capabilities.

1. Sentiment Analysis

Use:

cardiffnlp/twitter-roberta-base-sentiment-latest
2. Review Summarization

Use:

facebook/bart-large-cnn
3. Topic Extraction

Use:

BERTopic
KeyBERT
4. Product Recommendation

Create a rule-based or ML recommendation engine.

5. Trust Score

Combine:

Sentiment
AI Review Detection
Review count
Rating
Verified purchase ratio (if available)
Complete InsightCart Roadmap
Notebook (Research)
        │
        ▼
DVC
        │
        ▼
Production Components
        │
        ▼
Training Pipeline
        │
        ▼
Inference Pipeline
        │
        ▼
FastAPI
        │
        ▼
Chrome Extension
        │
        ▼
Docker
        │
        ▼
GitHub Actions
        │
        ▼
Prometheus + Grafana
        │
        ▼
Sentiment Analysis
        │
        ▼
Review Summarization
        │
        ▼
Topic Extraction
        │
        ▼
Product Recommendation
My recommendation


python -m uvicorn app.main:app --reload