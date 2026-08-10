# InsightCart
 

### AI-Powered Shopping Intelligence Platform with MLOps

**Tagline**

> **Shop Smarter with AI-Driven Product Insights**

---

# Project Overview

**InsightCart** is an AI-powered Chrome Extension and MLOps platform that helps online shoppers make informed purchasing decisions by automatically analyzing product reviews using Natural Language Processing (NLP) and Machine Learning.

Instead of manually reading hundreds or thousands of reviews, users can instantly understand the strengths, weaknesses, overall sentiment, and trustworthiness of a product. The extension extracts reviews from the current shopping page and sends them to an AI backend that performs multiple analyses, including review summarization, fake review detection, topic extraction, sentiment analysis, and personalized buying recommendations.

Unlike a traditional machine learning application, InsightCart is designed as an end-to-end production-ready MLOps project. It includes automated model training, experiment tracking, model versioning, containerized deployment, continuous integration and deployment (CI/CD), monitoring, and cloud deployment on AWS.

The goal of the project is not only to build an intelligent shopping assistant but also to demonstrate the complete lifecycle of developing, deploying, monitoring, and maintaining machine learning systems in production.

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
-  Demonstrate production deployment using Docker and AWS. 
-  Track machine learning experiments using MLflow. 
-  Automate deployment using GitHub Actions. 
-  Monitor deployed models and APIs. 

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

AWS Deployment

↓

Monitoring

────────────────────────────────────────────
```

---

# Technology Stack

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

-  PostgreSQL 
-  Neon PostgreSQL (Free Tier) 

Stores:

-  Product Information 
-  Review History 
-  User Feedback 
-  Prediction Results 
-  Model Metadata 

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

Deploy to AWS.

↓

### Step 8

Monitor production.

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

Every component will run inside Docker containers.

Containers include:

-  Backend API 
-  PostgreSQL (for local development) 
-  MLflow 
-  Prometheus 
-  Grafana 

Docker Compose will orchestrate all services for local development.

---

# CI/CD (GitHub Actions)

Whenever code is pushed to GitHub:

1.  Install dependencies. 
2.  Run linting. 
3.  Execute unit tests. 
4.  Build Docker images. 
5.  Run integration tests. 
6.  (Learning phase) Push image to Amazon ECR. 
7.  (Learning phase) Deploy to EC2. 
8.  Restart containers automatically. 

After learning AWS deployment, you can disable the deployment step while keeping the workflow file for demonstration.

---

# AWS Services

## Amazon EC2

-  Host the FastAPI backend 
-  Run Docker containers 
-  Serve API requests 

---

## Amazon ECR (Optional)

-  Store Docker images 
-  Pull the latest image during deployment 

---

## Amazon S3 (Optional)

Store:

-  Trained models 
-  Dataset snapshots 
-  Logs 
-  Reports 

---

## AWS IAM

Manage secure access and permissions for deployment.

---

## CloudWatch

Monitor:

-  CPU usage 
-  Memory 
-  Application logs 
-  Error rates 

---

## Security Groups

Restrict access to:

-  API 
-  SSH 
-  Monitoring endpoints 

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
├── notebooks/
│   └── 01_AI_Generated_Review_Detection.ipynb
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
├── mlruns/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── utils/
│   │   ├── common.py
│   │   └── preprocessing.py
│   │
│   ├── logger.py
│   ├── exception.py
│   └── constants.py
│
├── config/
│   ├── config.yaml
│   ├── params.yaml
│   └── schema.yaml
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_training.py
│   └── test_prediction.py
│
├── app/
│   ├── main.py
│   ├── routes.py
│   ├── models.py
│   └── schemas.py
│
├── chrome_extension/
│   ├── manifest.json
│   ├── popup/
│   ├── background/
│   ├── content/
│   └── assets/
│
├── .github/
│   └── workflows/
│       └── ci_cd.yml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
├── dvc.yaml
├── .gitignore
├── README.md
└── LICENSE
```