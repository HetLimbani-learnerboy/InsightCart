"""
Project : InsightCart
File    : constants.py
Purpose : Store project constants
"""

import os

PROJECT_NAME = "InsightCart"

RANDOM_STATE = 42

TEST_SIZE = 0.20

TARGET_COLUMN = "label"

TEXT_COLUMN = "text_"

CATEGORY_COLUMN = "category"

RATING_COLUMN = "rating"

CLEAN_TEXT_COLUMN = "clean_review"

RAW_DATA_PATH = os.path.join(
    "data",
    "raw",
    "fake_reviews_dataset.csv"
)

PROCESSED_DATA_PATH = os.path.join(
    "data",
    "processed",
    "preprocessed_dataset.csv"
)

MODEL_DIR = os.path.join(
    "artifacts",
    "models"
)

REPORT_DIR = os.path.join(
    "artifacts",
    "reports"
)

PLOT_DIR = os.path.join(
    "artifacts",
    "plots"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_linear_svm_calibrated.pkl"
)

TFIDF_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "category_encoder.pkl"
)